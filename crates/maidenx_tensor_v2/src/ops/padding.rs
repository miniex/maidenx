use crate::{
    eager, get_mode, insert_metadata, next_tensor_id,
    utils::graph::{add_to_backward_graph, add_to_forward_graph},
    Tensor, TensorId, TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    buffer::BufferManager,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};
use std::sync::Arc;

macro_rules! impl_padding_execute_with_scalar {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target_tid: TensorId,
            target_dtype: DType,
            target_layout: Layout,
            paddings: &[(usize, usize)],
            pad_scalar: Scalar,
        ) -> Result<Self> {
            eager!();

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };

                let metadata = input_tensor.prepare_metadata_for_padding(paddings);

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        target_size,
                        input_tensor.ndim(),
                        Some(&metadata),
                        pad_scalar,
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target_tid, sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));

            crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

            Ok(Tensor(target_tid))
        }
    };
}

macro_rules! impl_padding_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target_tid: TensorId,
            target_dtype: DType,
            target_layout: Layout,
            paddings: &[(usize, usize)],
        ) -> Result<Self> {
            crate::eager!();

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };

                let metadata = input_tensor.prepare_metadata_for_padding(paddings);

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        target_size,
                        input_tensor.ndim(),
                        Some(&metadata),
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target_tid, sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));

            crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

            Ok(Tensor(target_tid))
        }
    };
}

macro_rules! impl_padding_backward {
    ($self:expr, $result_grad:expr, $paddings:expr, $op_name:expr, $backend_fn:path) => {{
        let input_shape = $self.shape();
        let input_strides = $self.strides();
        let input_offset = $self.offset();
        let paddings_vec = $paddings.to_vec();
        let input_shape_clone = input_shape.clone();
        let input_strides_clone = input_strides.clone();

        add_to_backward_graph($op_name, &$result_grad, $self, &input_shape, move |grad_out_id| {
            eager!();
            let grad_out = Tensor(*grad_out_id);

            let target_tid = next_tensor_id();
            let target_layout = Layout::from_shape(&input_shape_clone);
            let metadata = TensorMetadata {
                device: grad_out.device(),
                dtype: grad_out.dtype(),
                layout: target_layout.clone(),
                grad_tensor_id: None,
                graph_id: None,
                mode: get_mode(),
                update_status: TensorUpdateStatus::Pending,
            };
            insert_metadata(target_tid, metadata);

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, grad_out.device(), grad_out.dtype())?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let metadata = Tensor::prepare_metadata_for_padding_backward(
                    &input_shape_clone,
                    &input_strides_clone,
                    input_offset,
                    &paddings_vec,
                );

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        grad_out.storage()?.buffer(),
                        target_size,
                        grad_out.size(),
                        grad_out.ndim(),
                        Some(&metadata),
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target_tid, sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));
            crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

            Ok(Tensor(target_tid))
        })?;
    }};
}

/// ## Tensor Padding Operations
///
/// This `impl` block provides methods for applying padding to tensors.
/// Padding adds extra elements around the edges of a tensor, which is commonly
/// used in neural networks for convolution operations to maintain spatial dimensions.
///
/// There are several main padding modes:
///
/// * **Constant padding**: `pad`, `pad_with_constant` - fills with a constant value
/// * **Reflection padding**: `pad_with_reflection` - mirrors values at the border
/// * **Replication padding**: `pad_with_replication` - repeats edge values
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Key features:**
/// * **Multiple padding modes**: Support for constant, reflection, and replication padding
/// * **Dimension validation**: Ensures padding specification matches tensor dimensions
/// * **Type promotion**: Boolean tensors are promoted to U8 before padding
/// * **Device compatibility**: Operations maintain tensor device placement
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct computation
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **Autograd behavior:**
/// When the input tensor has `requires_grad()` set to true, the result tensor will:
/// - Automatically enable gradient computation
/// - Register backward functions for gradient computation using unpadding operations
/// - Maintain the computational graph for backpropagation
///
/// **Padding specification:**
/// * Paddings are specified as `&[(before, after)]` for each dimension
/// * Each tuple `(before, after)` specifies padding before and after the dimension
/// * The number of padding pairs must match the tensor's number of dimensions
///
/// **Gradient computation:**
/// - **All padding modes**: Gradients flow back by removing the padding from the output gradient
/// - The backward pass extracts the original tensor shape from the padded gradient
/// - Gradient computation is independent of the padding mode used in the forward pass
impl Tensor {
    /// Runs [`try_pad`](Self::try_pad) and panics on failure.
    ///
    /// Applies constant padding to the tensor with the specified value.
    /// This is an alias for `pad_with_constant`.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow back by removing the padding from the output gradient.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]); // Shape [2, 2]
    /// let padded = tensor.pad(&[(1, 1), (1, 1)], 0.0);
    /// // Result shape [4, 4] with 0.0 padding around the original values
    /// ```
    ///
    /// # Panics
    ///
    /// * When the number of padding pairs doesn't match tensor dimensions
    /// * When memory allocation fails
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn pad(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Tensor {
        self.try_pad(paddings, pad_value).expect("failed to pad tensor")
    }

    /// Runs [`try_pad_with_constant`](Self::try_pad_with_constant) and panics on failure.
    ///
    /// Applies constant padding to the tensor, filling new elements with a constant value.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow back by removing the padding from the output gradient.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0]); // Shape [3]
    /// let padded = tensor.pad_with_constant(&[(2, 1)], -1.0);
    /// // Result: [-1.0, -1.0, 1.0, 2.0, 3.0, -1.0] with shape [6]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the number of padding pairs doesn't match tensor dimensions
    /// * When memory allocation fails
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn pad_with_constant(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Tensor {
        self.try_pad_with_constant(paddings, pad_value)
            .expect("failed to pad tensor with constant")
    }

    /// Runs [`try_pad_with_reflection`](Self::try_pad_with_reflection) and panics on failure.
    ///
    /// Applies reflection padding to the tensor, mirroring values at the borders.
    /// The padding values are reflected across the boundary without repeating the edge values.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow back by removing the padding from the output gradient.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]); // Shape [4]
    /// let padded = tensor.pad_with_reflection(&[(2, 1)]);
    /// // Result: [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0] with shape [7]
    /// // Reflects: [3,2] + [1,2,3,4] + [3]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the number of padding pairs doesn't match tensor dimensions
    /// * When padding width is >= dimension size (reflection requires sufficient data)
    /// * When trying to reflect on dimensions with size <= 1
    /// * When memory allocation fails
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn pad_with_reflection(&self, paddings: &[(usize, usize)]) -> Tensor {
        self.try_pad_with_reflection(paddings)
            .expect("failed to pad tensor with reflection")
    }

    /// Runs [`try_pad_with_replication`](Self::try_pad_with_replication) and panics on failure.
    ///
    /// Applies replication padding to the tensor, repeating the edge values.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow back by removing the padding from the output gradient.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]); // Shape [4]
    /// let padded = tensor.pad_with_replication(&[(2, 1)]);
    /// // Result: [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0] with shape [7]
    /// // Replicates: [1,1] + [1,2,3,4] + [4]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the number of padding pairs doesn't match tensor dimensions
    /// * When memory allocation fails
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn pad_with_replication(&self, paddings: &[(usize, usize)]) -> Tensor {
        self.try_pad_with_replication(paddings)
            .expect("failed to pad tensor with replication")
    }

    /// Attempts to apply constant padding to the tensor.
    ///
    /// This is an alias for [`try_pad_with_constant`](Self::try_pad_with_constant).
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow back by removing the padding from the output gradient.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn pad_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    ///     let padded = tensor.try_pad(&[(1, 1), (0, 1)], 0.0)?;
    ///     // Adds 1 row above and below, 1 column on the right
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of padding pairs doesn't match tensor dimensions
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_pad(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor> {
        self.try_pad_with_constant(paddings, pad_value)
    }

    /// Attempts to apply constant padding to the tensor.
    ///
    /// Fills new elements created by padding with a constant value.
    /// Boolean tensors are automatically promoted to U8 before padding.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// The backward pass removes padding from the output gradient, extracting
    /// the gradient corresponding to the original tensor shape.
    ///
    /// # Parameters
    /// * `paddings`: Array of `(before, after)` pairs for each dimension
    /// * `pad_value`: The constant value to fill new elements with
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn constant_pad_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let padded = tensor.try_pad_with_constant(&[(2, 1)], -1.0)?;
    ///     // Result: [-1.0, -1.0, 1.0, 2.0, 3.0, -1.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of padding pairs doesn't match tensor dimensions
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    ///
    /// # Gradient Flow
    /// During backpropagation, the gradient tensor (which has the padded shape)
    /// is "unpadded" to extract only the gradient values corresponding to the
    /// original input tensor positions. The padding values in the gradient are discarded.
    pub fn try_pad_with_constant(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let mut output_shape = Vec::with_capacity(self.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(self.shape()[i] + pad_before + pad_after);
        }

        let target_layout = Layout::from_shape(&output_shape);
        let pad_scalar = pad_value.into();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_pad_with_constant(target_tid, target_dtype, target_layout, paddings, pad_scalar)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let paddings_vec = paddings.to_vec();
                let result = add_to_forward_graph(
                    "pad_with_constant",
                    &[self],
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let input_tensor = Tensor(input_tids[0]);
                        let result = input_tensor.execute_pad_with_constant(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            &paddings_vec,
                            pad_scalar,
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            impl_padding_backward!(
                self,
                result_grad,
                paddings,
                "pad_with_constant_backward",
                maidenx_core::be::ops::padding::pad_with_constant_backward
            );
        }

        Ok(result)
    }

    /// Attempts to apply reflection padding to the tensor.
    ///
    /// Mirrors values across the tensor boundaries without repeating edge values.
    /// This creates a symmetric reflection where the edge values are not duplicated.
    /// Boolean tensors are automatically promoted to U8 before padding.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// The backward pass removes padding from the output gradient, summing gradients
    /// that were reflected to multiple positions back to their original positions.
    ///
    /// # Parameters
    /// * `paddings`: Array of `(before, after)` pairs for each dimension
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn reflection_pad_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let padded = tensor.try_pad_with_reflection(&[(2, 1)])?;
    ///     // Result: [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0]
    ///     // Reflects: [3,2] + [1,2,3,4] + [3]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of padding pairs doesn't match tensor dimensions
    /// - Padding width is >= dimension size (not enough data to reflect)
    /// - Trying to pad dimensions with size <= 1 (insufficient data for reflection)
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    ///
    /// # Constraints
    /// * For reflection padding, each dimension must have size > 1 for non-zero padding
    /// * Padding width must be less than the dimension size
    ///
    /// # Gradient Flow
    /// During backpropagation, gradients from reflected positions are accumulated
    /// back to their original positions. For example, if a value was reflected to
    /// multiple padding positions, the gradients from all those positions are summed
    /// and assigned to the original position.
    pub fn try_pad_with_reflection(&self, paddings: &[(usize, usize)]) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            let dim_size = self.shape()[i];

            if pad_before >= dim_size || pad_after >= dim_size {
                return Err(Error::InvalidArgument(format!(
                    "Reflection padding width ({}, {}) must be less than the dimension size ({})",
                    pad_before, pad_after, dim_size
                )));
            }

            if dim_size <= 1 && (pad_before > 0 || pad_after > 0) {
                return Err(Error::InvalidArgument(format!(
                    "Reflection padding requires input dimension > 1 for non-zero padding, but got dim {} with size {}",
                    i, dim_size
                )));
            }
        }

        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let mut output_shape = Vec::with_capacity(self.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(self.shape()[i] + pad_before + pad_after);
        }

        let target_layout = Layout::from_shape(&output_shape);

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_pad_with_reflection(target_tid, target_dtype, target_layout, paddings)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let paddings_vec = paddings.to_vec();
                let result = add_to_forward_graph(
                    "pad_with_reflection",
                    &[self],
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let input_tensor = Tensor(input_tids[0]);
                        let result = input_tensor.execute_pad_with_reflection(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            &paddings_vec,
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            impl_padding_backward!(
                self,
                result_grad,
                paddings,
                "pad_with_reflection_backward",
                maidenx_core::be::ops::padding::pad_with_reflection_backward
            );
        }

        Ok(result)
    }

    /// Attempts to apply replication padding to the tensor.
    ///
    /// Repeats the edge values of the tensor to fill the padding areas.
    /// This is also known as "edge" padding where boundary values are replicated.
    /// Boolean tensors are automatically promoted to U8 before padding.
    ///
    /// **Autograd support**: If the tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// The backward pass removes padding from the output gradient, summing gradients
    /// from replicated edge positions back to their original edge positions.
    ///
    /// # Parameters
    /// * `paddings`: Array of `(before, after)` pairs for each dimension
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn replication_pad_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let padded = tensor.try_pad_with_replication(&[(2, 1)])?;
    ///     // Result: [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
    ///     // Replicates: [1,1] + [1,2,3,4] + [4]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of padding pairs doesn't match tensor dimensions
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    ///
    /// # Notes
    /// * Replication padding works with any tensor size, including single-element dimensions
    /// * Edge values are simply repeated to fill the padding areas
    /// * This is the most permissive padding mode in terms of input constraints
    ///
    /// # Gradient Flow
    /// During backpropagation, gradients from all replicated positions of an edge value
    /// are summed and assigned back to the original edge position. For example, if an
    /// edge value was replicated 3 times, the gradients from all 3 positions are
    /// accumulated and assigned to the original edge position.
    pub fn try_pad_with_replication(&self, paddings: &[(usize, usize)]) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let mut output_shape = Vec::with_capacity(self.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(self.shape()[i] + pad_before + pad_after);
        }

        let target_layout = Layout::from_shape(&output_shape);

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_pad_with_replication(target_tid, target_dtype, target_layout, paddings)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let paddings_vec = paddings.to_vec();
                let result = add_to_forward_graph(
                    "pad_with_replication",
                    &[self],
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let input_tensor = Tensor(input_tids[0]);
                        let result = input_tensor.execute_pad_with_replication(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            &paddings_vec,
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            impl_padding_backward!(
                self,
                result_grad,
                paddings,
                "pad_with_replication_backward",
                maidenx_core::be::ops::padding::pad_with_replication_backward
            );
        }

        Ok(result)
    }

    impl_padding_execute_with_scalar!(
        execute_pad_with_constant,
        maidenx_core::be::ops::padding::pad_with_constant
    );
    impl_padding_execute!(
        execute_pad_with_reflection,
        maidenx_core::be::ops::padding::pad_with_reflection
    );
    impl_padding_execute!(
        execute_pad_with_replication,
        maidenx_core::be::ops::padding::pad_with_replication
    );

    fn prepare_metadata_for_padding(&self, paddings: &[(usize, usize)]) -> Vec<usize> {
        let mut info = Vec::new();

        let input_shape = self.shape();
        let input_strides = self.strides();

        info.extend_from_slice(&input_shape);
        info.extend_from_slice(&input_strides);
        info.push(self.offset());

        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            info.push(input_shape[i] + pad_before + pad_after);
        }

        for &(pad_before, pad_after) in paddings {
            info.push(pad_before);
            info.push(pad_after);
        }

        info
    }

    fn prepare_metadata_for_padding_backward(
        input_shape: &[usize],
        input_strides: &[usize],
        input_offset: usize,
        paddings: &[(usize, usize)],
    ) -> Vec<usize> {
        let mut info = Vec::new();

        info.extend_from_slice(input_shape);
        info.extend_from_slice(input_strides);
        info.push(input_offset);

        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            info.push(input_shape[i] + pad_before + pad_after);
        }

        for &(pad_before, pad_after) in paddings {
            info.push(pad_before);
            info.push(pad_after);
        }

        info
    }
}
