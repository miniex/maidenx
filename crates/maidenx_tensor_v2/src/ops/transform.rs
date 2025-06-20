use crate::{
    eager, get_mode, insert_metadata, link_tensor_to_storage, next_tensor_id,
    utils::graph::{add_to_backward_graph, add_to_forward_graph},
    Tensor, TensorId, TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};

/// ## Tensor shape transformation helpers
///
/// This `impl` block provides methods for changing tensor shapes and layouts.
/// There are two main categories:
///
/// * **Shape transformation**: `view`, `reshape` - changes tensor dimensions
/// * **Layout operations**: Operations that modify memory layout without copying data
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct layout modification
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **Performance notes:**
/// * View operations are zero-copy when possible (contiguous tensors)
/// * Reshape operations may require contiguous conversion for non-contiguous tensors
/// * Layout operations only modify metadata, not actual data
impl Tensor {
    /// Runs [`try_view`](Self::try_view) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[6]);
    /// let reshaped = tensor.view(&[2, 3]);
    /// assert_eq!(reshaped.shape(), vec![2, 3]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When the new shape is incompatible with tensor size
    /// * When tensor is not contiguous and cannot be viewed
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Self {
        self.try_view(shape).expect("failed to create view")
    }

    /// Runs [`try_squeeze`](Self::try_squeeze) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[1, 3, 1, 4]);
    /// let squeezed = tensor.squeeze(&[0, 2]);
    /// assert_eq!(squeezed.shape(), vec![3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When specified dimensions are not of size 1
    /// * When dimension indices are out of bounds
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn squeeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Self {
        self.try_squeeze(dims).expect("failed to squeeze tensor")
    }

    /// Runs [`try_squeeze_all`](Self::try_squeeze_all) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[1, 3, 1, 4, 1]);
    /// let squeezed = tensor.squeeze_all();
    /// assert_eq!(squeezed.shape(), vec![3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn squeeze_all(&self) -> Self {
        self.try_squeeze_all().expect("failed to squeeze all dimensions")
    }

    /// Runs [`try_unsqueeze`](Self::try_unsqueeze) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[3, 4]);
    /// let unsqueezed = tensor.unsqueeze(&[0, 2]);
    /// assert_eq!(unsqueezed.shape(), vec![1, 3, 1, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When dimension indices are out of bounds
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn unsqueeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Self {
        self.try_unsqueeze(dims).expect("failed to unsqueeze tensor")
    }

    /// Runs [`try_transpose`](Self::try_transpose) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[2, 3, 4]);
    /// let transposed = tensor.transpose(0, 2);
    /// assert_eq!(transposed.shape(), vec![4, 3, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When dimensions are out of bounds
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn transpose(&self, dim0: impl Into<Scalar>, dim1: impl Into<Scalar>) -> Self {
        self.try_transpose(dim0, dim1).expect("failed to transpose tensor")
    }

    /// Runs [`try_slice`](Self::try_slice) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[5, 4]);
    /// let sliced = tensor.slice(0, 1, Some(4), 1);
    /// assert_eq!(sliced.shape(), vec![3, 4]); // rows 1,2,3
    /// ```
    ///
    /// # Panics
    ///
    /// * When dimension is out of bounds
    /// * When slice indices are invalid
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn slice(
        &self,
        dim: impl Into<Scalar>,
        start: impl Into<Scalar>,
        end: Option<impl Into<Scalar>>,
        step: impl Into<Scalar>,
    ) -> Self {
        self.try_slice(dim, start, end, step).expect("failed to slice tensor")
    }

    /// Runs [`try_unfold`](Self::try_unfold) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[5]);
    /// let unfolded = tensor.unfold(0, 3, 1);
    /// assert_eq!(unfolded.shape(), vec![3, 3]); // sliding window of size 3
    /// ```
    ///
    /// # Panics
    ///
    /// * When dimension is out of bounds
    /// * When unfold parameters are invalid
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn unfold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Self {
        self.try_unfold(dim, size, step).expect("failed to unfold tensor")
    }

    /// Runs [`try_broadcast`](Self::try_broadcast) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[3, 1]);
    /// let broadcasted = tensor.broadcast(&[3, 4]);
    /// assert_eq!(broadcasted.shape(), vec![3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When shapes are incompatible for broadcasting
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn broadcast(&self, shape: &[usize]) -> Self {
        self.try_broadcast(shape).expect("failed to broadcast tensor")
    }

    /// Runs [`try_broadcast_like`](Self::try_broadcast_like) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[3, 1]);
    /// let other = Tensor::zeros(&[3, 4]);
    /// let broadcasted = tensor.broadcast_like(&other);
    /// assert_eq!(broadcasted.shape(), vec![3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When shapes are incompatible for broadcasting
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn broadcast_like(&self, other: &Self) -> Self {
        self.try_broadcast_like(other)
            .expect("failed to broadcast tensor like other")
    }

    /// Runs [`try_broadcast_left`](Self::try_broadcast_left) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[3, 4]);
    /// let broadcasted = tensor.broadcast_left(&[2, 5]);
    /// assert_eq!(broadcasted.shape(), vec![2, 5, 3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When broadcast operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn broadcast_left(&self, batch_dims: &[usize]) -> Self {
        self.try_broadcast_left(batch_dims)
            .expect("failed to broadcast tensor left")
    }

    /// Runs [`try_reshape`](Self::try_reshape) and panics on failure.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[6]);
    /// let reshaped = tensor.reshape(&[2, 3]);
    /// assert_eq!(reshaped.shape(), vec![2, 3]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When the new shape is incompatible with tensor size
    /// * When contiguous conversion fails for non-contiguous tensors
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Self {
        self.try_reshape(shape).expect("failed to reshape tensor")
    }

    /// Attempts to create a view of the tensor with a new shape.
    ///
    /// This operation creates a new tensor that shares the same storage but with
    /// a different shape. The total number of elements must remain the same.
    /// This is a zero-copy operation when the tensor is contiguous.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn create_view() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[6]);
    ///     let view = tensor.try_view(&[2, 3])?;
    ///     assert_eq!(view.shape(), vec![2, 3]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The new shape is incompatible with the tensor's total size
    /// - The tensor is not contiguous and cannot be viewed directly
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * This is a zero-copy operation that only modifies metadata
    /// * Requires the tensor to be contiguous for the view to be valid
    /// * Much faster than reshape as no data copying is involved
    pub fn try_view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let mut new_layout = self.layout();
        new_layout.view(&computed_shape)?;

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "view",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if self.requires_grad() {
                let shape = self.shape();
                let shape_clone = shape.clone();
                add_to_backward_graph("view_backward", &result_grad, self, &shape, move |grad_out_id| {
                    eager!();
                    Tensor(*grad_out_id).try_reshape(&shape_clone)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to squeeze dimensions of size 1 at the specified positions.
    ///
    /// This operation removes dimensions of size 1 from the tensor at the specified
    /// dimension indices. All specified dimensions must have size 1.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn squeeze_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[1, 3, 1, 4]);
    ///     let squeezed = tensor.try_squeeze(&[0, 2])?;
    ///     assert_eq!(squeezed.shape(), vec![3, 4]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any specified dimension is not of size 1
    /// - Any dimension index is out of bounds
    /// - Graph operations fail in lazy mode
    pub fn try_squeeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Result<Self> {
        let current_shape = self.shape();
        let rank = current_shape.len();

        let mut normalized_dims = Vec::new();
        for dim in dims {
            let dim_i32 = dim.clone().into().as_i32();
            let normalized_dim: usize = if dim_i32 < 0 {
                (rank as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };

            if normalized_dim >= rank {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Dimension {} is out of bounds for tensor with {} dimensions",
                        normalized_dim, rank
                    ),
                });
            }

            if current_shape[normalized_dim] != 1 {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Cannot squeeze dimension {} with size {}",
                        normalized_dim, current_shape[normalized_dim]
                    ),
                });
            }

            normalized_dims.push(normalized_dim);
        }

        let mut new_shape = Vec::new();
        normalized_dims.sort_unstable();
        normalized_dims.dedup();

        for (i, &size) in current_shape.iter().enumerate() {
            if !normalized_dims.contains(&i) {
                new_shape.push(size);
            }
        }

        let mut new_layout = self.layout();
        new_layout.view(&new_shape)?;

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "squeeze",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if self.requires_grad() {
                let shape = self.shape();
                let shape_clone = shape.clone();
                add_to_backward_graph("squeeze_backward", &result_grad, self, &shape, move |grad_out_id| {
                    eager!();
                    Tensor(*grad_out_id).try_reshape(&shape_clone)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to squeeze all dimensions of size 1.
    ///
    /// This operation removes all dimensions of size 1 from the tensor.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn squeeze_all_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[1, 3, 1, 4, 1]);
    ///     let squeezed = tensor.try_squeeze_all()?;
    ///     assert_eq!(squeezed.shape(), vec![3, 4]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Graph operations fail in lazy mode
    pub fn try_squeeze_all(&self) -> Result<Self> {
        let current_shape = self.shape();
        let mut new_shape = Vec::new();

        for &size in &current_shape {
            if size != 1 {
                new_shape.push(size);
            }
        }

        if new_shape.len() == current_shape.len() {
            return self.try_view(&current_shape);
        }

        let mut new_layout = self.layout();
        new_layout.view(&new_shape)?;

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "squeeze_all",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if self.requires_grad() {
                let shape = self.shape();
                let shape_clone = shape.clone();
                add_to_backward_graph("squeeze_all_backward", &result_grad, self, &shape, move |grad_out_id| {
                    eager!();
                    Tensor(*grad_out_id).try_reshape(&shape_clone)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to add dimensions of size 1 at the specified positions.
    ///
    /// This operation inserts new dimensions of size 1 at the specified indices.
    /// The dimension indices are relative to the final shape after all insertions.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn unsqueeze_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[3, 4]);
    ///     let unsqueezed = tensor.try_unsqueeze(&[0, 2])?;
    ///     assert_eq!(unsqueezed.shape(), vec![1, 3, 1, 4]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any dimension index is out of bounds for the target shape
    /// - Graph operations fail in lazy mode
    pub fn try_unsqueeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Result<Self> {
        let current_shape = self.shape();
        let target_rank = current_shape.len() + dims.len();

        let mut normalized_dims = Vec::new();
        for dim in dims {
            let dim_i32 = dim.clone().into().as_i32();
            let normalized_dim: usize = if dim_i32 < 0 {
                (target_rank as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };

            if normalized_dim >= target_rank {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Dimension {} is out of bounds for target tensor with {} dimensions",
                        normalized_dim, target_rank
                    ),
                });
            }

            normalized_dims.push(normalized_dim);
        }

        let mut new_shape = vec![1; target_rank];
        normalized_dims.sort_unstable();
        normalized_dims.dedup();

        let mut src_idx = 0;
        for i in 0..target_rank {
            if !normalized_dims.contains(&i) {
                if src_idx >= current_shape.len() {
                    return Err(Error::InvalidShape {
                        message: "Internal error: source index out of bounds".to_string(),
                    });
                }
                new_shape[i] = current_shape[src_idx];
                src_idx += 1;
            }
        }

        let mut new_layout = self.layout();
        new_layout.view(&new_shape)?;

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "unsqueeze",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if self.requires_grad() {
                let shape = self.shape();
                let shape_clone = shape.clone();
                add_to_backward_graph("unsqueeze_backward", &result_grad, self, &shape, move |grad_out_id| {
                    eager!();
                    Tensor(*grad_out_id).try_reshape(&shape_clone)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to transpose two dimensions of the tensor.
    ///
    /// This operation swaps the specified dimensions of the tensor.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn transpose_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[2, 3, 4]);
    ///     let transposed = tensor.try_transpose(0, 2)?;
    ///     assert_eq!(transposed.shape(), vec![4, 3, 2]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either dimension is out of bounds
    /// - Graph operations fail in lazy mode
    pub fn try_transpose(&self, dim0: impl Into<Scalar>, dim1: impl Into<Scalar>) -> Result<Self> {
        let rank = self.shape().len();

        let dim0_i32 = dim0.into().as_i32();
        let dim1_i32 = dim1.into().as_i32();

        let dim0_normalized: usize = if dim0_i32 < 0 {
            (rank as i32 + dim0_i32) as usize
        } else {
            dim0_i32 as usize
        };

        let dim1_normalized: usize = if dim1_i32 < 0 {
            (rank as i32 + dim1_i32) as usize
        } else {
            dim1_i32 as usize
        };

        if dim0_normalized >= rank {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim0_normalized, rank
                ),
            });
        }
        if dim1_normalized >= rank {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim1_normalized, rank
                ),
            });
        }

        let mut new_layout = self.layout();
        new_layout.transpose(dim0_normalized, dim1_normalized)?;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "transpose",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to slice the tensor along a specified dimension.
    ///
    /// This operation creates a view into a subset of the tensor along the specified
    /// dimension. The slice is defined by start, end, and step parameters.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn slice_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[5, 4]);
    ///     let sliced = tensor.try_slice(0, 1, Some(4), 1)?;
    ///     assert_eq!(sliced.shape(), vec![3, 4]); // rows 1,2,3
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dimension index is out of bounds
    /// - The slice parameters are invalid
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * This is a zero-copy operation that only modifies layout metadata
    /// * No data copying is involved, making it very efficient
    /// * The resulting tensor shares storage with the original
    pub fn try_slice(
        &self,
        dim: impl Into<Scalar>,
        start: impl Into<Scalar>,
        end: Option<impl Into<Scalar>>,
        step: impl Into<Scalar>,
    ) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let start_i32 = start.into().as_i32();
        let end_i32 = end.map(|e| e.into().as_i32());
        let step_i32 = step.into().as_i32();

        let dim_normalized: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim_normalized >= self.ndim() {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim_normalized,
                    self.ndim()
                ),
            });
        }

        let new_layout = self.layout().slice(
            dim_normalized,
            start_i32 as isize,
            end_i32.map(|e| e as isize),
            step_i32 as isize,
        )?;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "slice",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to unfold the tensor along a specified dimension.
    ///
    /// This operation extracts sliding windows from the tensor along the specified
    /// dimension. It's commonly used for implementing convolution-like operations
    /// or extracting patches from data.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn unfold_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[5]);
    ///     let unfolded = tensor.try_unfold(0, 3, 1)?;
    ///     assert_eq!(unfolded.shape(), vec![3, 3]); // 3 windows of size 3
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dimension index is out of bounds
    /// - The size or step parameters are invalid
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * This is a zero-copy operation that only modifies layout metadata
    /// * Creates a view with overlapping windows into the original data
    /// * Very efficient for sliding window operations
    pub fn try_unfold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let size_i32 = size.into().as_i32();
        let step_i32 = step.into().as_i32();

        let dim_usize: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim_usize >= self.ndim() {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim_usize,
                    self.ndim()
                ),
            });
        }

        if size_i32 <= 0 {
            return Err(Error::InvalidShape {
                message: format!("Unfold size must be positive, got {}", size_i32),
            });
        }

        if step_i32 <= 0 {
            return Err(Error::InvalidShape {
                message: format!("Unfold step must be positive, got {}", step_i32),
            });
        }

        let new_layout = self.layout().unfold(dim_usize, size_i32 as usize, step_i32 as usize)?;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "unfold",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to broadcast the tensor to a new shape.
    ///
    /// This operation creates a new tensor with the specified shape by broadcasting
    /// the current tensor. Broadcasting allows tensors with different shapes to be
    /// used together in operations by virtually expanding dimensions of size 1.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn broadcast_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[3, 1]);
    ///     let broadcasted = tensor.try_broadcast(&[3, 4])?;
    ///     assert_eq!(broadcasted.shape(), vec![3, 4]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The shapes are incompatible for broadcasting
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * This is a zero-copy operation that only modifies layout metadata
    /// * No actual data copying is involved
    /// * The resulting tensor shares storage with the original
    pub fn try_broadcast(&self, shape: &[usize]) -> Result<Self> {
        if self.shape() == shape {
            return Ok(self.clone());
        }

        let old_shape = self.shape();

        if old_shape.is_empty() {
            return self.try_broadcast_scalar_to(shape);
        }

        if shape.is_empty() {
            if !old_shape.is_empty() {
                return Err(Error::InvalidShape {
                    message: format!("Cannot broadcast non-scalar shape {:?} to scalar ()", old_shape),
                });
            } else {
                return Ok(self.clone());
            }
        }

        let new_layout = self.layout().broadcast_to(shape)?;

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: new_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "broadcast",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[new_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if self.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if self.requires_grad() {
                let shape = self.shape();
                let shape_clone = shape.clone();
                add_to_backward_graph("broadcast_backward", &result_grad, self, &shape, move |grad_out_id| {
                    eager!();
                    Tensor(*grad_out_id).try_sum_to_shape(&shape_clone)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to broadcast the tensor to match another tensor's shape.
    ///
    /// This is a convenience method that calls `try_broadcast` with the shape
    /// of the provided tensor.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn broadcast_like_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[3, 1]);
    ///     let other = Tensor::zeros(&[3, 4]);
    ///     let broadcasted = tensor.try_broadcast_like(&other)?;
    ///     assert_eq!(broadcasted.shape(), other.shape());
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The shapes are incompatible for broadcasting
    /// - Graph operations fail in lazy mode
    pub fn try_broadcast_like(&self, other: &Self) -> Result<Self> {
        self.try_broadcast(&other.shape())
    }

    /// Attempts to broadcast the tensor by adding batch dimensions on the left.
    ///
    /// This operation prepends the specified dimensions to the tensor's shape
    /// and broadcasts the tensor to the resulting shape.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn broadcast_left_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[3, 4]);
    ///     let broadcasted = tensor.try_broadcast_left(&[2, 5])?;
    ///     assert_eq!(broadcasted.shape(), vec![2, 5, 3, 4]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The broadcast operation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * Common pattern for adding batch dimensions
    /// * Useful for preparing tensors for batch operations
    pub fn try_broadcast_left(&self, batch_dims: &[usize]) -> Result<Self> {
        let mut new_shape = batch_dims.to_vec();
        new_shape.extend(self.shape());
        self.try_broadcast(&new_shape)
    }

    /// Helper function to broadcast a scalar tensor to a given shape.
    ///
    /// This creates a view with the target shape that shares the same scalar storage.
    /// This is a zero-copy operation - all indices point to the same scalar value.
    fn try_broadcast_scalar_to(&self, shape: &[usize]) -> Result<Self> {
        if self.size() != 1 {
            return Err(Error::InvalidShape {
                message: "Cannot broadcast non-scalar tensor using scalar broadcast".to_string(),
            });
        }

        let layout = self.layout().broadcast_to(shape)?;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = crate::TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: crate::TensorUpdateStatus::Pending,
                };
                crate::insert_metadata(target_tid, metadata);
                self.execute_view(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "broadcast_scalar_to",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_view(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to reshape the tensor to a new shape.
    ///
    /// This operation creates a new tensor with the specified shape. If the tensor
    /// is contiguous, this becomes a view operation. If not, the tensor is first
    /// made contiguous and then reshaped. The total number of elements must remain the same.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn reshape_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[6]);
    ///     let reshaped = tensor.try_reshape(&[2, 3])?;
    ///     assert_eq!(reshaped.shape(), vec![2, 3]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The new shape is incompatible with the tensor's total size
    /// - Contiguous conversion fails for non-contiguous tensors
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * For contiguous tensors, this is equivalent to a view (zero-copy)
    /// * For non-contiguous tensors, requires making a contiguous copy first
    /// * Use `try_view` when you know the tensor is contiguous for better performance
    pub fn try_reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let target_layout = Layout::from_shape(&computed_shape);

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: target_layout,
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_reshape(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_forward_graph(
                    "reshape",
                    &[self],
                    &[self.device()],
                    &[self.dtype()],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let result = Tensor(input_tids[0]).execute_reshape(target_tids[0])?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Common execution function for layout-only operations (view, squeeze, squeeze_all, unsqueeze, transpose, slice, unfold)
    /// that share storage and only modify metadata.
    fn execute_view(&self, target_tid: TensorId) -> Result<Self> {
        let src_storage_id = crate::get_storage_id(self.id())
            .ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;

        link_tensor_to_storage(target_tid, src_storage_id);

        crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

        Ok(Tensor(target_tid))
    }

    fn execute_reshape(&self, target_tid: TensorId) -> Result<Self> {
        if self.is_contiguous() {
            self.execute_view(target_tid)
        } else {
            let contiguous = self.try_contiguous()?;

            let src_storage_id = crate::get_storage_id(contiguous.id())
                .ok_or_else(|| Error::InvalidState("contiguous tensor storage id not found".into()))?;
            crate::link_tensor_to_storage(target_tid, src_storage_id);

            crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

            Ok(Tensor(target_tid))
        }
    }

    fn compute_shape_with_auto<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Vec<usize>> {
        let total_elements = self.size();
        let mut product: i64 = 1;
        let mut auto_dim_idx = None;

        let shape_i64: Vec<i64> = shape.iter().map(|x| x.clone().into().as_i32() as i64).collect();

        for (i, &dim) in shape_i64.iter().enumerate() {
            if dim == -1 {
                if auto_dim_idx.is_some() {
                    return Err(Error::InvalidShape {
                        message: "Only one dimension can be -1".to_string(),
                    });
                }
                auto_dim_idx = Some(i);
            } else if dim <= 0 && dim != -1 {
                return Err(Error::InvalidShape {
                    message: format!("Invalid dimension size: {}", dim),
                });
            } else {
                product *= dim;
            }
        }

        let mut result: Vec<usize> = Vec::with_capacity(shape.len());
        if let Some(idx) = auto_dim_idx {
            if product == 0 {
                return Err(Error::InvalidShape {
                    message: "Cannot infer size for dimension -1".to_string(),
                });
            }

            let auto_dim = total_elements as i64 / product;
            if auto_dim * product != total_elements as i64 {
                return Err(Error::InvalidShape {
                    message: "Cannot reshape tensor: incompatible dimensions".to_string(),
                });
            }

            for (i, &dim) in shape_i64.iter().enumerate() {
                if i == idx {
                    result.push(auto_dim as usize);
                } else {
                    result.push(dim as usize);
                }
            }
        } else {
            if product != total_elements as i64 {
                return Err(Error::InvalidShape {
                    message: "Total size of new shape must be equal to old size".to_string(),
                });
            }
            result = shape_i64.iter().map(|&x| x as usize).collect();
        }

        Ok(result)
    }
}
