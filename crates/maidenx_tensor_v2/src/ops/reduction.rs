use crate::{
    get_mode, insert_metadata, next_tensor_id, utils::graph::add_to_graph, Tensor, TensorId, TensorMetadata,
    TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    buffer::BufferManager,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};
use std::sync::Arc;

macro_rules! impl_reduction_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target_tid: TensorId,
            target_dtype: DType,
            target_layout: Layout,
            dim: usize,
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

                // Prepare metadata for the backend operation
                let metadata = input_tensor.prepare_reduction_metadata(dim);

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        input_tensor.ndim(),
                        1, // This parameter often represents the 'number of reductions' or similar in backend ops
                        Some(&metadata),
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target_tid, sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));

            crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

            Ok(Tensor {
                tid: target_tid,
                gtid: TensorId(0),
                gid: self.gid(),
            })
        }
    };
}

macro_rules! impl_reduction_execute_special {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target_tid: TensorId,
            target_dtype: DType,
            target_layout: Layout,
            metadata: Vec<usize>, // Metadata is prepared externally and passed directly
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

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        input_tensor.ndim(),
                        Some(&metadata), // Use the passed metadata
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target_tid, sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));

            crate::utils::tensor::update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

            Ok(Tensor {
                tid: target_tid,
                gtid: TensorId(0),
                gid: self.gid(),
            })
        }
    };
}

/// ## Tensor Reduction Operations
///
/// This `impl` block provides methods for performing reduction operations on tensors,
/// which aggregate elements along specified dimensions or across the entire tensor.
///
/// There are several main categories:
///
/// * **Summation**: `sum`, `sum_all`, `sum_to_shape`
/// * **Mean**: `mean`, `mean_all`
/// * **Min/Max**: `min`, `max`, `min_all`, `max_all`
/// * **Norm**: `norm`, `norm_all` (L1 and L2 norms are commonly implemented)
/// * **Variance/Standard Deviation**: `var`, `var_all`, `std`, `std_all`
/// * **Fold**: `fold` (a more general reduction for windowed operations)
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Key features:**
/// * **Dimension specification**: Reductions can be applied along a specific dimension (`dim`).
/// * **Keep dimensions**: Option to retain the reduced dimension with size 1 (`keep_dim`).
/// * **Type promotion**: Data types are promoted when necessary to maintain precision (e.g., integer mean to F32).
/// * **Device compatibility**: Operations require tensors on the same device.
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct computation.
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution.
impl Tensor {
    /// Runs [`try_sum`](Self::try_sum) and panics on failure.
    ///
    /// Computes the sum of elements along a given dimension.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let sum_dim0 = tensor.sum(0, false); // Sum along rows: [4.0, 6.0]
    /// let sum_dim1 = tensor.sum(1, false); // Sum along columns: [3.0, 7.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn sum(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Tensor {
        self.try_sum(dim, keep_dim).expect("failed to sum tensor")
    }

    /// Runs [`try_sum_all`](Self::try_sum_all) and panics on failure.
    ///
    /// Computes the sum of all elements in the tensor, resulting in a scalar tensor.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let total_sum = tensor.sum_all(); // Result: [10.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn sum_all(&self) -> Self {
        self.try_sum_all().expect("failed to sum all tensor")
    }

    /// Runs [`try_sum_to_shape`](Self::try_sum_to_shape) and panics on failure.
    ///
    /// Sums elements to reshape the tensor to a target shape. This operation
    /// is essentially a batched reduction where elements from the original
    /// dimensions that are 'collapsed' into the new shape are summed.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[1, 2, 3, 4, 5, 6, 7, 8]).reshape(&[2, 2, 2]);
    /// // Sum [1,2] and [3,4] into first elements of target, etc.
    /// let result = tensor.sum_to_shape(&[2, 2]);
    /// // Original: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    /// // Summing to [2, 2] might mean:
    /// // [1+2+3+4, 5+6+7+8] if reducing along first dim
    /// // Or other patterns depending on the backend's interpretation of sum_to_shape.
    /// // Typically, it means reducing outer dimensions to match the target shape.
    /// // This example assumes summing along the last dimension implicitly for illustration.
    /// // (Note: The exact behavior of sum_to_shape often needs careful definition,
    /// // the example is illustrative of reduction for reshaping.)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the target shape is incompatible (e.g., incorrect number of dimensions, non-divisible dimensions).
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn sum_to_shape(&self, shape: &[usize]) -> Tensor {
        self.try_sum_to_shape(shape).expect("failed to sum to shape")
    }

    /// Runs [`try_mean`](Self::try_mean) and panics on failure.
    ///
    /// Computes the mean (average) of elements along a given dimension.
    /// Integer tensors are promoted to F32 for accurate mean calculation.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let mean_dim0 = tensor.mean(0, false); // Mean along rows: [2.0, 3.0]
    /// let mean_dim1 = tensor.mean(1, false); // Mean along columns: [1.5, 3.5]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn mean(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Self {
        self.try_mean(dim, keep_dim).expect("failed to mean tensor")
    }

    /// Runs [`try_mean_all`](Self::try_mean_all) and panics on failure.
    ///
    /// Computes the mean (average) of all elements in the tensor, resulting in a scalar tensor.
    /// Integer tensors are promoted to F32 for accurate mean calculation.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let total_mean = tensor.mean_all(); // Result: [2.5]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn mean_all(&self) -> Self {
        self.try_mean_all().expect("failed to mean all tensor")
    }

    /// Runs [`try_fold`](Self::try_fold) and panics on failure.
    ///
    /// Performs a generalized fold operation, often used for windowed or strided reductions.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// // Example: Fold with dim 0, window size 3, step 1 (like a moving sum)
    /// // Actual behavior depends heavily on backend implementation.
    /// // For example, it might turn [1,2,3,4,5,6] into a series of windows:
    /// // [[1,2,3], [2,3,4], [3,4,5], [4,5,6]] and then sum each window.
    /// // The `fold` function appears to handle output shape calculation based on parameters.
    /// let result = tensor.fold(0, 3, 1);
    /// // This specific example output would depend on backend 'fold' definition.
    /// // If it implies sum over windows: [6.0, 9.0, 12.0, 15.0] (for window size 3, step 1)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When window size mismatches or other input parameters are invalid.
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn fold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Tensor {
        self.try_fold(dim, size, step).expect("failed to fold tensor")
    }

    /// Runs [`try_max`](Self::try_max) and panics on failure.
    ///
    /// Computes the maximum value of elements along a given dimension.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]);
    /// let max_dim0 = tensor.max(0, false); // Max along rows: [3.0, 5.0]
    /// let max_dim1 = tensor.max(1, false); // Max along columns: [5.0, 3.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn max(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Tensor {
        self.try_max(dim, keep_dim).expect("failed to max tensor")
    }

    /// Runs [`try_max_all`](Self::try_max_all) and panics on failure.
    ///
    /// Computes the maximum value among all elements in the tensor, resulting in a scalar tensor.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]);
    /// let total_max = tensor.max_all(); // Result: [5.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn max_all(&self) -> Self {
        self.try_max_all().expect("failed to max all tensor")
    }

    /// Runs [`try_min`](Self::try_min) and panics on failure.
    ///
    /// Computes the minimum value of elements along a given dimension.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]);
    /// let min_dim0 = tensor.min(0, false); // Min along rows: [1.0, 2.0]
    /// let min_dim1 = tensor.min(1, false); // Min along columns: [1.0, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn min(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Tensor {
        self.try_min(dim, keep_dim).expect("failed to min tensor")
    }

    /// Runs [`try_min_all`](Self::try_min_all) and panics on failure.
    ///
    /// Computes the minimum value among all elements in the tensor, resulting in a scalar tensor.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]);
    /// let total_min = tensor.min_all(); // Result: [1.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn min_all(&self) -> Self {
        self.try_min_all().expect("failed to min all tensor")
    }

    /// Runs [`try_norm`](Self::try_norm) and panics on failure.
    ///
    /// Computes the vector norm (L1, L2, etc.) of elements along a given dimension.
    /// Integer tensors are promoted to F32 for accurate norm calculation.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[[3.0, 4.0], [5.0, 12.0]]);
    /// // L2 norm along dimension 0 (columns)
    /// let norm_l2_dim0 = tensor.norm(2, 0, false); // Sqrt(3^2+5^2), Sqrt(4^2+12^2) -> [5.83, 13.0]
    /// // L1 norm along dimension 1 (rows)
    /// let norm_l1_dim1 = tensor.norm(1, 1, false); // (3+4), (5+12) -> [7.0, 17.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails (e.g., negative base with fractional exponent for `pow` internally).
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn norm(&self, p: impl Into<Scalar>, dim: impl Into<Scalar>, keep_dim: bool) -> Self {
        self.try_norm(p, dim, keep_dim).expect("failed to norm tensor")
    }

    /// Runs [`try_norm_all`](Self::try_norm_all) and panics on failure.
    ///
    /// Computes the vector norm (L1, L2, etc.) of all elements in the tensor.
    /// Integer tensors are promoted to F32 for accurate norm calculation.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[3.0, 4.0]);
    /// let norm_l2_all = tensor.norm_all(2); // Sqrt(3^2 + 4^2) = Sqrt(9 + 16) = Sqrt(25) = 5.0
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn norm_all(&self, p: impl Into<Scalar>) -> Tensor {
        self.try_norm_all(p).expect("failed to norm all tensor")
    }

    /// Runs [`try_var`](Self::try_var) and panics on failure.
    ///
    /// Computes the variance of elements along a given dimension.
    /// Integer tensors are promoted to F32. Supports biased or unbiased variance.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// // Variance of [1.0, 2.0, 3.0] (unbiased): ((1-2)^2 + (2-2)^2 + (3-2)^2) / (3-1) = (1+0+1)/2 = 1.0
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let variance = tensor.var(0, false, true); // var over dim 0, no keep_dim, unbiased
    /// // Result: [1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails (e.g., division by zero for unbiased variance on single element).
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn var(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Self {
        self.try_var(dim, keep_dim, unbiased).expect("failed to var tensor")
    }

    /// Runs [`try_var_all`](Self::try_var_all) and panics on failure.
    ///
    /// Computes the variance of all elements in the tensor.
    /// Integer tensors are promoted to F32. Supports biased or unbiased variance.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let variance_all = tensor.var_all(true); // unbiased variance of all elements
    /// // Result: [1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn var_all(&self, unbiased: bool) -> Tensor {
        self.try_var_all(unbiased).expect("failed to var all tensor")
    }

    /// Runs [`try_std`](Self::try_std) and panics on failure.
    ///
    /// Computes the standard deviation of elements along a given dimension.
    /// Integer tensors are promoted to F32. Supports biased or unbiased standard deviation.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// // Standard deviation of [1.0, 2.0, 3.0] (unbiased): Sqrt(1.0) = 1.0
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let std_dev = tensor.std(0, false, true);
    /// // Result: [1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the specified dimension is out of bounds.
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn std(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Tensor {
        self.try_std(dim, keep_dim, unbiased).expect("failed to std tensor")
    }

    /// Runs [`try_std_all`](Self::try_std_all) and panics on failure.
    ///
    /// Computes the standard deviation of all elements in the tensor.
    /// Integer tensors are promoted to F32. Supports biased or unbiased standard deviation.
    ///
    /// # Examples
    /// ```
    /// use crate::Tensor;
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let std_dev_all = tensor.std_all(true);
    /// // Result: [1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails.
    /// * When forward pass fails (for tensors in computation graphs).
    pub fn std_all(&self, unbiased: bool) -> Tensor {
        self.try_std_all(unbiased).expect("failed to std all tensor")
    }

    /// Attempts to compute the sum of elements along a given dimension.
    ///
    /// If `keep_dim` is true, the output tensor will have the same number of dimensions
    /// as the input, with the reduced dimension having size 1. Otherwise, the reduced
    /// dimension is removed.
    ///
    /// # Parameters
    /// * `dim`: The dimension along which to sum. Can be a negative index for reverse lookup.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn sum_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1, 2], [3, 4]]); // Shape [2, 2]
    ///     let sum_dim0 = tensor.try_sum(0, false)?; // Result shape [2], values [4, 6]
    ///     let sum_dim1_keep = tensor.try_sum(1, true)?; // Result shape [2, 1], values [[3], [7]]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - Memory allocation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_sum(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&shape);

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_sum(target_tid, target_dtype, target_layout, dim)?
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_graph(
                    &[self],
                    "sum",
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_sum(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            dim,
                        )?])
                    },
                )?;
                result.into_iter().next().unwrap()
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    /// Attempts to compute the sum of all elements in the tensor.
    ///
    /// This operation repeatedly calls `try_sum` to reduce all dimensions
    /// until a scalar tensor remains.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn sum_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    ///     let total_sum = tensor.try_sum_all()?; // Result: [10.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any intermediate `try_sum` operation fails.
    pub fn try_sum_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_sum(dim, false)?;
        }
        Ok(result)
    }

    /// Attempts to sum elements of the tensor to reshape it to a target shape.
    ///
    /// This is typically used for operations like reducing batch dimensions or
    /// combining feature dimensions. The elements of original dimensions that are
    /// being "collapsed" into the new shape are summed.
    ///
    /// # Parameters
    /// * `shape`: The target shape to sum the tensor into.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn sum_to_specific_shape() -> Result<()> {
    ///     let tensor = Tensor::new(&[1, 2, 3, 4, 5, 6, 7, 8]).reshape(&[2, 2, 2])?;
    ///     // Example: Sum over the last dimension to get a [2, 2] tensor.
    ///     // Input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    ///     // Target: [2, 2]
    ///     // Expected for this example: [[1+2, 3+4], [5+6, 7+8]] = [[3, 7], [11, 15]]
    ///     let result = tensor.try_sum_to_shape(&[2, 2])?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The `target_shape` has a different number of dimensions than the input tensor,
    ///   and reshaping by sum is not implicitly handled (e.g., if target has more dims).
    /// - Any dimension in the input shape is not divisible by the corresponding dimension in `target_shape`.
    /// - Memory allocation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_sum_to_shape(&self, shape: &[usize]) -> Result<Tensor> {
        if shape.len() != self.ndim() {
            if self.ndim() > shape.len() {
                let mut result = self.clone();
                while result.ndim() > shape.len() {
                    result = result.try_sum(0, false)?;
                }
                return result.try_sum_to_shape(shape);
            } else {
                return Err(Error::ShapeMismatch {
                    expected: self.ndim(),
                    got: shape.len(),
                    msg: "Target shape has more dimensions than input".to_string(),
                });
            }
        }

        for (i, &dim) in shape.iter().enumerate() {
            if self.shape()[i] % dim != 0 {
                return Err(Error::ShapeMismatch {
                    expected: self.shape()[i],
                    got: dim,
                    msg: format!("Dimension {} is not divisible: {} -> {}", i, self.shape()[i], dim),
                });
            }
        }

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(shape);
        let metadata = self.prepare_metadata_for_shape(shape);

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata_tensor = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata_tensor);
                self.execute_sum_to_shape(target_tid, target_dtype, target_layout, metadata)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let metadata_clone = metadata.clone();
                let result = add_to_graph(
                    &[self],
                    "sum_to_shape",
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_sum_to_shape(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            metadata_clone.clone(),
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute the mean (average) of elements along a given dimension.
    ///
    /// Integer tensors are promoted to F32 to avoid precision loss.
    /// If `keep_dim` is true, the output tensor will retain the reduced dimension with size 1.
    ///
    /// # Parameters
    /// * `dim`: The dimension along which to compute the mean. Can be a negative index.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn mean_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1, 2], [3, 4]]); // Shape [2, 2]
    ///     let mean_dim0 = tensor.try_mean(0, false)?; // Result shape [2], values [2.0, 3.0]
    ///     let mean_dim1_keep = tensor.try_mean(1, true)?; // Result shape [2, 1], values [[1.5], [3.5]]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - Memory allocation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    ///
    /// # Type Promotion
    /// * Integer tensors are promoted to F32 to maintain precision.
    /// * Float tensors maintain their type.
    pub fn try_mean(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };
        let target_layout = Layout::from_shape(&shape);

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_mean(target_tid, target_dtype, target_layout, dim)?
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_graph(
                    &[self],
                    "mean",
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_mean(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            dim,
                        )?])
                    },
                )?;
                result.into_iter().next().unwrap()
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    /// Attempts to compute the mean (average) of all elements in the tensor.
    ///
    /// This operation repeatedly calls `try_mean` to reduce all dimensions
    /// until a scalar tensor remains. Integer tensors are promoted to F32.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn mean_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]);
    ///     let total_mean = tensor.try_mean_all()?; // Result: [2.5]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any intermediate `try_mean` operation fails.
    pub fn try_mean_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_mean(dim, false)?;
        }
        Ok(result)
    }

    /// Attempts to perform a generalized fold operation, often used for windowed or strided reductions.
    ///
    /// This function applies a sliding window reduction along `dim + 1` based on `size` and `step`.
    /// The `fold_dim` corresponds to the dimension along which the windows are created,
    /// and `window_dim` (which is `fold_dim + 1`) is the dimension that gets 'folded'.
    ///
    /// # Parameters
    /// * `dim`: The dimension that defines the "batch" of windows. The actual windowing happens on `dim + 1`. Can be a negative index.
    /// * `size`: The size of the sliding window. If 0, it means the window size is inferred from the `window_dim` length.
    /// * `step`: The step (stride) of the sliding window.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn fold_tensor() -> Result<()> {
    ///     // Example: Fold operation. The exact semantics depend on the backend's `fold` implementation.
    ///     // Generally, it involves sliding a window and performing a reduction (like sum or max)
    ///     // within each window, producing an output with adjusted dimensions.
    ///     // A common use case is for convolutions or pooling where windows slide.
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///     // If `fold` sums 2-element windows with stride 1:
    ///     // [1+2, 2+3, 3+4, 4+5, 5+6] = [3, 5, 7, 9, 11]
    ///     let result = tensor.try_fold(0, 2, 1)?; // Assuming dim 0, size 2, step 1
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dim` is out of bounds or leads to an invalid `window_dim`.
    /// - The specified `size` does not match `window_dim` size (unless `size` is 0).
    /// - Memory allocation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_fold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let size_i32 = size.into().as_i32();
        let step_i32 = step.into().as_i32();

        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim >= self.ndim() - 1 {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let window_dim = dim + 1;
        let window_size = self.shape()[window_dim];
        if size_i32 as usize != 0 && size_i32 as usize != window_size {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch: window size is {}, but requested size is {}",
                window_size, size_i32
            )));
        }

        let n_windows = self.shape()[dim];
        let orig_dim_size = (n_windows - 1) * (step_i32 as usize) + window_size;

        let mut output_shape = Vec::with_capacity(self.ndim() - 1);
        for d in 0..self.ndim() {
            if d == dim {
                output_shape.push(orig_dim_size);
            } else if d == window_dim {
                continue;
            } else {
                output_shape.push(self.shape()[d]);
            }
        }

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&output_shape);
        let metadata = self.prepare_metadata_for_fold(dim, window_dim, orig_dim_size, step_i32 as usize, window_size);

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata_tensor = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata_tensor);
                self.execute_fold(target_tid, target_dtype, target_layout, metadata)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let metadata_clone = metadata.clone();
                let result = add_to_graph(
                    &[self],
                    "fold",
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_fold(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            metadata_clone.clone(),
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute the maximum value of elements along a given dimension.
    ///
    /// If `keep_dim` is true, the output tensor will have the same number of dimensions
    /// as the input, with the reduced dimension having size 1. Otherwise, the reduced
    /// dimension is removed.
    ///
    /// # Parameters
    /// * `dim`: The dimension along which to find the maximum. Can be a negative index.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn max_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]); // Shape [2, 2]
    ///     let max_dim0 = tensor.try_max(0, false)?; // Result shape [2], values [3.0, 5.0]
    ///     let max_dim1_keep = tensor.try_max(1, true)?; // Result shape [2, 1], values [[5.0], [3.0]]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - Memory allocation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_max(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&shape);

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_max(target_tid, target_dtype, target_layout, dim)?
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_graph(
                    &[self],
                    "max",
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_max(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            dim,
                        )?])
                    },
                )?;
                result.into_iter().next().unwrap()
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    /// Attempts to compute the maximum value among all elements in the tensor.
    ///
    /// This operation repeatedly calls `try_max` to reduce all dimensions
    /// until a scalar tensor remains.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn max_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]);
    ///     let total_max = tensor.try_max_all()?; // Result: [5.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any intermediate `try_max` operation fails.
    pub fn try_max_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_max(dim, false)?;
        }
        Ok(result)
    }

    /// Attempts to compute the minimum value of elements along a given dimension.
    ///
    /// If `keep_dim` is true, the output tensor will have the same number of dimensions
    /// as the input, with the reduced dimension having size 1. Otherwise, the reduced
    /// dimension is removed.
    ///
    /// # Parameters
    /// * `dim`: The dimension along which to find the minimum. Can be a negative index.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn min_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]); // Shape [2, 2]
    ///     let min_dim0 = tensor.try_min(0, false)?; // Result shape [2], values [1.0, 2.0]
    ///     let min_dim1_keep = tensor.try_min(1, true)?; // Result shape [2, 1], values [[1.0], [2.0]]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - Memory allocation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_min(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&shape);

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_min(target_tid, target_dtype, target_layout, dim)?
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_graph(
                    &[self],
                    "min",
                    &[self.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_min(
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                            dim,
                        )?])
                    },
                )?;
                result.into_iter().next().unwrap()
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    /// Attempts to compute the minimum value among all elements in the tensor.
    ///
    /// This operation repeatedly calls `try_min` to reduce all dimensions
    /// until a scalar tensor remains.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn min_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[[1.0, 5.0], [3.0, 2.0]]);
    ///     let total_min = tensor.try_min_all()?; // Result: [1.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any intermediate `try_min` operation fails.
    pub fn try_min_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_min(dim, false)?;
        }
        Ok(result)
    }

    /// Attempts to compute the vector **p-norm** of elements along a given dimension.
    ///
    /// The p-norm is calculated as `(sum(abs(x)^p))^(1/p)`.
    /// Commonly used norms:
    /// - `p=1` (L1 norm): `sum(abs(x))`
    /// - `p=2` (L2 norm or Euclidean norm): `sqrt(sum(x^2))`
    ///
    /// Integer tensors are promoted to F32 for accurate calculations, as norms often
    /// involve floating-point operations like `sqrt` and `pow`.
    ///
    /// # Parameters
    /// * `p`: The order of the norm (e.g., 1.0 for L1, 2.0 for L2). Can be any scalar.
    /// * `dim`: The dimension along which to compute the norm. Can be a negative index.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn norm_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[[3.0, 4.0], [5.0, 12.0]]);
    ///     // L2 norm along dimension 0 (summing squares down columns, then sqrt)
    ///     let norm_l2_dim0 = tensor.try_norm(2, 0, false)?; // Result: [5.830952, 13.0]
    ///
    ///     // L1 norm along dimension 1 (summing absolute values across rows)
    ///     let norm_l1_dim1 = tensor.try_norm(1, 1, false)?; // Result: [7.0, 17.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - Any intermediate operation (`abs`, `pow`, `sum`, `sqrt`) fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_norm(&self, p: impl Into<Scalar>, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Self> {
        let p_f32 = p.into().as_f32();
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        if p_f32 == 1.0 {
            // L1 norm: sum(abs(x))
            let abs_input = self.try_abs()?;
            let result = abs_input.try_sum(dim, keep_dim)?;
            Ok(result)
        } else if p_f32 == 2.0 {
            // L2 norm: sqrt(sum(x^2))
            let squared = self.try_pow(2.0)?;
            let sum_squared = squared.try_sum(dim, keep_dim)?;
            let result = sum_squared.try_sqrt()?;
            Ok(result)
        } else {
            // Generalized p-norm: (sum(abs(x)^p))^(1/p)
            let abs_input = self.try_abs()?;
            let pow_input = abs_input.try_pow(p_f32)?;
            let sum_result = pow_input.try_sum(dim, keep_dim)?;
            let result = sum_result.try_pow(1.0 / p_f32)?;
            Ok(result)
        }
    }

    /// Attempts to compute the vector **p-norm** of all elements in the tensor.
    ///
    /// This operation calculates the p-norm across all elements, resulting in a scalar tensor.
    /// It leverages `try_abs`, `try_pow`, and `try_sum_all` for the computation.
    /// Integer tensors are promoted to F32.
    ///
    /// # Parameters
    /// * `p`: The order of the norm (e.g., 1.0 for L1, 2.0 for L2).
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn norm_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[3.0, 4.0]);
    ///     let norm_l2_all = tensor.try_norm_all(2)?; // Result: [5.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any intermediate operation fails.
    pub fn try_norm_all(&self, p: impl Into<Scalar>) -> Result<Tensor> {
        let p_f32 = p.into().as_f32();

        if p_f32 == 1.0 {
            self.try_abs()?.try_sum_all()
        } else if p_f32 == 2.0 {
            let squared = self.try_pow(2.0)?;
            let sum_squared = squared.try_sum_all()?;
            sum_squared.try_sqrt()
        } else {
            let abs_values = self.try_abs()?;
            let pow_values = abs_values.try_pow(p_f32)?;
            let sum_result = pow_values.try_sum_all()?;
            sum_result.try_pow(1.0 / p_f32)
        }
    }

    /// Attempts to compute the **variance** of elements along a given dimension.
    ///
    /// The variance measures the average of the squared differences from the mean.
    /// It can be calculated as **biased** (dividing by `n`) or **unbiased** (dividing by `n-1`).
    /// Integer tensors are promoted to F32 to ensure accurate floating-point results.
    ///
    /// # Parameters
    /// * `dim`: The dimension along which to compute the variance. Can be a negative index.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    /// * `unbiased`: If true, uses `n-1` in the denominator for an unbiased estimate; otherwise, uses `n`.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn variance_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0]); // Mean = 2.0
    ///     // Unbiased variance: ((1-2)^2 + (2-2)^2 + (3-2)^2) / (3-1) = (1+0+1)/2 = 1.0
    ///     let var_unbiased = tensor.try_var(0, false, true)?; // Result: [1.0]
    ///
    ///     // Biased variance: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1+0+1)/3 = 0.666...
    ///     let var_biased = tensor.try_var(0, false, false)?; // Result: [0.666...]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - The number of elements in the dimension is 1 and `unbiased` is true (division by zero).
    /// - Any intermediate operation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    ///
    /// # Type Promotion
    /// * Integer tensors are promoted to F32 for computation.
    pub fn try_var(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = self.try_to_dtype(target_dtype)?;

        let mean = input.try_mean(dim, true)?;
        let centered = input.try_sub(&mean)?;
        let squared_diff = centered.try_pow(2.0)?;
        let sum_squared_diff = squared_diff.try_sum(dim, keep_dim)?;
        let n = input.shape()[dim] as f32;
        let divisor = if unbiased { n - 1.0 } else { n };
        if divisor == 0.0 {
            return Err(Error::InvalidArgument(
                "Cannot compute unbiased variance for dimension with 1 element.".to_string(),
            ));
        }
        let result = sum_squared_diff.try_div_scalar(divisor)?;

        Ok(result)
    }

    /// Attempts to compute the **variance** of all elements in the tensor.
    ///
    /// This is a convenience method that computes variance across all elements,
    /// resulting in a scalar tensor. It supports biased or unbiased estimation.
    /// Integer tensors are promoted to F32.
    ///
    /// # Parameters
    /// * `unbiased`: If true, uses `N-1` in the denominator for an unbiased estimate; otherwise, uses `N`.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn variance_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let var_all_unbiased = tensor.try_var_all(true)?; // Result: [1.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor has only 1 element and `unbiased` is true (division by zero).
    /// - Any intermediate operation fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_var_all(&self, unbiased: bool) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let input = self.try_to_dtype(target_dtype)?;
        let mean = input.try_mean_all()?;
        let centered = input.try_sub(&mean)?;
        let squared_diff = centered.try_pow(2.0)?;
        let sum_squared_diff = squared_diff.try_sum_all()?;

        let n = self.shape().iter().product::<usize>() as f32;
        let divisor = if unbiased { n - 1.0 } else { n };
        if divisor == 0.0 {
            return Err(Error::InvalidArgument(
                "Cannot compute unbiased variance for a tensor with 1 element.".to_string(),
            ));
        }
        let result = sum_squared_diff.try_div_scalar(divisor)?;

        Ok(result)
    }

    /// Attempts to compute the **standard deviation** of elements along a given dimension.
    ///
    /// The standard deviation is the square root of the variance.
    /// Integer tensors are promoted to F32. Supports biased or unbiased standard deviation.
    ///
    /// # Parameters
    /// * `dim`: The dimension along which to compute the standard deviation. Can be a negative index.
    /// * `keep_dim`: If true, retains the reduced dimension with size 1.
    /// * `unbiased`: If true, uses `n-1` for variance calculation; otherwise, uses `n`.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn std_tensor_dim() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0]); // Variance (unbiased) = 1.0
    ///     let std_dev = tensor.try_std(0, false, true)?; // Result: [1.0] (sqrt of 1.0)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified dimension is out of bounds.
    /// - Any intermediate operation (`try_var`, `try_sqrt`) fails.
    /// - Backend computation fails.
    /// - Graph operations fail in lazy mode.
    pub fn try_std(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Result<Tensor> {
        let var_result = self.try_var(dim, keep_dim, unbiased)?;
        let result = var_result.try_sqrt()?;
        Ok(result)
    }

    /// Attempts to compute the **standard deviation** of all elements in the tensor.
    ///
    /// This is a convenience method that computes standard deviation across all elements,
    /// resulting in a scalar tensor. It supports biased or unbiased estimation.
    /// Integer tensors are promoted to F32.
    ///
    /// # Parameters
    /// * `unbiased`: If true, computes unbiased standard deviation.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn std_all_elements() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let std_dev_all = tensor.try_std_all(true)?; // Result: [1.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any intermediate operation fails.
    pub fn try_std_all(&self, unbiased: bool) -> Result<Tensor> {
        let var_result = self.try_var_all(unbiased)?;
        let result = var_result.try_sqrt()?;
        Ok(result)
    }

    impl_reduction_execute!(execute_sum, maidenx_core::be::ops::reduction::sum);
    impl_reduction_execute_special!(execute_sum_to_shape, maidenx_core::be::ops::reduction::sum_to_shape);
    impl_reduction_execute!(execute_mean, maidenx_core::be::ops::reduction::mean);
    impl_reduction_execute_special!(execute_fold, maidenx_core::be::ops::reduction::fold);
    impl_reduction_execute!(execute_max, maidenx_core::be::ops::reduction::max);
    impl_reduction_execute!(execute_min, maidenx_core::be::ops::reduction::min);

    fn prepare_reduction_metadata(&self, dim: usize) -> Vec<usize> {
        let mut info = Vec::new();
        let shape = self.shape();
        let strides = self.strides();
        info.extend_from_slice(&shape);
        info.extend_from_slice(&strides);
        info.push(shape[dim]);
        info.push(strides[dim]);
        info.push(self.offset());
        info
    }

    fn prepare_metadata_for_shape(&self, target_shape: &[usize]) -> Vec<usize> {
        let mut info = Vec::new();
        let input_shape = self.shape();
        let input_strides = self.strides();
        info.extend_from_slice(&input_shape);
        info.extend_from_slice(&input_strides);
        info.extend_from_slice(target_shape);
        info.push(self.offset());
        info
    }

    fn prepare_metadata_for_fold(
        &self,
        fold_dim: usize,
        window_dim: usize,
        fold_size: usize,
        step: usize,
        window_size: usize,
    ) -> Vec<usize> {
        let mut info = Vec::new();
        let input_shape = self.shape();
        let input_strides = self.strides();

        info.extend_from_slice(&input_shape);
        info.extend_from_slice(&input_strides);

        info.push(fold_dim);
        info.push(window_dim);
        info.push(fold_size);
        info.push(step);
        info.push(window_size);

        info.push(self.offset());

        info
    }
}

