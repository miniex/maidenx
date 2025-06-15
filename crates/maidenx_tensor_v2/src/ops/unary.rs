use crate::{
    get_mode, insert_metadata, next_tensor_id, utils::graph::add_to_graph, Tensor, TensorId, TensorMetadata,
    TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};

macro_rules! impl_unary_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(&self, target_tid: TensorId, target_dtype: DType) -> Result<Self> {
            crate::eager!();

            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let shape = self.shape();
            let size = shape.iter().product();
            let mut buffer = BufferManager::create(size, self.device(), target_dtype)?;

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
                        size,
                        shape.len(),
                        Some(&self.prepare_metadata()),
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

macro_rules! impl_unary_with_scalar_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(&self, target_tid: TensorId, target_dtype: DType, scalar: Scalar) -> Result<Self> {
            crate::eager!();

            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let shape = self.shape();
            let size = shape.iter().product();
            let mut buffer = BufferManager::create(size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };
                let input_scalar = if scalar.dtype() != target_dtype {
                    scalar.to_dtype(target_dtype)
                } else {
                    scalar.clone()
                };

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_scalar,
                        size,
                        shape.len(),
                        Some(&self.prepare_metadata()),
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

/// ## Tensor unary operations helpers
///
/// This `impl` block provides methods for applying unary operations to tensors.
/// There are several main categories:
///
/// * **Basic arithmetic**: `neg`, `abs`, `sign` - basic mathematical operations
/// * **Mathematical functions**: `sqrt`, `exp`, `log`, `sin`, `cos`, etc.
/// * **Activation functions**: `relu`, `sigmoid`, `tanh`, `gelu`, etc.
/// * **Scalar operations**: `add_scalar`, `mul_scalar`, etc.
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct computation
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **Performance notes:**
/// * All operations create new tensors and do not modify the original
/// * Type promotion is handled automatically when needed
/// * Backend-specific optimizations are applied transparently
impl Tensor {
    /// Runs [`try_neg`](Self::try_neg) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, -2.0, 3.0]);
    /// let negated = tensor.neg();
    /// // Result: [-1.0, 2.0, -3.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn neg(&self) -> Self {
        self.try_neg().expect("failed to negate tensor")
    }

    /// Runs [`try_abs`](Self::try_abs) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, -2.0, 3.0]);
    /// let absolute = tensor.abs();
    /// // Result: [1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn abs(&self) -> Self {
        self.try_abs().expect("failed to compute absolute value")
    }

    /// Runs [`try_sign`](Self::try_sign) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, -2.0, 0.0, 3.0]);
    /// let signs = tensor.sign();
    /// // Result: [1.0, -1.0, 0.0, 1.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn sign(&self) -> Self {
        self.try_sign().expect("failed to compute sign")
    }

    /// Runs [`try_square`](Self::try_square) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[2.0, 3.0, 4.0]);
    /// let squared = tensor.square();
    /// // Result: [4.0, 9.0, 16.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn square(&self) -> Self {
        self.try_square().expect("failed to compute square")
    }

    /// Runs [`try_sqrt`](Self::try_sqrt) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[4.0, 9.0, 16.0]);
    /// let sqrt_result = tensor.sqrt();
    /// // Result: [2.0, 3.0, 4.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn sqrt(&self) -> Self {
        self.try_sqrt().expect("failed to compute sqrt")
    }

    /// Runs [`try_relu`](Self::try_relu) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let relu_result = tensor.relu();
    /// // Result: [0.0, 0.0, 0.0, 1.0, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn relu(&self) -> Self {
        self.try_relu().expect("failed to compute relu")
    }

    /// Runs [`try_sigmoid`](Self::try_sigmoid) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-1.0, 0.0, 1.0]);
    /// let sigmoid_result = tensor.sigmoid();
    /// // Result: [0.269, 0.5, 0.731] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn sigmoid(&self) -> Self {
        self.try_sigmoid().expect("failed to compute sigmoid")
    }

    /// Runs [`try_tanh`](Self::try_tanh) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-1.0, 0.0, 1.0]);
    /// let tanh_result = tensor.tanh();
    /// // Result: [-0.762, 0.0, 0.762] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn tanh(&self) -> Self {
        self.try_tanh().expect("failed to compute tanh")
    }

    /// Runs [`try_gelu`](Self::try_gelu) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-1.0, 0.0, 1.0]);
    /// let gelu_result = tensor.gelu();
    /// // Result: [-0.159, 0.0, 0.841] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn gelu(&self) -> Self {
        self.try_gelu().expect("failed to compute gelu")
    }

    /// Runs [`try_sin`](Self::try_sin) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[0.0, std::f32::consts::PI / 2.0]);
    /// let sin_result = tensor.sin();
    /// // Result: [0.0, 1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn sin(&self) -> Self {
        self.try_sin().expect("failed to compute sin")
    }

    /// Runs [`try_cos`](Self::try_cos) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[0.0, std::f32::consts::PI]);
    /// let cos_result = tensor.cos();
    /// // Result: [1.0, -1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn cos(&self) -> Self {
        self.try_cos().expect("failed to compute cos")
    }

    /// Runs [`try_tan`](Self::try_tan) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[0.0, std::f32::consts::PI / 4.0]);
    /// let tan_result = tensor.tan();
    /// // Result: [0.0, 1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn tan(&self) -> Self {
        self.try_tan().expect("failed to compute tan")
    }

    /// Runs [`try_ln`](Self::try_ln) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, std::f32::consts::E]);
    /// let ln_result = tensor.ln();
    /// // Result: [0.0, 1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn ln(&self) -> Self {
        self.try_ln().expect("failed to compute ln")
    }

    /// Runs [`try_log`](Self::try_log) and panics on failure.
    ///
    /// This is an alias for [`ln`](Self::ln).
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, std::f32::consts::E]);
    /// let log_result = tensor.log();
    /// // Result: [0.0, 1.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn log(&self) -> Self {
        self.try_log().expect("failed to compute log")
    }

    /// Runs [`try_log10`](Self::try_log10) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 10.0, 100.0]);
    /// let log10_result = tensor.log10();
    /// // Result: [0.0, 1.0, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn log10(&self) -> Self {
        self.try_log10().expect("failed to compute log10")
    }

    /// Runs [`try_log2`](Self::try_log2) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 4.0, 8.0]);
    /// let log2_result = tensor.log2();
    /// // Result: [0.0, 1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn log2(&self) -> Self {
        self.try_log2().expect("failed to compute log2")
    }

    /// Runs [`try_exp`](Self::try_exp) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[0.0, 1.0, 2.0]);
    /// let exp_result = tensor.exp();
    /// // Result: [1.0, 2.718, 7.389] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn exp(&self) -> Self {
        self.try_exp().expect("failed to compute exp")
    }

    /// Runs [`try_exp10`](Self::try_exp10) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[0.0, 1.0, 2.0, 3.0]);
    /// let exp10_result = tensor.exp10();
    /// // Result: [1.0, 10.0, 100.0, 1000.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn exp10(&self) -> Self {
        self.try_exp10().expect("failed to compute exp10")
    }

    /// Runs [`try_exp2`](Self::try_exp2) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    /// let exp2_result = tensor.exp2();
    /// // Result: [1.0, 2.0, 4.0, 8.0, 16.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn exp2(&self) -> Self {
        self.try_exp2().expect("failed to compute exp2")
    }

    /// Runs [`try_softplus`](Self::try_softplus) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let softplus_result = tensor.softplus();
    /// // Result: [0.127, 0.313, 0.693, 1.313, 2.127] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn softplus(&self) -> Self {
        self.try_softplus().expect("failed to compute softplus")
    }

    /// Runs [`try_recip`](Self::try_recip) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 4.0, 0.5]);
    /// let recip_result = tensor.recip();
    /// // Result: [1.0, 0.5, 0.25, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn recip(&self) -> Self {
        self.try_recip().expect("failed to compute recip")
    }

    /// Runs [`try_logical_not`](Self::try_logical_not) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[true, false, true, false]);
    /// let not_result = tensor.logical_not();
    /// // Result: [false, true, false, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn logical_not(&self) -> Self {
        self.try_logical_not().expect("failed to compute logical_not")
    }

    /// Runs [`try_add_scalar`](Self::try_add_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.add_scalar(5.0);
    /// // Result: [6.0, 7.0, 8.0, 9.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn add_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_add_scalar(scalar).expect("failed to add scalar")
    }

    /// Runs [`try_sub_scalar`](Self::try_sub_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[6.0, 7.0, 8.0, 9.0]);
    /// let result = tensor.sub_scalar(5.0);
    /// // Result: [1.0, 2.0, 3.0, 4.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn sub_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_sub_scalar(scalar).expect("failed to subtract scalar")
    }

    /// Runs [`try_mul_scalar`](Self::try_mul_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.mul_scalar(2.0);
    /// // Result: [2.0, 4.0, 6.0, 8.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn mul_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_mul_scalar(scalar).expect("failed to multiply by scalar")
    }

    /// Runs [`try_div_scalar`](Self::try_div_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[2.0, 4.0, 6.0, 8.0]);
    /// let result = tensor.div_scalar(2.0);
    /// // Result: [1.0, 2.0, 3.0, 4.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn div_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_div_scalar(scalar).expect("failed to divide by scalar")
    }

    /// Runs [`try_maximum_scalar`](Self::try_maximum_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.maximum_scalar(2.5);
    /// // Result: [2.5, 2.5, 3.0, 4.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn maximum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_maximum_scalar(scalar)
            .expect("failed to compute maximum_scalar")
    }

    /// Runs [`try_minimum_scalar`](Self::try_minimum_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.minimum_scalar(2.5);
    /// // Result: [1.0, 2.0, 2.5, 2.5]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn minimum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_minimum_scalar(scalar)
            .expect("failed to compute minimum_scalar")
    }

    /// Runs [`try_pow`](Self::try_pow) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[2.0, 3.0, 4.0]);
    /// let result = tensor.pow(2.0);
    /// // Result: [4.0, 9.0, 16.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn pow<T: Into<Scalar>>(&self, exponent: T) -> Self {
        self.try_pow(exponent).expect("failed to compute pow")
    }

    /// Runs [`try_leaky_relu`](Self::try_leaky_relu) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = tensor.leaky_relu(0.01);
    /// // Result: [-0.02, -0.01, 0.0, 1.0, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn leaky_relu<T: Into<Scalar>>(&self, alpha: T) -> Self {
        self.try_leaky_relu(alpha).expect("failed to compute leaky_relu")
    }

    /// Runs [`try_elu`](Self::try_elu) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = tensor.elu(1.0);
    /// // Result: [-0.865, -0.632, 0.0, 1.0, 2.0] (approximately)
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn elu<T: Into<Scalar>>(&self, alpha: T) -> Self {
        self.try_elu(alpha).expect("failed to compute elu")
    }

    /// Runs [`try_eq_scalar`](Self::try_eq_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 2.0]);
    /// let result = tensor.eq_scalar(2.0);
    /// // Result: [false, true, false, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn eq_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_eq_scalar(scalar).expect("failed to compute eq_scalar")
    }

    /// Runs [`try_ne_scalar`](Self::try_ne_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 2.0]);
    /// let result = tensor.ne_scalar(2.0);
    /// // Result: [true, false, true, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn ne_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_ne_scalar(scalar).expect("failed to compute ne_scalar")
    }

    /// Runs [`try_lt_scalar`](Self::try_lt_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.lt_scalar(3.0);
    /// // Result: [true, true, false, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn lt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_lt_scalar(scalar).expect("failed to compute lt_scalar")
    }

    /// Runs [`try_le_scalar`](Self::try_le_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.le_scalar(3.0);
    /// // Result: [true, true, true, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn le_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_le_scalar(scalar).expect("failed to compute le_scalar")
    }

    /// Runs [`try_gt_scalar`](Self::try_gt_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.gt_scalar(2.0);
    /// // Result: [false, false, true, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn gt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_gt_scalar(scalar).expect("failed to compute gt_scalar")
    }

    /// Runs [`try_ge_scalar`](Self::try_ge_scalar) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = tensor.ge_scalar(2.0);
    /// // Result: [false, true, true, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When the operation fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn ge_scalar<T: Into<Scalar>>(&self, scalar: T) -> Self {
        self.try_ge_scalar(scalar).expect("failed to compute ge_scalar")
    }

    /// Attempts to negate all elements in the tensor.
    ///
    /// This operation computes the negation (-x) of each element in the tensor.
    /// Boolean tensors are promoted to U8 before negation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn negate_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, -2.0, 3.0]);
    ///     let negated = tensor.try_neg()?;
    ///     // Result: [-1.0, 2.0, -3.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Boolean tensors are promoted to U8 and then to signed types
    /// * The result maintains the promoted type for signed arithmetic
    pub fn try_neg(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8.to_signed()
        } else {
            self.dtype().to_signed()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_neg(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "neg",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_neg(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute the absolute value of all elements in the tensor.
    ///
    /// This operation computes |x| for each element in the tensor.
    /// Boolean tensors are promoted to U8 before computation.
    /// For unsigned integer types, returns a clone since they're already non-negative.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn absolute_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, -2.0, 3.0]);
    ///     let absolute = tensor.try_abs()?;
    ///     // Result: [1.0, 2.0, 3.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * For unsigned integer types, this is a zero-copy clone operation
    /// * For other types, computes actual absolute values
    pub fn try_abs(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        // For unsigned types, absolute value is identity
        if target_dtype.is_uint() && !self.dtype().is_bool() {
            return Ok(self.clone());
        }

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_abs(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "abs",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_abs(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute the sign of all elements in the tensor.
    ///
    /// This operation returns:
    /// * 1 for positive elements
    /// * -1 for negative elements  
    /// * 0 for zero elements
    ///
    /// Boolean tensors are promoted to U8 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn sign_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, -2.0, 0.0, 3.0]);
    ///     let signs = tensor.try_sign()?;
    ///     // Result: [1.0, -1.0, 0.0, 1.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_sign(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_sign(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "sign",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_sign(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute the square of all elements in the tensor.
    ///
    /// This operation computes x² for each element in the tensor.
    /// Boolean tensors are promoted to U8 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn square_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[2.0, 3.0, 4.0]);
    ///     let squared = tensor.try_square()?;
    ///     // Result: [4.0, 9.0, 16.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Boolean tensors are promoted to U8
    /// * Other types maintain their type
    pub fn try_square(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_square(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "square",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_square(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute the square root of all elements in the tensor.
    ///
    /// This operation computes √x for each element in the tensor.
    /// Boolean and integer tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn sqrt_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[4.0, 9.0, 16.0]);
    ///     let sqrt_result = tensor.try_sqrt()?;
    ///     // Result: [2.0, 3.0, 4.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_sqrt(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_sqrt(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "sqrt",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_sqrt(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply ReLU (Rectified Linear Unit) activation function.
    ///
    /// This operation computes max(0, x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn relu_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    ///     let relu_result = tensor.try_relu()?;
    ///     // Result: [0.0, 0.0, 0.0, 1.0, 2.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_relu(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_relu(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "relu",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_relu(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply sigmoid activation function.
    ///
    /// This operation computes 1/(1 + exp(-x)) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn sigmoid_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-1.0, 0.0, 1.0]);
    ///     let sigmoid_result = tensor.try_sigmoid()?;
    ///     // Result: [0.269, 0.5, 0.731] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_sigmoid(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_sigmoid(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "sigmoid",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_sigmoid(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply tanh (hyperbolic tangent) activation function.
    ///
    /// This operation computes tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x)) for each element.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn tanh_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-1.0, 0.0, 1.0]);
    ///     let tanh_result = tensor.try_tanh()?;
    ///     // Result: [-0.762, 0.0, 0.762] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_tanh(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_tanh(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "tanh",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_tanh(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply GELU (Gaussian Error Linear Unit) activation function.
    ///
    /// This operation computes GELU(x) = x * Φ(x), where Φ(x) is the cumulative distribution
    /// function of the standard Gaussian distribution. Non-float tensors are promoted to F32.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn gelu_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-1.0, 0.0, 1.0]);
    ///     let gelu_result = tensor.try_gelu()?;
    ///     // Result: [-0.159, 0.0, 0.841] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_gelu(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_gelu(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "gelu",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_gelu(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute sine of all elements in the tensor.
    ///
    /// This operation computes sin(x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn sin_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
    ///     let sin_result = tensor.try_sin()?;
    ///     // Result: [0.0, 1.0, 0.0] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_sin(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_sin(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "sin",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_sin(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute cosine of all elements in the tensor.
    ///
    /// This operation computes cos(x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn cos_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
    ///     let cos_result = tensor.try_cos()?;
    ///     // Result: [1.0, 0.0, -1.0] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    pub fn try_cos(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_cos(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "cos",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_cos(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute tangent of all elements in the tensor.
    ///
    /// This operation computes tan(x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn tan_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[0.0, std::f32::consts::PI / 4.0]);
    ///     let tan_result = tensor.try_tan()?;
    ///     // Result: [0.0, 1.0] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// Be aware that tan(x) has singularities at x = π/2 + nπ where the function approaches infinity.
    pub fn try_tan(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_tan(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "tan",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_tan(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute natural logarithm of all elements in the tensor.
    ///
    /// This operation computes ln(x) = log_e(x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn ln_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E]);
    ///     let ln_result = tensor.try_ln()?;
    ///     // Result: [0.0, 1.0, 2.0] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// ln(x) is undefined for x ≤ 0. For x = 0, the result is -∞, and for x < 0, the result is NaN.
    pub fn try_ln(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_ln(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "ln",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_ln(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute natural logarithm of all elements (alias for ln).
    ///
    /// This is an alias for [`try_ln`](Self::try_ln) that computes ln(x) = log_e(x)
    /// for each element in the tensor. Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn log_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E]);
    ///     let log_result = tensor.try_log()?;
    ///     // Result: [0.0, 1.0, 2.0] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// log(x) is undefined for x ≤ 0. For x = 0, the result is -∞, and for x < 0, the result is NaN.
    /// This function is mathematically identical to [`try_ln`](Self::try_ln).
    pub fn try_log(&self) -> Result<Self> {
        self.try_ln()
    }

    /// Attempts to compute base-10 logarithm of all elements in the tensor.
    ///
    /// This operation computes log10(x) = log_10(x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn log10_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 10.0, 100.0, 1000.0]);
    ///     let log10_result = tensor.try_log10()?;
    ///     // Result: [0.0, 1.0, 2.0, 3.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// log10(x) is undefined for x ≤ 0. For x = 0, the result is -∞, and for x < 0, the result is NaN.
    pub fn try_log10(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_log10(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "log10",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_log10(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute base-2 logarithm of all elements in the tensor.
    ///
    /// This operation computes log2(x) = log_2(x) for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn log2_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 4.0, 8.0, 16.0]);
    ///     let log2_result = tensor.try_log2()?;
    ///     // Result: [0.0, 1.0, 2.0, 3.0, 4.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// log2(x) is undefined for x ≤ 0. For x = 0, the result is -∞, and for x < 0, the result is NaN.
    pub fn try_log2(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_log2(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "log2",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_log2(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute exponential of all elements in the tensor.
    ///
    /// This operation computes exp(x) = e^x for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn exp_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[0.0, 1.0, 2.0]);
    ///     let exp_result = tensor.try_exp()?;
    ///     // Result: [1.0, 2.718, 7.389] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// exp(x) can overflow for large values of x, resulting in infinity.
    pub fn try_exp(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_exp(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "exp",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_exp(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute base-10 exponential of all elements in the tensor.
    ///
    /// This operation computes exp10(x) = 10^x for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn exp10_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[0.0, 1.0, 2.0, 3.0]);
    ///     let exp10_result = tensor.try_exp10()?;
    ///     // Result: [1.0, 10.0, 100.0, 1000.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// exp10(x) can overflow for large values of x, resulting in infinity.
    pub fn try_exp10(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_exp10(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "exp10",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_exp10(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute base-2 exponential of all elements in the tensor.
    ///
    /// This operation computes exp2(x) = 2^x for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn exp2_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    ///     let exp2_result = tensor.try_exp2()?;
    ///     // Result: [1.0, 2.0, 4.0, 8.0, 16.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// exp2(x) can overflow for large values of x, resulting in infinity.
    pub fn try_exp2(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_exp2(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "exp2",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_exp2(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply softplus activation function.
    ///
    /// This operation computes softplus(x) = ln(1 + exp(x)) for each element in the tensor.
    /// Softplus is a smooth approximation to the ReLU activation function.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn softplus_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    ///     let softplus_result = tensor.try_softplus()?;
    ///     // Result: [0.127, 0.313, 0.693, 1.313, 2.127] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Mathematical Properties
    ///
    /// * Always positive: softplus(x) > 0 for all x
    /// * Smooth approximation to ReLU: softplus(x) ≈ max(0, x) for |x| >> 1
    /// * Derivative is sigmoid: d/dx softplus(x) = sigmoid(x)
    pub fn try_softplus(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_softplus(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "softplus",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_softplus(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute reciprocal of all elements in the tensor.
    ///
    /// This operation computes recip(x) = 1/x for each element in the tensor.
    /// Non-float tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn recip_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 4.0, 0.5, 0.25]);
    ///     let recip_result = tensor.try_recip()?;
    ///     // Result: [1.0, 0.5, 0.25, 2.0, 4.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Non-float tensors are promoted to F32
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// recip(0) results in infinity. For very small values close to zero,
    /// the result can be very large and may overflow.
    pub fn try_recip(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_recip(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "recip",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_recip(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute logical NOT of all elements in the tensor.
    ///
    /// This operation computes !x for each element in the tensor, where the result
    /// is a boolean tensor. For non-boolean input tensors, zero values are treated
    /// as false and non-zero values are treated as true.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn logical_not_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[true, false, true, false]);
    ///     let not_result = tensor.try_logical_not()?;
    ///     // Result: [false, true, false, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Behavior
    ///
    /// * Output is always BOOL type regardless of input type
    /// * For non-boolean inputs: 0 becomes true, non-zero becomes false
    /// * For boolean inputs: true becomes false, false becomes true
    pub fn try_logical_not(&self) -> Result<Self> {
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_logical_not(target_tid, target_dtype)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "logical_not",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_logical_not(target_tids[0], target_dtype)?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to add a scalar value to all elements in the tensor.
    ///
    /// This operation computes tensor + scalar for each element.
    /// Type promotion is applied based on the scalar type and value.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn add_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_add_scalar(5.0)?;
    ///     // Result: [6.0, 7.0, 8.0, 9.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * If scalar is float and not integer value, promotes to F32
    /// * Otherwise maintains tensor's original type
    pub fn try_add_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_add_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "add_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_add_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to subtract a scalar value from all elements in the tensor.
    ///
    /// This operation computes tensor - scalar for each element.
    /// Type promotion is applied based on the scalar type and value.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn sub_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[6.0, 7.0, 8.0, 9.0]);
    ///     let result = tensor.try_sub_scalar(5.0)?;
    ///     // Result: [1.0, 2.0, 3.0, 4.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * If scalar is float and not integer value, promotes to F32
    /// * Otherwise maintains tensor's original type
    pub fn try_sub_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_sub_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "sub_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_sub_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to multiply all elements in the tensor by a scalar value.
    ///
    /// This operation computes tensor * scalar for each element.
    /// Type promotion is applied based on the scalar type and value.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn mul_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_mul_scalar(2.0)?;
    ///     // Result: [2.0, 4.0, 6.0, 8.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * If scalar is float and not integer value, promotes to F32
    /// * Otherwise maintains tensor's original type
    pub fn try_mul_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_mul_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "mul_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_mul_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to divide all elements in the tensor by a scalar value.
    ///
    /// This operation computes tensor / scalar for each element.
    /// Integer tensors are promoted to F32 for division to maintain precision.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn div_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[2.0, 4.0, 6.0, 8.0]);
    ///     let result = tensor.try_div_scalar(2.0)?;
    ///     // Result: [1.0, 2.0, 3.0, 4.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Integer tensors are promoted to F32 to maintain precision
    /// * Float tensors maintain their type
    ///
    /// # Note
    ///
    /// Division by zero results in infinity. Very small divisors may cause overflow.
    pub fn try_div_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_div_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "div_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_div_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise maximum between tensor elements and a scalar.
    ///
    /// This operation computes max(element, scalar) for each element in the tensor.
    /// The scalar is promoted to match the tensor's data type if needed.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn maximum_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_maximum_scalar(2.5)?;
    ///     // Result: [2.5, 2.5, 3.0, 4.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * If scalar is float and not an integer value, promotes to F32
    /// * Otherwise maintains the tensor's data type
    /// * The scalar is promoted to match the target tensor type
    pub fn try_maximum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_maximum_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "maximum_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_maximum_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise minimum between tensor elements and a scalar.
    ///
    /// This operation computes min(element, scalar) for each element in the tensor.
    /// The scalar is promoted to match the tensor's data type if needed.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn minimum_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_minimum_scalar(2.5)?;
    ///     // Result: [1.0, 2.0, 2.5, 2.5]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * If scalar is float and not an integer value, promotes to F32
    /// * Otherwise maintains the tensor's data type
    /// * The scalar is promoted to match the target tensor type
    pub fn try_minimum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_minimum_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "minimum_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_minimum_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to raise each element to the power of the given exponent.
    ///
    /// This operation computes element^exponent for each element in the tensor.
    /// The exponent can be any scalar value (integer or float).
    /// Type promotion is applied based on the exponent type.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn pow_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[2.0, 3.0, 4.0]);
    ///     let result = tensor.try_pow(2.0)?;
    ///     // Result: [4.0, 9.0, 16.0]
    ///     
    ///     let tensor2 = Tensor::new(&[4.0, 9.0, 16.0]);
    ///     let result2 = tensor2.try_pow(0.5)?;
    ///     // Result: [2.0, 3.0, 4.0] (square root)
    ///     
    ///     let tensor3 = Tensor::new(&[2, 4, 8]); // integer tensor
    ///     let result3 = tensor3.try_pow(-2)?;
    ///     // Result: [0.25, 0.0625, 0.015625] (promoted to F32)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Invalid operations (e.g., negative base with fractional exponent)
    ///
    /// # Type Promotion
    ///
    /// * If exponent is float and not an integer value, uses exponent's dtype
    /// * If exponent is negative (even if integer), promotes to F32 since results can be fractional
    /// * Otherwise maintains the tensor's data type
    /// * The exponent is promoted to match the target tensor type
    ///
    /// # Mathematical Notes
    ///
    /// * For negative bases with fractional exponents, behavior is backend-dependent
    /// * For zero base with negative exponent, may result in infinity
    /// * Very large exponents may cause overflow
    pub fn try_pow<T: Into<Scalar>>(&self, exponent: T) -> Result<Self> {
        let exponent = exponent.into();
        let target_dtype = if exponent.as_f32() < 0.0 {
            DType::F32
        } else if exponent.is_float() && !exponent.is_integer_value() {
            exponent.dtype()
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_pow(target_tid, target_dtype, exponent)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "pow",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_pow(target_tids[0], target_dtype, exponent)?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply Leaky ReLU activation function with the specified negative slope.
    ///
    /// This operation computes leaky_relu(x) = max(0, x) + alpha * min(0, x) for each element.
    /// Integer tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn leaky_relu_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    ///     let result = tensor.try_leaky_relu(0.01)?;
    ///     // Result: [-0.02, -0.01, 0.0, 1.0, 2.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Integer tensors are promoted to F32 for computation
    /// * Float tensors maintain their precision
    pub fn try_leaky_relu<T: Into<Scalar>>(&self, alpha: T) -> Result<Self> {
        let alpha = alpha.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_leaky_relu(target_tid, target_dtype, alpha)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "leaky_relu",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_leaky_relu(
                            target_tids[0],
                            target_dtype,
                            alpha,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to apply ELU (Exponential Linear Unit) activation function with the specified alpha.
    ///
    /// This operation computes elu(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0.
    /// Integer tensors are promoted to F32 before computation.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn elu_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    ///     let result = tensor.try_elu(1.0)?;
    ///     // Result: [-0.865, -0.632, 0.0, 1.0, 2.0] (approximately)
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Type Promotion
    ///
    /// * Integer tensors are promoted to F32 for computation
    /// * Float tensors maintain their precision
    pub fn try_elu<T: Into<Scalar>>(&self, alpha: T) -> Result<Self> {
        let alpha = alpha.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_elu(target_tid, target_dtype, alpha)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "elu",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_elu(target_tids[0], target_dtype, alpha)?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to perform element-wise equality comparison with a scalar.
    ///
    /// This operation compares each element with the scalar and returns a boolean tensor.
    /// The result dtype is always BOOL regardless of input types.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn eq_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 2.0]);
    ///     let result = tensor.try_eq_scalar(2.0)?;
    ///     // Result: [false, true, false, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_eq_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_eq_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "eq_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_eq_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to perform element-wise not-equal comparison with a scalar.
    ///
    /// This operation compares each element with the scalar and returns a boolean tensor.
    /// The result dtype is always BOOL regardless of input types.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn ne_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 2.0]);
    ///     let result = tensor.try_ne_scalar(2.0)?;
    ///     // Result: [true, false, true, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_ne_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_ne_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "ne_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_ne_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to perform element-wise less-than comparison with a scalar.
    ///
    /// This operation compares each element with the scalar and returns a boolean tensor.
    /// The result dtype is always BOOL regardless of input types.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn lt_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_lt_scalar(3.0)?;
    ///     // Result: [true, true, false, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_lt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_lt_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "lt_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_lt_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to perform element-wise less-than-or-equal comparison with a scalar.
    ///
    /// This operation compares each element with the scalar and returns a boolean tensor.
    /// The result dtype is always BOOL regardless of input types.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn le_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_le_scalar(3.0)?;
    ///     // Result: [true, true, true, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_le_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_le_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "le_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_le_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to perform element-wise greater-than comparison with a scalar.
    ///
    /// This operation compares each element with the scalar and returns a boolean tensor.
    /// The result dtype is always BOOL regardless of input types.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn gt_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_gt_scalar(2.0)?;
    ///     // Result: [false, false, true, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_gt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_gt_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "gt_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_gt_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to perform element-wise greater-than-or-equal comparison with a scalar.
    ///
    /// This operation compares each element with the scalar and returns a boolean tensor.
    /// The result dtype is always BOOL regardless of input types.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn ge_scalar_tensor() -> Result<()> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let result = tensor.try_ge_scalar(2.0)?;
    ///     // Result: [false, true, true, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_ge_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: self.device(),
                    dtype: target_dtype,
                    layout: self.layout().clone(),
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                self.execute_ge_scalar(target_tid, target_dtype, scalar)
            },
            TensorMode::Lazy => {
                let layout = self.layout().clone();
                let result = add_to_graph(
                    &[self],
                    "ge_scalar",
                    &[self.device()],
                    &[target_dtype],
                    &[layout],
                    move |tensors, target_tids| {
                        Ok(vec![tensors[0].execute_ge_scalar(
                            target_tids[0],
                            target_dtype,
                            scalar,
                        )?])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    impl_unary_execute!(execute_neg, maidenx_core::be::ops::unary::neg);
    impl_unary_execute!(execute_abs, maidenx_core::be::ops::unary::abs);
    impl_unary_execute!(execute_sign, maidenx_core::be::ops::unary::sign);
    impl_unary_execute!(execute_square, maidenx_core::be::ops::unary::square);
    impl_unary_execute!(execute_sqrt, maidenx_core::be::ops::unary::sqrt);
    impl_unary_execute!(execute_relu, maidenx_core::be::ops::unary::relu);
    impl_unary_execute!(execute_sigmoid, maidenx_core::be::ops::unary::sigmoid);
    impl_unary_execute!(execute_tanh, maidenx_core::be::ops::unary::tanh);
    impl_unary_execute!(execute_gelu, maidenx_core::be::ops::unary::gelu);
    impl_unary_execute!(execute_sin, maidenx_core::be::ops::unary::sin);
    impl_unary_execute!(execute_cos, maidenx_core::be::ops::unary::cos);
    impl_unary_execute!(execute_tan, maidenx_core::be::ops::unary::tan);
    impl_unary_execute!(execute_ln, maidenx_core::be::ops::unary::ln);
    impl_unary_execute!(execute_log10, maidenx_core::be::ops::unary::log10);
    impl_unary_execute!(execute_log2, maidenx_core::be::ops::unary::log2);
    impl_unary_execute!(execute_exp, maidenx_core::be::ops::unary::exp);
    impl_unary_execute!(execute_exp10, maidenx_core::be::ops::unary::exp10);
    impl_unary_execute!(execute_exp2, maidenx_core::be::ops::unary::exp2);
    impl_unary_execute!(execute_softplus, maidenx_core::be::ops::unary::softplus);
    impl_unary_execute!(execute_recip, maidenx_core::be::ops::unary::recip);
    impl_unary_execute!(execute_logical_not, maidenx_core::be::ops::unary::logical_not);

    impl_unary_with_scalar_execute!(execute_add_scalar, maidenx_core::be::ops::unary::add_scalar);
    impl_unary_with_scalar_execute!(execute_sub_scalar, maidenx_core::be::ops::unary::sub_scalar);
    impl_unary_with_scalar_execute!(execute_mul_scalar, maidenx_core::be::ops::unary::mul_scalar);
    impl_unary_with_scalar_execute!(execute_div_scalar, maidenx_core::be::ops::unary::div_scalar);
    impl_unary_with_scalar_execute!(execute_maximum_scalar, maidenx_core::be::ops::unary::maximum_scalar);
    impl_unary_with_scalar_execute!(execute_minimum_scalar, maidenx_core::be::ops::unary::minimum_scalar);
    impl_unary_with_scalar_execute!(execute_pow, maidenx_core::be::ops::unary::pow);
    impl_unary_with_scalar_execute!(execute_leaky_relu, maidenx_core::be::ops::unary::leaky_relu);
    impl_unary_with_scalar_execute!(execute_elu, maidenx_core::be::ops::unary::elu);
    impl_unary_with_scalar_execute!(execute_eq_scalar, maidenx_core::be::ops::unary::eq_scalar);
    impl_unary_with_scalar_execute!(execute_ne_scalar, maidenx_core::be::ops::unary::ne_scalar);
    impl_unary_with_scalar_execute!(execute_lt_scalar, maidenx_core::be::ops::unary::lt_scalar);
    impl_unary_with_scalar_execute!(execute_le_scalar, maidenx_core::be::ops::unary::le_scalar);
    impl_unary_with_scalar_execute!(execute_gt_scalar, maidenx_core::be::ops::unary::gt_scalar);
    impl_unary_with_scalar_execute!(execute_ge_scalar, maidenx_core::be::ops::unary::ge_scalar);

    fn prepare_metadata(&self) -> Vec<usize> {
        let mut info = Vec::new();
        info.extend_from_slice(&self.shape());
        info.extend_from_slice(&self.strides());
        info.push(self.offset());
        info
    }
}
