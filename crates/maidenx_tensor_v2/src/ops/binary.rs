use crate::{
    eager, get_mode, insert_metadata, next_tensor_id,
    utils::{
        broadcast::broadcast_tensors,
        graph::{add_to_backward_graph, add_to_forward_graph},
        promotion::get_promoted_dtype,
    },
    Tensor, TensorId, TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};

macro_rules! impl_binary_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            rhs: &Tensor,
            target_tid: TensorId,
            target_dtype: DType,
            target_layout: Layout,
        ) -> Result<Self> {
            eager!();

            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let lhs_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };
                let rhs_tensor = if rhs.dtype() != target_dtype {
                    rhs.try_to_dtype(target_dtype)?
                } else {
                    rhs.clone()
                };

                let metadata = Tensor::prepare_binary_metadata(&lhs_tensor, &rhs_tensor);

                let lhs_storage = lhs_tensor.storage()?;
                let lhs_buffer = lhs_storage.buffer();

                let rhs_storage = rhs_tensor.storage()?;
                let rhs_buffer = rhs_storage.buffer();

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        lhs_buffer,
                        rhs_buffer,
                        lhs_tensor.size(),
                        lhs_tensor.ndim(),
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

/// ## Tensor binary operations helpers
///
/// This `impl` block provides methods for applying binary operations between tensors.
/// There are several main categories:
///
/// * **Arithmetic operations with autograd**: `add`, `sub`, `mul`, `div` - basic mathematical operations
/// * **Comparison operations with autograd**: `maximum`, `minimum` - element-wise comparison
/// * **Logical operations (no autograd)**: `logical_and`, `logical_or`, `logical_xor` - boolean logic
/// * **Relational operations (no autograd)**: `eq`, `ne`, `lt`, `le`, `gt`, `ge` - comparison predicates
/// * **In-place operations (no autograd)**: `add_`, `sub_`, `mul_`, `div_` - modify tensors in place
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Autograd support:**
/// * **With autograd**: `add`, `sub`, `mul`, `div`, `maximum`, `minimum` - support gradient computation
/// * **No autograd**: Logical, relational, and in-place operations do not support gradient computation
///
/// **Key features:**
/// * **Broadcasting**: Tensors with different shapes are automatically broadcast
/// * **Type promotion**: Data types are promoted following standard rules
/// * **Device compatibility**: Operations require tensors on the same device
/// * **In-place operations**: Modify the left operand directly for memory efficiency
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct computation
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **Type promotion rules:**
/// * Integer division promotes to F32 for precision
/// * Logical operations always return BOOL tensors
/// * Mixed-type operations follow standard promotion hierarchy
/// * Boolean tensors are promoted to U8 before computation
impl Tensor {
    /// Runs [`try_add`](Self::try_add) and panics on failure.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow unchanged to both operands during backpropagation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[4.0, 5.0, 6.0]);
    /// let result = a.add(&b);
    /// // Result: [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn add(&self, rhs: &Self) -> Self {
        self.try_add(rhs).expect("failed to add tensors")
    }

    /// Runs [`try_sub`](Self::try_sub) and panics on failure.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Left operand receives gradient unchanged, right operand receives negated gradient.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[5.0, 7.0, 9.0]);
    /// let b = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let result = a.sub(&b);
    /// // Result: [4.0, 5.0, 6.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn sub(&self, rhs: &Self) -> Self {
        self.try_sub(rhs).expect("failed to subtract tensors")
    }

    /// Runs [`try_mul`](Self::try_mul) and panics on failure.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Each operand receives gradient multiplied by the other operand's value.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[2.0, 3.0, 4.0]);
    /// let b = Tensor::new(&[5.0, 6.0, 7.0]);
    /// let result = a.mul(&b);
    /// // Result: [10.0, 18.0, 28.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn mul(&self, rhs: &Self) -> Self {
        self.try_mul(rhs).expect("failed to multiply tensors")
    }

    /// Runs [`try_div`](Self::try_div) and panics on failure.
    ///
    /// Division of integer tensors promotes the result to F32 for precision.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Left operand receives `grad / rhs`, right operand receives `-grad * lhs / rhs²`.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[6.0, 8.0, 10.0]);
    /// let b = Tensor::new(&[2.0, 4.0, 5.0]);
    /// let result = a.div(&b);
    /// // Result: [3.0, 2.0, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn div(&self, rhs: &Self) -> Self {
        self.try_div(rhs).expect("failed to divide tensors")
    }

    /// Runs [`try_maximum`](Self::try_maximum) and panics on failure.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow to the operand with the larger value, with equal distribution for tied values.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 5.0, 3.0]);
    /// let b = Tensor::new(&[4.0, 2.0, 6.0]);
    /// let result = a.maximum(&b);
    /// // Result: [4.0, 5.0, 6.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn maximum(&self, rhs: &Self) -> Self {
        self.try_maximum(rhs).expect("failed to compute maximum")
    }

    /// Runs [`try_minimum`](Self::try_minimum) and panics on failure.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow to the operand with the smaller value, with equal distribution for tied values.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 5.0, 3.0]);
    /// let b = Tensor::new(&[4.0, 2.0, 6.0]);
    /// let result = a.minimum(&b);
    /// // Result: [1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn minimum(&self, rhs: &Self) -> Self {
        self.try_minimum(rhs).expect("failed to compute minimum")
    }

    /// Runs [`try_logical_and`](Self::try_logical_and) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise logical AND operation.
    /// **Note**: Logical operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[true, false, true]);
    /// let b = Tensor::new(&[true, true, false]);
    /// let result = a.logical_and(&b);
    /// // Result: [true, false, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn logical_and(&self, rhs: &Self) -> Self {
        self.try_logical_and(rhs).expect("failed to compute logical_and")
    }

    /// Runs [`try_logical_or`](Self::try_logical_or) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise logical OR operation.
    /// **Note**: Logical operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[true, false, true]);
    /// let b = Tensor::new(&[false, true, false]);
    /// let result = a.logical_or(&b);
    /// // Result: [true, true, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn logical_or(&self, rhs: &Self) -> Self {
        self.try_logical_or(rhs).expect("failed to compute logical_or")
    }

    /// Runs [`try_logical_xor`](Self::try_logical_xor) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise logical XOR operation.
    /// **Note**: Logical operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[true, false, true]);
    /// let b = Tensor::new(&[true, true, false]);
    /// let result = a.logical_xor(&b);
    /// // Result: [false, true, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn logical_xor(&self, rhs: &Self) -> Self {
        self.try_logical_xor(rhs).expect("failed to compute logical_xor")
    }

    /// Runs [`try_eq`](Self::try_eq) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise equality comparison.
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[1.0, 5.0, 3.0]);
    /// let result = a.eq(&b);
    /// // Result: [true, false, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn eq(&self, rhs: &Self) -> Self {
        self.try_eq(rhs).expect("failed to compute eq")
    }

    /// Runs [`try_ne`](Self::try_ne) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise inequality comparison.
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[1.0, 5.0, 3.0]);
    /// let result = a.ne(&b);
    /// // Result: [false, true, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn ne(&self, rhs: &Self) -> Self {
        self.try_ne(rhs).expect("failed to compute ne")
    }

    /// Runs [`try_lt`](Self::try_lt) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise less than comparison.
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[2.0, 2.0, 1.0]);
    /// let result = a.lt(&b);
    /// // Result: [true, false, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn lt(&self, rhs: &Self) -> Self {
        self.try_lt(rhs).expect("failed to compute lt")
    }

    /// Runs [`try_le`](Self::try_le) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise less than or equal comparison.
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[2.0, 2.0, 1.0]);
    /// let result = a.le(&b);
    /// // Result: [true, true, false]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn le(&self, rhs: &Self) -> Self {
        self.try_le(rhs).expect("failed to compute le")
    }

    /// Runs [`try_gt`](Self::try_gt) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise greater than comparison.
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[2.0, 2.0, 1.0]);
    /// let result = a.gt(&b);
    /// // Result: [false, false, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn gt(&self, rhs: &Self) -> Self {
        self.try_gt(rhs).expect("failed to compute gt")
    }

    /// Runs [`try_ge`](Self::try_ge) and panics on failure.
    ///
    /// Returns a BOOL tensor with element-wise greater than or equal comparison.
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[2.0, 2.0, 1.0]);
    /// let result = a.ge(&b);
    /// // Result: [false, true, true]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn ge(&self, rhs: &Self) -> Self {
        self.try_ge(rhs).expect("failed to compute ge")
    }

    /// Runs [`try_add_`](Self::try_add_) and panics on failure.
    ///
    /// Modifies this tensor in-place by adding the right-hand side tensor.
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let mut a = Tensor::new(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::new(&[4.0, 5.0, 6.0]);
    /// a.add_(&b);
    /// // a is now [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When in-place operation fails
    pub fn add_(&mut self, rhs: &Self) {
        self.try_add_(rhs).expect("failed to add_ tensors")
    }

    /// Runs [`try_sub_`](Self::try_sub_) and panics on failure.
    ///
    /// Modifies this tensor in-place by subtracting the right-hand side tensor.
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let mut a = Tensor::new(&[5.0, 7.0, 9.0]);
    /// let b = Tensor::new(&[1.0, 2.0, 3.0]);
    /// a.sub_(&b);
    /// // a is now [4.0, 5.0, 6.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When in-place operation fails
    pub fn sub_(&mut self, rhs: &Self) {
        self.try_sub_(rhs).expect("failed to sub_ tensors")
    }

    /// Runs [`try_mul_`](Self::try_mul_) and panics on failure.
    ///
    /// Modifies this tensor in-place by multiplying with the right-hand side tensor.
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let mut a = Tensor::new(&[2.0, 3.0, 4.0]);
    /// let b = Tensor::new(&[5.0, 6.0, 7.0]);
    /// a.mul_(&b);
    /// // a is now [10.0, 18.0, 28.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When in-place operation fails
    pub fn mul_(&mut self, rhs: &Self) {
        self.try_mul_(rhs).expect("failed to mul_ tensors")
    }

    /// Runs [`try_div_`](Self::try_div_) and panics on failure.
    ///
    /// Modifies this tensor in-place by dividing by the right-hand side tensor.
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// let mut a = Tensor::new(&[6.0, 8.0, 10.0]);
    /// let b = Tensor::new(&[2.0, 4.0, 5.0]);
    /// a.div_(&b);
    /// // a is now [3.0, 2.0, 2.0]
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When broadcasting fails due to incompatible shapes
    /// * When in-place operation fails
    pub fn div_(&mut self, rhs: &Self) {
        self.try_div_(rhs).expect("failed to div_ tensors")
    }

    /// Attempts to compute element-wise addition between tensors.
    ///
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules. The result has the broadcasted
    /// shape and the promoted data type.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow unchanged to both operands during backpropagation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn add_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[4.0, 5.0, 6.0]);
    ///     let result = a.try_add(&b)?;
    ///     // Result: [5.0, 7.0, 9.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_add(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_add(&rhs, target_tid, target_dtype, target_layout)?
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "add",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_add(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                result.into_iter().next().unwrap()
            },
        };

        if lhs.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if lhs.requires_grad() {
                let lhs_shape = lhs.shape();
                add_to_backward_graph("add_backward_lhs", &result_grad, &lhs, &lhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id);
                    Ok(result)
                })?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                add_to_backward_graph("add_backward_rhs", &result_grad, &rhs, &rhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id);
                    Ok(result)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to compute element-wise subtraction between tensors.
    ///
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules. The result has the broadcasted
    /// shape and the promoted data type.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Left operand receives gradient unchanged, right operand receives negated gradient.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn sub_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[5.0, 7.0, 9.0]);
    ///     let b = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let result = a.try_sub(&b)?;
    ///     // Result: [4.0, 5.0, 6.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_sub(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_sub(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "sub",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_sub(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if lhs.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if lhs.requires_grad() {
                let lhs_shape = lhs.shape();
                add_to_backward_graph("sub_backward_lhs", &result_grad, &lhs, &lhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id);
                    Ok(result)
                })?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                add_to_backward_graph("sub_backward_rhs", &result_grad, &rhs, &rhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id).try_neg()?;
                    Ok(result)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to compute element-wise multiplication between tensors.
    ///
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules. The result has the broadcasted
    /// shape and the promoted data type.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Each operand receives gradient multiplied by the other operand's value.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn mul_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[2.0, 3.0, 4.0]);
    ///     let b = Tensor::new(&[5.0, 6.0, 7.0]);
    ///     let result = a.try_mul(&b)?;
    ///     // Result: [10.0, 18.0, 28.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_mul(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_mul(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "mul",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_mul(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if lhs.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if lhs.requires_grad() {
                let lhs_shape = lhs.shape();
                let rhs_clone = rhs.clone();
                add_to_backward_graph("mul_backward_lhs", &result_grad, &lhs, &lhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id).try_mul(&rhs_clone)?;
                    Ok(result)
                })?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                let lhs_clone = lhs.clone();
                add_to_backward_graph("mul_backward_rhs", &result_grad, &rhs, &rhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id).try_mul(&lhs_clone)?;
                    Ok(result)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to compute element-wise division between tensors.
    ///
    /// Division of integer tensors promotes the result to F32 for precision.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Left operand receives `grad / rhs`, right operand receives `-grad * lhs / rhs²`.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn div_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[6.0, 8.0, 10.0]);
    ///     let b = Tensor::new(&[2.0, 4.0, 5.0]);
    ///     let result = a.try_div(&b)?;
    ///     // Result: [3.0, 2.0, 2.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_div(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = if self.dtype().is_int() && rhs.dtype().is_int() {
            DType::F32
        } else {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        };
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_div(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "div",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_div(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if lhs.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if lhs.requires_grad() {
                let lhs_shape = lhs.shape();
                let rhs_clone = rhs.clone();
                add_to_backward_graph("div_backward_lhs", &result_grad, &lhs, &lhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id).try_div(&rhs_clone)?;
                    Ok(result)
                })?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                let lhs_clone = lhs.clone();
                let rhs_clone = rhs.clone();
                add_to_backward_graph("div_backward_rhs", &result_grad, &rhs, &rhs_shape, move |grad_out_id| {
                    eager!();
                    let result = Tensor(*grad_out_id)
                        .try_mul(&lhs_clone)?
                        .try_div(&rhs_clone.try_square()?)?
                        .try_neg()?;
                    Ok(result)
                })?;
            }
        }

        Ok(result)
    }

    /// Attempts to compute element-wise maximum between tensors.
    ///
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules. The result has the broadcasted
    /// shape and the promoted data type.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow to the operand with the larger value, with equal distribution for tied values.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn maximum_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 5.0, 3.0]);
    ///     let b = Tensor::new(&[4.0, 2.0, 6.0]);
    ///     let result = a.try_maximum(&b)?;
    ///     // Result: [4.0, 5.0, 6.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_maximum(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_maximum(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "maximum",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_maximum(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if lhs.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if lhs.requires_grad() {
                let lhs_shape = lhs.shape();
                let lhs_clone = lhs.clone();
                let rhs_clone = rhs.clone();
                add_to_backward_graph(
                    "maximum_backward_lhs",
                    &result_grad,
                    &lhs,
                    &lhs_shape,
                    move |grad_out_id| {
                        eager!();
                        let lhs_mask = lhs_clone.try_gt(&rhs_clone)?;
                        let equal_mask = lhs_clone.try_eq(&rhs_clone)?;
                        let equal_half = equal_mask.try_mul_scalar(0.5)?;

                        let result = Tensor(*grad_out_id)
                            .try_mul(&lhs_mask)?
                            .try_add(&equal_half.try_mul(&Tensor(*grad_out_id))?)?;
                        Ok(result)
                    },
                )?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                let lhs_clone = lhs.clone();
                let rhs_clone = rhs.clone();
                add_to_backward_graph(
                    "maximum_backward_rhs",
                    &result_grad,
                    &rhs,
                    &rhs_shape,
                    move |grad_out_id| {
                        eager!();
                        let rhs_mask = rhs_clone.try_gt(&lhs_clone)?;
                        let equal_mask = lhs_clone.try_eq(&rhs_clone)?;
                        let equal_half = equal_mask.try_mul_scalar(0.5)?;

                        let result = Tensor(*grad_out_id)
                            .try_mul(&rhs_mask)?
                            .try_add(&equal_half.try_mul(&Tensor(*grad_out_id))?)?;
                        Ok(result)
                    },
                )?;
            }
        }

        Ok(result)
    }

    /// Attempts to compute element-wise minimum between tensors.
    ///
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules. The result has the broadcasted
    /// shape and the promoted data type.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients flow to the operand with the smaller value, with equal distribution for tied values.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn minimum_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 5.0, 3.0]);
    ///     let b = Tensor::new(&[4.0, 2.0, 6.0]);
    ///     let result = a.try_minimum(&b)?;
    ///     // Result: [1.0, 2.0, 3.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    pub fn try_minimum(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_minimum(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "minimum",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_minimum(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }?;

        if lhs.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if lhs.requires_grad() {
                let lhs_shape = lhs.shape();
                let lhs_clone = lhs.clone();
                let rhs_clone = rhs.clone();
                add_to_backward_graph(
                    "minimum_backward_lhs",
                    &result_grad,
                    &lhs,
                    &lhs_shape,
                    move |grad_out_id| {
                        eager!();
                        let lhs_mask = lhs_clone.try_lt(&rhs_clone)?;
                        let equal_mask = lhs_clone.try_eq(&rhs_clone)?;
                        let equal_half = equal_mask.try_mul_scalar(0.5)?;

                        let result = Tensor(*grad_out_id)
                            .try_mul(&lhs_mask)?
                            .try_add(&equal_half.try_mul(&Tensor(*grad_out_id))?)?;
                        Ok(result)
                    },
                )?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                let lhs_clone = lhs.clone();
                let rhs_clone = rhs.clone();
                add_to_backward_graph(
                    "minimum_backward_rhs",
                    &result_grad,
                    &rhs,
                    &rhs_shape,
                    move |grad_out_id| {
                        eager!();
                        let rhs_mask = rhs_clone.try_lt(&lhs_clone)?;
                        let equal_mask = lhs_clone.try_eq(&rhs_clone)?;
                        let equal_half = equal_mask.try_mul_scalar(0.5)?;

                        let result = Tensor(*grad_out_id)
                            .try_mul(&rhs_mask)?
                            .try_add(&equal_half.try_mul(&Tensor(*grad_out_id))?)?;
                        Ok(result)
                    },
                )?;
            }
        }

        Ok(result)
    }

    /// Attempts to compute element-wise logical AND between tensors.
    ///
    /// Returns a BOOL tensor with element-wise logical AND operation.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Logical operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn logical_and_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[true, false, true]);
    ///     let b = Tensor::new(&[true, true, false]);
    ///     let result = a.try_logical_and(&b)?;
    ///     // Result: [true, false, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_logical_and(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_logical_and(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "logical_and",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_logical_and(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise logical OR between tensors.
    ///
    /// Returns a BOOL tensor with element-wise logical OR operation.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Logical operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn logical_or_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[true, false, true]);
    ///     let b = Tensor::new(&[false, true, false]);
    ///     let result = a.try_logical_or(&b)?;
    ///     // Result: [true, true, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_logical_or(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_logical_or(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "logical_or",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_logical_or(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise logical XOR between tensors.
    ///
    /// Returns a BOOL tensor with element-wise logical XOR operation.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Logical operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn logical_xor_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[true, false, true]);
    ///     let b = Tensor::new(&[true, true, false]);
    ///     let result = a.try_logical_xor(&b)?;
    ///     // Result: [false, true, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_logical_xor(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_logical_xor(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "logical_xor",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_logical_xor(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise equality comparison between tensors.
    ///
    /// Returns a BOOL tensor with element-wise equality comparison.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn eq_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[1.0, 5.0, 3.0]);
    ///     let result = a.try_eq(&b)?;
    ///     // Result: [true, false, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_eq(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_eq(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "eq",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_eq(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise inequality comparison between tensors.
    ///
    /// Returns a BOOL tensor with element-wise inequality comparison.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn ne_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[1.0, 5.0, 3.0]);
    ///     let result = a.try_ne(&b)?;
    ///     // Result: [false, true, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_ne(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_ne(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "ne",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_ne(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise less than comparison between tensors.
    ///
    /// Returns a BOOL tensor with element-wise less than comparison.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn lt_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[2.0, 2.0, 1.0]);
    ///     let result = a.try_lt(&b)?;
    ///     // Result: [true, false, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_lt(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_lt(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "lt",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_lt(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise less than or equal comparison between tensors.
    ///
    /// Returns a BOOL tensor with element-wise less than or equal comparison.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn le_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[2.0, 2.0, 1.0]);
    ///     let result = a.try_le(&b)?;
    ///     // Result: [true, true, false]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_le(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_le(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "le",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_le(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise greater than comparison between tensors.
    ///
    /// Returns a BOOL tensor with element-wise greater than comparison.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn gt_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[2.0, 2.0, 1.0]);
    ///     let result = a.try_gt(&b)?;
    ///     // Result: [false, false, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_gt(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_gt(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "gt",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_gt(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to compute element-wise greater than or equal comparison between tensors.
    ///
    /// Returns a BOOL tensor with element-wise greater than or equal comparison.
    /// This operation broadcasts tensors to compatible shapes if needed and promotes
    /// data types according to standard promotion rules.
    ///
    /// **Note**: Comparison operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn ge_tensors() -> Result<()> {
    ///     let a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[2.0, 2.0, 1.0]);
    ///     let result = a.try_ge(&b)?;
    ///     // Result: [false, true, true]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    pub fn try_ge(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs.device(),
                    dtype: result_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);
                lhs.execute_ge(&rhs, target_tid, target_dtype, target_layout)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "ge",
                    &[&lhs, &rhs],
                    &[lhs.device()],
                    &[result_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);
                        let result = lhs_tensor.execute_ge(
                            &rhs_tensor,
                            target_tids[0],
                            target_dtype,
                            target_layout_clone.clone(),
                        )?;
                        Ok(vec![result.id()])
                    },
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to add another tensor to this tensor in-place.
    ///
    /// In eager mode, performs the operation directly on the tensor's buffer.
    /// In lazy mode, replaces this tensor with the result of the add operation.
    ///
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn add_inplace() -> Result<()> {
    ///     let mut a = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     let b = Tensor::new(&[4.0, 5.0, 6.0]);
    ///     a.try_add_(&b)?;
    ///     // a is now [5.0, 7.0, 9.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - In-place operation fails
    /// - Graph operations fail in lazy mode
    pub fn try_add_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::add_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_add(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    /// Attempts to subtract another tensor from this tensor in-place.
    ///
    /// In eager mode, performs the operation directly on the tensor's buffer.
    /// In lazy mode, replaces this tensor with the result of the sub operation.
    ///
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn sub_inplace() -> Result<()> {
    ///     let mut a = Tensor::new(&[5.0, 7.0, 9.0]);
    ///     let b = Tensor::new(&[1.0, 2.0, 3.0]);
    ///     a.try_sub_(&b)?;
    ///     // a is now [4.0, 5.0, 6.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - In-place operation fails
    /// - Graph operations fail in lazy mode
    pub fn try_sub_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::sub_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_sub(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    /// Attempts to multiply this tensor with another tensor in-place.
    ///
    /// In eager mode, performs the operation directly on the tensor's buffer.
    /// In lazy mode, replaces this tensor with the result of the mul operation.
    ///
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn mul_inplace() -> Result<()> {
    ///     let mut a = Tensor::new(&[2.0, 3.0, 4.0]);
    ///     let b = Tensor::new(&[5.0, 6.0, 7.0]);
    ///     a.try_mul_(&b)?;
    ///     // a is now [10.0, 18.0, 28.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - In-place operation fails
    /// - Graph operations fail in lazy mode
    pub fn try_mul_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::mul_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_mul(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    /// Attempts to divide this tensor by another tensor in-place.
    ///
    /// In eager mode, performs the operation directly on the tensor's buffer.
    /// In lazy mode, replaces this tensor with the result of the div operation.
    ///
    /// **Note**: In-place operations do not support gradient computation.
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor; // Assuming Tensor is in the current crate root
    ///
    /// fn div_inplace() -> Result<()> {
    ///     let mut a = Tensor::new(&[6.0, 8.0, 10.0]);
    ///     let b = Tensor::new(&[2.0, 4.0, 5.0]);
    ///     a.try_div_(&b)?;
    ///     // a is now [3.0, 2.0, 2.0]
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Broadcasting fails due to incompatible shapes
    /// - In-place operation fails
    /// - Graph operations fail in lazy mode
    pub fn try_div_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::div_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_div(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    impl_binary_execute!(execute_add, maidenx_core::be::ops::binary::add);
    impl_binary_execute!(execute_sub, maidenx_core::be::ops::binary::sub);
    impl_binary_execute!(execute_mul, maidenx_core::be::ops::binary::mul);
    impl_binary_execute!(execute_div, maidenx_core::be::ops::binary::div);
    impl_binary_execute!(execute_maximum, maidenx_core::be::ops::binary::maximum);
    impl_binary_execute!(execute_minimum, maidenx_core::be::ops::binary::minimum);
    impl_binary_execute!(execute_logical_and, maidenx_core::be::ops::binary::logical_and);
    impl_binary_execute!(execute_logical_or, maidenx_core::be::ops::binary::logical_or);
    impl_binary_execute!(execute_logical_xor, maidenx_core::be::ops::binary::logical_xor);
    impl_binary_execute!(execute_eq, maidenx_core::be::ops::binary::eq);
    impl_binary_execute!(execute_ne, maidenx_core::be::ops::binary::ne);
    impl_binary_execute!(execute_lt, maidenx_core::be::ops::binary::lt);
    impl_binary_execute!(execute_le, maidenx_core::be::ops::binary::le);
    impl_binary_execute!(execute_gt, maidenx_core::be::ops::binary::gt);
    impl_binary_execute!(execute_ge, maidenx_core::be::ops::binary::ge);

    fn prepare_binary_metadata(lhs: &Tensor, rhs: &Tensor) -> Vec<usize> {
        let mut metadata = Vec::new();
        metadata.extend_from_slice(&lhs.shape());
        metadata.extend_from_slice(&lhs.strides());
        metadata.extend_from_slice(&rhs.strides());
        metadata.push(lhs.offset());
        metadata.push(rhs.offset());
        metadata
    }
}
