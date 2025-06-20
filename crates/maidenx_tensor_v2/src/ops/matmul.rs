use crate::{
    eager, get_mode, insert_metadata, insert_storage, link_tensor_to_storage, next_storage_id, next_tensor_id,
    utils::{
        graph::{add_to_backward_graph, add_to_forward_graph},
        promotion::get_promoted_dtype,
        tensor::update_tensor_status,
    },
    Tensor, TensorMetadata, TensorMode, TensorStorage, TensorUpdateStatus,
};
use maidenx_core::{
    buffer::BufferManager,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::Arc;

/// ## Tensor Matrix Multiplication Operations
///
/// This `impl` block provides methods for performing matrix multiplication between tensors.
/// Matrix multiplication is a fundamental operation in linear algebra and neural networks,
/// supporting various tensor shapes through broadcasting and automatic shape unification.
///
/// **Key features:**
/// * **Shape flexibility**: Supports 1D vectors, 2D matrices, and higher-dimensional batch operations
/// * **Broadcasting**: Automatically handles batch dimensions through broadcasting rules
/// * **Type promotion**: Intelligent type promotion to prevent overflow in integer operations
/// * **Shape unification**: Automatically reshapes inputs to compatible matrix multiplication forms
/// * **Device compatibility**: Operations require tensors on the same device
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct computation
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **Type promotion rules:**
/// * Integer×Integer operations are promoted to larger integer types to prevent overflow
/// * Mixed-type operations follow standard promotion hierarchy
/// * Boolean tensors are automatically promoted before computation
///
/// **Shape handling:**
/// * 1D×1D: Vector dot product, results in scalar
/// * 1D×2D: Vector-matrix multiplication, 1D is treated as row vector
/// * 2D×1D: Matrix-vector multiplication, 1D is treated as column vector
/// * 2D×2D: Standard matrix multiplication
/// * Higher dimensions: Batch matrix multiplication with broadcasting on leading dimensions
///
/// **Gradient computation:**
/// - **Left operand**: `grad_out @ rhs.T` (gradient times transpose of right operand)
/// - **Right operand**: `lhs.T @ grad_out` (transpose of left operand times gradient)
/// - Automatic shape adjustment and type promotion for gradient tensors
/// - Broadcasting dimension reduction when necessary
///
/// **Mathematical notation:**
/// * For matrices A(m×k) and B(k×n), result is C(m×n) where C[i,j] = Σ(A[i,l] * B[l,j])
/// * Batch operations apply the same rule to each matrix in the batch
impl Tensor {
    /// Runs [`try_matmul`](Self::try_matmul) and panics on failure.
    ///
    /// Performs matrix multiplication between two tensors with automatic shape handling
    /// and type promotion.
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients are computed using transpose operations: `grad_out @ rhs.T` for left operand
    /// and `lhs.T @ grad_out` for right operand.
    ///
    /// # Examples
    /// ```rust
    /// use crate::Tensor;
    ///
    /// // 2D matrix multiplication
    /// let a = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]]); // 2×2
    /// let b = Tensor::new(&[[5.0, 6.0], [7.0, 8.0]]); // 2×2
    /// let result = a.matmul(&b); // 2×2 result
    ///
    /// // Vector-matrix multiplication
    /// let v = Tensor::new(&[1.0, 2.0]); // 1D vector
    /// let m = Tensor::new(&[[3.0, 4.0], [5.0, 6.0]]); // 2×2 matrix
    /// let result = v.matmul(&m); // 1D result
    ///
    /// // Batch matrix multiplication
    /// let batch_a = Tensor::new(&[[[1.0, 2.0]], [[3.0, 4.0]]]); // 2×1×2
    /// let batch_b = Tensor::new(&[[[5.0], [6.0]], [[7.0], [8.0]]]); // 2×2×1
    /// let result = batch_a.matmul(&batch_b); // 2×1×1
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensors are on different devices
    /// * When inner dimensions don't match for multiplication
    /// * When memory allocation fails
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When gradient computation setup fails
    pub fn matmul(&self, rhs: &Self) -> Tensor {
        self.try_matmul(rhs).expect("failed to perform matrix multiplication")
    }

    /// Attempts to perform matrix multiplication between two tensors.
    ///
    /// This operation implements full matrix multiplication with support for:
    /// - Vector-vector, vector-matrix, matrix-vector, and matrix-matrix operations
    /// - Batch matrix multiplication with broadcasting
    /// - Automatic type promotion to prevent integer overflow
    /// - Shape unification and dimension squeezing for vectors
    ///
    /// **Autograd support**: If either tensor requires gradients, the result will
    /// automatically enable gradient computation and register backward functions.
    /// Gradients are computed using transpose operations with automatic shape adjustment
    /// and type promotion to match original tensor types.
    ///
    /// # Parameters
    /// * `rhs`: The right-hand side tensor for multiplication
    ///
    /// # Examples
    /// ```rust
    /// use maidenx_core::error::Result;
    /// use crate::Tensor;
    ///
    /// fn matrix_multiply() -> Result<()> {
    ///     // Standard matrix multiplication
    ///     let a = Tensor::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // 2×3
    ///     let b = Tensor::new(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]); // 3×2
    ///     let result = a.try_matmul(&b)?; // 2×2 result
    ///
    ///     // Batch multiplication with broadcasting
    ///     let batch_a = Tensor::new(&[[[1.0, 2.0]], [[3.0, 4.0]]]); // 2×1×2
    ///     let single_b = Tensor::new(&[[5.0], [6.0]]); // 2×1
    ///     let result = batch_a.try_matmul(&single_b)?; // Broadcasting: 2×1×1
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensors are on different devices
    /// - Inner dimensions don't match (A[..., k] × B[..., k, ...])
    /// - Batch dimensions cannot be broadcast together
    /// - Memory allocation fails
    /// - Backend computation fails
    /// - Graph operations fail in lazy mode
    /// - Gradient computation setup fails
    ///
    /// # Type Promotion
    /// * U8×U8 → U16, I8×I8 → I16 (prevent 8-bit overflow)
    /// * U16×U16 → U32, I16×I16 → I32 (prevent 16-bit overflow)
    /// * Mixed integer types promote to larger signed type
    /// * Float types follow standard promotion rules
    /// * Gradient tensors are promoted to maintain compatibility
    ///
    /// # Shape Rules
    /// * Input shapes are unified by adding dimensions: 1D becomes 2D
    /// * Batch dimensions (all but last 2) are broadcast using standard rules
    /// * Output dimensions are squeezed back for vector inputs
    /// * Final shape: batch_shape + [M, N] where M×K and K×N are the core matrix dimensions
    ///
    /// # Gradient Computation Details
    /// * **Left gradient**: `grad_out @ rhs.transpose(-2, -1)`
    /// * **Right gradient**: `lhs.transpose(-2, -1) @ grad_out`
    /// * Shape adjustment through summation and reshaping when broadcasting was applied
    /// * Type conversion back to original operand types after gradient computation
    pub fn try_matmul(&self, rhs: &Self) -> Result<Tensor> {
        let l_ndim = self.ndim();
        let r_ndim = rhs.ndim();

        let target_dtype = match (self.dtype(), rhs.dtype()) {
            (DType::U8, DType::U8) => DType::U16,
            (DType::U8, DType::I8) | (DType::I8, DType::U8) => DType::I16,
            (DType::I8, DType::I8) => DType::I16,
            (DType::U16, DType::U16) => DType::U32,
            (DType::U16, DType::I16) | (DType::I16, DType::U16) => DType::I32,
            (DType::I16, DType::I16) => DType::I32,
            (dtype1, dtype2) if dtype1 != dtype2 => get_promoted_dtype(dtype1, dtype2),
            (dtype, _) => dtype,
        };

        let lhs = if self.dtype() != target_dtype {
            self.try_to_dtype(target_dtype)?
        } else {
            self.clone()
        };
        let rhs = if rhs.dtype() != target_dtype {
            rhs.try_to_dtype(target_dtype)?
        } else {
            rhs.clone()
        };

        let (lhs_unified, rhs_unified) = unify_matmul_shapes(&lhs, &rhs)?;
        let (_, final_out_shape) = lhs_unified.prepare_matmul_metadata(&rhs_unified);
        let target_layout = Layout::from_shape(&final_out_shape);

        let mut result = match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = TensorMetadata {
                    device: lhs_unified.device(),
                    dtype: target_dtype,
                    layout: target_layout.clone(),
                    grad_tensor_id: None,
                    graph_id: None,
                    mode: get_mode(),
                    update_status: TensorUpdateStatus::Pending,
                };
                insert_metadata(target_tid, metadata);

                eager!();

                let target_size = target_layout.size();
                let mut buffer = BufferManager::create(target_size, lhs_unified.device(), target_dtype)?;

                {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    let lhs_tensor = if lhs_unified.dtype() != target_dtype {
                        lhs_unified.try_to_dtype(target_dtype)?
                    } else {
                        lhs_unified.clone()
                    };
                    let rhs_tensor = if rhs_unified.dtype() != target_dtype {
                        rhs_unified.try_to_dtype(target_dtype)?
                    } else {
                        rhs_unified.clone()
                    };

                    let (lhs_final, rhs_final) = unify_matmul_shapes(&lhs_tensor, &rhs_tensor)?;
                    let (metadata, _) = lhs_final.prepare_matmul_metadata(&rhs_final);

                    unsafe {
                        maidenx_core::be::ops::matmul::matmul(
                            buffer_mut,
                            lhs_final.storage()?.buffer(),
                            rhs_final.storage()?.buffer(),
                            target_size,
                            Some(&metadata),
                        )?;
                    }
                }

                let sid = next_storage_id();
                link_tensor_to_storage(target_tid, sid);
                insert_storage(sid, TensorStorage::new(buffer));

                update_tensor_status(target_tid, TensorUpdateStatus::Materialized)?;

                Tensor(target_tid)
            },
            TensorMode::Lazy => {
                let target_layout_clone = target_layout.clone();
                let result = add_to_forward_graph(
                    "matmul",
                    &[&lhs_unified, &rhs_unified],
                    &[lhs_unified.device()],
                    &[target_dtype],
                    &[target_layout],
                    move |input_tids, target_tids| {
                        let lhs_tensor = Tensor(input_tids[0]);
                        let rhs_tensor = Tensor(input_tids[1]);

                        eager!();

                        let target_size = target_layout_clone.size();
                        let mut buffer = BufferManager::create(target_size, lhs_tensor.device(), target_dtype)?;

                        {
                            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                            let lhs_final = if lhs_tensor.dtype() != target_dtype {
                                lhs_tensor.try_to_dtype(target_dtype)?
                            } else {
                                lhs_tensor.clone()
                            };
                            let rhs_final = if rhs_tensor.dtype() != target_dtype {
                                rhs_tensor.try_to_dtype(target_dtype)?
                            } else {
                                rhs_tensor.clone()
                            };

                            let (lhs_unified, rhs_unified) = unify_matmul_shapes(&lhs_final, &rhs_final)?;
                            let (metadata, _) = lhs_unified.prepare_matmul_metadata(&rhs_unified);

                            unsafe {
                                maidenx_core::be::ops::matmul::matmul(
                                    buffer_mut,
                                    lhs_unified.storage()?.buffer(),
                                    rhs_unified.storage()?.buffer(),
                                    target_size,
                                    Some(&metadata),
                                )?;
                            }
                        }

                        let sid = next_storage_id();
                        link_tensor_to_storage(target_tids[0], sid);
                        insert_storage(sid, TensorStorage::new(buffer));

                        update_tensor_status(target_tids[0], TensorUpdateStatus::Materialized)?;

                        Ok(vec![target_tids[0]])
                    },
                )?;
                result.into_iter().next().unwrap()
            },
        };

        if l_ndim == 1 && r_ndim == 1 {
            result = result.try_squeeze_all()?;
        } else if l_ndim == 1 && r_ndim >= 2 {
            let last_dim = result.ndim() - 2;
            if result.shape()[last_dim] == 1 {
                result = result.try_squeeze(&[last_dim])?;
            }
        } else if l_ndim >= 2 && r_ndim == 1 {
            let last_dim = result.ndim() - 1;
            if result.shape()[last_dim] == 1 {
                result = result.try_squeeze(&[last_dim])?;
            }
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.try_enable_grad()?;
            let result_grad = result.grad();

            if self.requires_grad() {
                let lhs_shape = self.shape();
                let rhs_for_lhs = rhs.clone();
                add_to_backward_graph(
                    "matmul_backward_lhs",
                    &result_grad,
                    self,
                    &lhs_shape,
                    move |grad_out_id| {
                        eager!();
                        let grad_out = Tensor(*grad_out_id);
                        let rhs_t = rhs_for_lhs.try_transpose(-2, -1)?;
                        let grad_lhs = grad_out.try_matmul(&rhs_t)?;
                        Ok(grad_lhs)
                    },
                )?;
            }

            if rhs.requires_grad() {
                let rhs_shape = rhs.shape();
                let lhs_for_rhs = self.clone();
                add_to_backward_graph(
                    "matmul_backward_rhs",
                    &result_grad,
                    &rhs,
                    &rhs_shape,
                    move |grad_out_id| {
                        eager!();
                        let grad_out = Tensor(*grad_out_id);
                        let lhs_t = lhs_for_rhs.try_transpose(-2, -1)?;
                        let grad_rhs = lhs_t.try_matmul(&grad_out)?;
                        Ok(grad_rhs)
                    },
                )?;
            }
        }

        Ok(result)
    }

    fn prepare_matmul_metadata(&self, rhs: &Tensor) -> (Vec<usize>, Vec<usize>) {
        let (mut a_shape, mut b_shape) = (self.shape().to_vec(), rhs.shape().to_vec());
        let (mut a_strides, mut b_strides) = (self.strides().to_vec(), rhs.strides().to_vec());

        let mut out_shape = vec![];

        if a_shape.len() == 1 {
            a_shape.insert(0, 1);
            a_strides.insert(0, a_strides[0] * a_shape[1]);
        }

        if b_shape.len() == 1 {
            b_shape.push(1);
            b_strides.push(1);
        }

        let out_ndim = a_shape.len();

        out_shape.extend_from_slice(&a_shape[..out_ndim - 2]);
        out_shape.push(a_shape[out_ndim - 2]);
        out_shape.push(b_shape[out_ndim - 1]);

        let mut metadata = Vec::new();
        metadata.push(out_ndim);
        metadata.push(a_shape.len());
        metadata.push(b_shape.len());
        metadata.extend_from_slice(&out_shape);
        metadata.extend_from_slice(&a_shape);
        metadata.extend_from_slice(&b_shape);
        metadata.extend_from_slice(&a_strides);
        metadata.extend_from_slice(&b_strides);
        metadata.push(self.offset());
        metadata.push(rhs.offset());

        (metadata, out_shape)
    }
}

#[allow(non_snake_case)]
fn unify_matmul_shapes(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let mut a_exp = a.clone();
    let mut b_exp = b.clone();
    if a_exp.ndim() == 1 {
        a_exp = a_exp.try_unsqueeze(&[0])?;
    }
    if b_exp.ndim() == 1 {
        b_exp = b_exp.try_unsqueeze(&[b_exp.ndim()])?;
    }
    let a_nd = a_exp.ndim();
    let b_nd = b_exp.ndim();
    if a_nd < 2 || b_nd < 2 {
        return Err(Error::InvalidShape {
            message: "matmul: both inputs must have at least 2D after expand".to_string(),
        });
    }
    let a_shape = a_exp.shape();
    let b_shape = b_exp.shape();
    let a_leading = &a_shape[..a_nd - 2];
    let b_leading = &b_shape[..b_nd - 2];
    let M = a_shape[a_nd - 2];
    let Ka = a_shape[a_nd - 1];
    let Kb = b_shape[b_nd - 2];
    let N = b_shape[b_nd - 1];
    if Ka != Kb {
        return Err(Error::InvalidShape {
            message: format!("Incompatible shapes for matmul (K mismatch): {} vs {}", Ka, Kb),
        });
    }
    let leading_bc_shape = broadcast_leading_dims(a_leading, b_leading)?;
    let a_bc = expand_to_batch_shape(&a_exp, &leading_bc_shape, M, Ka)?;
    let b_bc = expand_to_batch_shape(&b_exp, &leading_bc_shape, Kb, N)?;
    Ok((a_bc, b_bc))
}

fn broadcast_leading_dims(a_leading: &[usize], b_leading: &[usize]) -> Result<Vec<usize>> {
    let max_len = a_leading.len().max(b_leading.len());
    let mut shape = vec![0; max_len];
    for i in 0..max_len {
        let a_dim = if i < a_leading.len() {
            a_leading[a_leading.len() - 1 - i]
        } else {
            1
        };
        let b_dim = if i < b_leading.len() {
            b_leading[b_leading.len() - 1 - i]
        } else {
            1
        };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(Error::InvalidShape {
                message: format!("Cannot broadcast leading dims: mismatch {} vs {}", a_dim, b_dim),
            });
        }
        shape[max_len - 1 - i] = a_dim.max(b_dim);
    }
    Ok(shape)
}

fn expand_to_batch_shape(a_exp: &Tensor, leading_bc_shape: &[usize], last0: usize, last1: usize) -> Result<Tensor> {
    let mut new_shape = leading_bc_shape.to_vec();
    new_shape.push(last0);
    new_shape.push(last1);
    a_exp.try_broadcast(&new_shape)
}
