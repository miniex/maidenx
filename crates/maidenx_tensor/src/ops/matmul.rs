use crate::{
    utils::promotion::{get_promoted_dtype, promote_tensor},
    Tensor, TensorNode,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
};

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();

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

        let lhs = promote_tensor(self, target_dtype)?;
        let rhs = promote_tensor(rhs, target_dtype)?;

        let (lhs, rhs) = unify_matmul_shapes(&lhs, &rhs)?;
        let (metadata, final_out_shape) = prepare_metadata(&lhs, &rhs);

        let mut result = Self::empty_with_spec(&final_out_shape, lhs.device(), lhs.dtype())?;

        unsafe {
            let num_els = final_out_shape.iter().product();
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::matmul::matmul(out_buf, lhs.buffer(), rhs.buffer(), num_els, Some(&metadata))?;

                Ok(())
            })?;
        }

        if l_ndim == 1 && r_ndim == 1 {
            result = result.squeeze_all()?;
        } else if l_ndim == 1 && r_ndim >= 2 {
            let last_dim = result.ndim() - 2;
            if result.shape()[last_dim] == 1 {
                result = result.squeeze(last_dim)?;
            }
        } else if l_ndim >= 2 && r_ndim == 1 {
            let last_dim = result.ndim() - 1;
            if result.shape()[last_dim] == 1 {
                result = result.squeeze(last_dim)?;
            }
        }

        if self.requires_grad() || rhs.requires_grad() {
            result.with_grad()?;

            let backward_fn = Box::new(move |inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let lhs = &inputs[0];
                let rhs = &inputs[1];

                let target_dtype = match (lhs.dtype(), rhs.dtype()) {
                    (DType::U8, DType::U8) => DType::U16,
                    (DType::U8, DType::I8) | (DType::I8, DType::U8) => DType::I16,
                    (DType::I8, DType::I8) => DType::I16,
                    (DType::U16, DType::U16) => DType::U32,
                    (DType::U16, DType::I16) | (DType::I16, DType::U16) => DType::I32,
                    (DType::I16, DType::I16) => DType::I32,
                    (dtype1, dtype2) if dtype1 != dtype2 => get_promoted_dtype(dtype1, dtype2),
                    (dtype, _) => dtype,
                };

                let target_dtype = if grad_out.dtype() != target_dtype {
                    get_promoted_dtype(target_dtype, grad_out.dtype())
                } else {
                    target_dtype
                };

                let lhs_promoted = promote_tensor(lhs, target_dtype)?;
                let rhs_promoted = promote_tensor(rhs, target_dtype)?;
                let grad_out_promoted = promote_tensor(grad_out, target_dtype)?;

                let (metadata, _) = prepare_metadata(&lhs_promoted, &rhs_promoted);

                let mut grad_lhs = Self::zeros_like(&lhs_promoted)?;
                let mut grad_rhs = Self::zeros_like(&rhs_promoted)?;

                let grad_lhs_size = grad_lhs.size();
                let grad_rhs_size = grad_rhs.size();

                unsafe {
                    grad_lhs.with_buffer_mut(|gl_buf| {
                        grad_rhs.with_buffer_mut(|gr_buf| {
                            maidenx_core::be::ops::matmul::matmul_backward(
                                Some(gl_buf),
                                Some(gr_buf),
                                grad_out_promoted.buffer(),
                                lhs_promoted.buffer(),
                                rhs_promoted.buffer(),
                                grad_lhs_size,
                                grad_rhs_size,
                                Some(&metadata),
                            )?;
                            Ok(())
                        })
                    })?;
                }

                let mut grad_lhs = grad_lhs;
                let mut grad_rhs = grad_rhs;

                // Reshape gradients to match original tensor shapes
                if grad_lhs.shape() != lhs.shape() {
                    while grad_lhs.ndim() > lhs.ndim() {
                        grad_lhs = grad_lhs.sum(0, false)?;
                    }
                    if grad_lhs.shape() != lhs.shape() {
                        grad_lhs = grad_lhs.reshape(lhs.shape())?;
                    }
                }

                if grad_rhs.shape() != rhs.shape() {
                    while grad_rhs.ndim() > rhs.ndim() {
                        grad_rhs = grad_rhs.sum(0, false)?;
                    }
                    if grad_rhs.shape() != rhs.shape() {
                        grad_rhs = grad_rhs.reshape(rhs.shape())?;
                    }
                }

                // Convert gradients back to original dtypes if needed
                if grad_lhs.dtype() != lhs.dtype() {
                    grad_lhs = promote_tensor(&grad_lhs, lhs.dtype())?;
                }

                if grad_rhs.dtype() != rhs.dtype() {
                    grad_rhs = promote_tensor(&grad_rhs, rhs.dtype())?;
                }

                Ok(vec![grad_lhs, grad_rhs])
            });

            let node = TensorNode::new(
                "matmul".to_string(),
                vec![original_lhs, original_rhs],
                Some(backward_fn),
            );
            result.node = Some(node);
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
fn unify_matmul_shapes(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let mut a_exp = a.clone();
    let mut b_exp = b.clone();
    if a_exp.ndim() == 1 {
        a_exp = a_exp.unsqueeze(0)?;
    }
    if b_exp.ndim() == 1 {
        b_exp = b_exp.unsqueeze(b_exp.ndim())?;
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
    a_exp.broadcast(&new_shape)
}

fn prepare_metadata(a: &Tensor, b: &Tensor) -> (Vec<usize>, Vec<usize>) {
    let (mut a_shape, mut b_shape) = (a.shape().to_vec(), b.shape().to_vec());
    let (mut a_strides, mut b_strides) = (a.strides().to_vec(), b.strides().to_vec());

    let mut out_shape = vec![];

    // Handle 1-dimensional inputs by reshaping them explicitly
    if a_shape.len() == 1 {
        a_shape.insert(0, 1); // [K] -> [1, K]
        a_strides.insert(0, a_strides[0] * a_shape[1]);
    }

    if b_shape.len() == 1 {
        b_shape.push(1); // [K] -> [K, 1]
        b_strides.push(1);
    }

    let out_ndim = a_shape.len();

    // Now safely access dimensions
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
    metadata.push(a.offset());
    metadata.push(b.offset());

    (metadata, out_shape)
}
