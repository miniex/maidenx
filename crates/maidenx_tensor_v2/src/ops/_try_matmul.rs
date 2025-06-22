use crate::{
    get_mode, insert_metadata, next_tensor_id,
    utils::{graph::add_to_graph, promotion::get_promoted_dtype},
    Tensor, TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};

impl Tensor {
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

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs_unified.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs_unified.execute_matmul(&rhs_unified, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph(
                    "matmul",
                    &[&lhs_unified, &rhs_unified],
                    &[&output],
                    move |inputs, outputs| {
                        inputs[0].execute_matmul(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                        Ok(())
                    },
                )?;
            },
        };

        let mut result = output;

        // Handle dimension adjustments based on input dimensions
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

        Ok(result)
    }

    fn execute_matmul(&self, rhs: &Tensor, target: &Tensor, target_dtype: DType, target_layout: Layout) -> Result<()> {
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

            let (lhs_unified, rhs_unified) = unify_matmul_shapes(&lhs_tensor, &rhs_tensor)?;
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

        let sid = crate::next_storage_id();
        crate::link_tensor_to_storage(target.id(), sid);
        crate::insert_storage(sid, crate::TensorStorage::new(buffer));

        crate::utils::tensor::update_tensor_status(target.id(), TensorUpdateStatus::Materialized)?;

        Ok(())
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
