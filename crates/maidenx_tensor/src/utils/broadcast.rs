use maidenx_core::error::{Error, Result};

use crate::Tensor;

pub fn compute_broadcast_shape(lhs_shape: &[usize], rhs_shape: &[usize]) -> Result<Vec<usize>> {
    // Special case: if lhs is a scalar (empty shape), use rhs shape
    if lhs_shape.is_empty() {
        return Ok(rhs_shape.to_vec());
    }

    // Special case: if rhs is a scalar (empty shape), use lhs shape
    if rhs_shape.is_empty() {
        return Ok(lhs_shape.to_vec());
    }

    let max_rank = lhs_shape.len().max(rhs_shape.len());
    let mut broadcasted_shape = Vec::with_capacity(max_rank);

    // Pad shapes with 1s
    let padded_lhs = pad_shape(lhs_shape, max_rank);
    let padded_rhs = pad_shape(rhs_shape, max_rank);

    // Compare dimensions and build output shape
    for (i, (&dim1, &dim2)) in padded_lhs.iter().zip(padded_rhs.iter()).enumerate() {
        if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
            return Err(Error::InvalidShape {
                message: format!("Cannot broadcast shapes {:?} and {:?} at dimension {}", lhs_shape, rhs_shape, i),
            });
        }
        broadcasted_shape.push(dim1.max(dim2));
    }

    Ok(broadcasted_shape)
}

pub fn pad_shape(shape: &[usize], target_rank: usize) -> Vec<usize> {
    let mut padded = vec![1; target_rank - shape.len()];
    padded.extend(shape);
    padded
}

pub fn broadcast_tensors(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    let output_ndim = a_ndim.max(b_ndim);
    let mut output_shape = vec![0; output_ndim];

    for i in 0..output_ndim {
        let a_dim = if i < a_ndim { a_shape[a_ndim - 1 - i] } else { 1 };
        let b_dim = if i < b_ndim { b_shape[b_ndim - 1 - i] } else { 1 };

        if a_dim == 1 || b_dim == 1 || a_dim == b_dim {
            output_shape[output_ndim - 1 - i] = a_dim.max(b_dim);
        } else {
            return Err(Error::InvalidShape {
                message: format!("Cannot broadcast shapes {:?} and {:?}", a_shape, b_shape),
            });
        }
    }

    let a_broadcasted = a.broadcast(&output_shape)?;
    let b_broadcasted = b.broadcast(&output_shape)?;

    Ok((a_broadcasted, b_broadcasted))
}
