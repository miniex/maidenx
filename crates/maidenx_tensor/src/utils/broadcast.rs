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

pub fn broadcast_tensor(src: &Tensor, target: &Tensor) -> Result<Tensor> {
    if src.shape() != target.shape() {
        let mut t = src.clone();
        t.with_broadcast(target)?;

        Ok(t)
    } else {
        Ok(src.clone())
    }
}
