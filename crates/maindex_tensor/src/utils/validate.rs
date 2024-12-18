use crate::{
    error::{TensorError, TensorResult},
    Tensor,
};

pub fn assert_shape_match(a: &Tensor, b: &Tensor) -> TensorResult<()> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

pub fn assert_device_match(a: &Tensor, b: &Tensor) -> TensorResult<()> {
    if a.device != b.device {
        return Err(TensorError::DeviceMismatch {
            expected: a.device.name(),
            got: b.device.name(),
        });
    }
    Ok(())
}

pub fn assert_mat_mul_shape_match(a: &Tensor, b: &Tensor) -> TensorResult<()> {
    // Check that both tensors are 2D matrices
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.shape()[0], a.shape()[1]], // Expected 2D matrix
            got: if a.shape().len() != 2 {
                a.shape().to_vec()
            } else {
                b.shape().to_vec()
            },
        });
    }

    // For matrix multiplication: (M x K) * (K x N)
    // Number of columns in first matrix (K) must match number of rows in second matrix (K)
    if a.shape()[1] != b.shape()[0] {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.shape()[0], a.shape()[1]], // First matrix shape
            got: b.shape().to_vec(),                    // Second matrix shape
        });
    }

    Ok(())
}

pub fn assert_broadcast_shapes(input1: &Tensor, input2: &Tensor) -> TensorResult<()> {
    let broadcast_shape = calculate_broadcast_shape(&input1.shape, &input2.shape)?;
    if input1.shape != broadcast_shape {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "Tensor1 shape {:?} does not match broadcast shape {:?}",
                input1.shape, broadcast_shape
            ),
        });
    }
    if input2.shape != broadcast_shape {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "Tensor2 shape {:?} does not match broadcast shape {:?}",
                input2.shape, broadcast_shape
            ),
        });
    }
    Ok(())
}

pub fn calculate_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> TensorResult<Vec<usize>> {
    let mut result_shape = Vec::new();

    // Get the maximum rank of the two shapes
    let max_rank = std::cmp::max(shape1.len(), shape2.len());

    // Reverse iterate through both shapes to align dimensions
    for i in 0..max_rank {
        let dim1 = shape1
            .get(shape1.len().saturating_sub(1).saturating_sub(i))
            .copied()
            .unwrap_or(1);
        let dim2 = shape2
            .get(shape2.len().saturating_sub(1).saturating_sub(i))
            .copied()
            .unwrap_or(1);

        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result_shape.push(std::cmp::max(dim1, dim2));
        } else {
            return Err(TensorError::InvalidShape {
                reason: format!("Cannot broadcast shapes {:?} and {:?}", shape1, shape2),
            });
        }
    }

    // Since we iterated in reverse order, reverse the result shape
    result_shape.reverse();
    Ok(result_shape)
}
