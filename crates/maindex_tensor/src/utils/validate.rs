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
