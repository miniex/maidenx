// TRANSPOSE, TRANSPOSE DIM

use std::sync::{Arc, Mutex};

use crate::{
    error::{TensorError, TensorResult},
    gradient::Node,
    tape::TENSOR_TAPE,
    Tensor,
};
use maidenx_cpu::ops::tensor_shape::{cpu_tensor_transpose_2d, cpu_tensor_transpose_dim};
#[cfg(feature = "cuda")]
use maidenx_cuda::ops::tensor_shape::{cuda_tensor_transpose_2d, cuda_tensor_transpose_dim};
use maidenx_device::Device;

impl Tensor {
    /// Transposes a 2D tensor.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The tensor is not 2D (shape.len() != 2)
    pub fn transpose(&self) -> TensorResult<Self> {
        if self.shape.len() != 2 {
            return Err(TensorError::OperationError {
                reason: format!(
                    "transpose requires a 2D tensor, but got shape {:?}",
                    self.shape
                ),
            });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let device = self.device;

        // Create output tensor with swapped dimensions
        let mut output =
            Self::from_vec_with_device(vec![0.0; self.size()], &[cols, rows], &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_transpose_2d(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rows,
                    cols,
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_transpose_2d(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rows,
                    cols,
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let grad_transposed = grad_output.transpose().unwrap();
                        vec![grad_transposed]
                    }
                })),
                inputs: vec![self
                    .node
                    .clone()
                    .map(|n| Arc::downgrade(&n))
                    .unwrap_or_default()],
                grad: None,
            })))
        } else {
            None
        };

        output.requires_grad = requires_grad;
        output.node = node;

        if requires_grad {
            TENSOR_TAPE.with(|tape| tape.borrow_mut().add(output.clone()));
        }

        Ok(output)
    }

    /// Transposes a tensor along the specified dimensions.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The dimensions are out of bounds
    /// - The dimensions are negative
    pub fn transpose_dim(&self, dim0: i32, dim1: i32) -> TensorResult<Self> {
        let num_dims = self.shape.len();

        // Validate dimensions
        if dim0 < 0 || dim0 >= num_dims as i32 || dim1 < 0 || dim1 >= num_dims as i32 {
            return Err(TensorError::OperationError {
                reason: format!(
                    "Invalid dimensions {} and {} for tensor with {} dimensions",
                    dim0, dim1, num_dims
                ),
            });
        }

        // If dimensions are the same, return a clone
        if dim0 == dim1 {
            return Ok(self.clone());
        }

        let device = self.device;

        // Create output shape with swapped dimensions
        let mut output_shape = self.shape.clone();
        output_shape.swap(dim0 as usize, dim1 as usize);

        let mut output =
            Self::from_vec_with_device(vec![0.0; self.size()], &output_shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_transpose_dim(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    &self.shape,
                    dim0,
                    dim1,
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_transpose_dim(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    &self.shape,
                    dim0,
                    dim1,
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let grad_transposed = grad_output.transpose_dim(dim1, dim0).unwrap();
                        vec![grad_transposed]
                    }
                })),
                inputs: vec![self
                    .node
                    .clone()
                    .map(|n| Arc::downgrade(&n))
                    .unwrap_or_default()],
                grad: None,
            })))
        } else {
            None
        };

        output.requires_grad = requires_grad;
        output.node = node;

        if requires_grad {
            TENSOR_TAPE.with(|tape| tape.borrow_mut().add(output.clone()));
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        let result = tensor.transpose()?;

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_transpose_square() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let result = tensor.transpose()?;

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_transpose_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        tensor.with_grad();

        let transposed = tensor.transpose()?;
        transposed.backward()?;

        let grad = tensor.grad()?.unwrap().to_vec()?;
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]); // Each value contributes 1.0
        Ok(())
    }

    #[test]
    fn test_transpose_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        tensor.with_grad();

        let transposed = tensor.transpose()?.transpose()?; // Double transpose
        transposed.backward()?;

        let grad = tensor.grad()?.unwrap().to_vec()?;
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]); // Same as original
        Ok(())
    }

    #[test]
    fn test_transpose_dim() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_vec_with_device(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 3, 2],
            &device,
        )?;
        let result = tensor.transpose_dim(1, 2)?;

        assert_eq!(result.shape(), &[2, 2, 3]);
        assert_eq!(
            result.to_vec()?,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 8.0, 10.0, 12.0]
        );
        Ok(())
    }

    #[test]
    fn test_transpose_dim_identity() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        let result = tensor.transpose_dim(1, 1)?;

        assert_eq!(result.shape(), tensor.shape());
        assert_eq!(result.to_vec()?, tensor.to_vec()?);
        Ok(())
    }

    #[test]
    fn test_transpose_dim_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        tensor.with_grad();

        let transposed = tensor.transpose_dim(0, 1)?;
        transposed.backward()?;

        let grad = tensor.grad()?.unwrap().to_vec()?;
        assert_eq!(grad, vec![1.0; 6]); // Same contribution for all elements
        Ok(())
    }

    #[test]
    fn test_transpose_dim_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();

        let mut tensor = Tensor::from_vec_with_device(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 3, 2],
            &device,
        )?;
        tensor.with_grad();

        let transposed = tensor
            .transpose_dim(0, 1)? // 2 x 3 x 2 -> 3 x 2 x 2
            .transpose_dim(1, 2)? // 3 x 2 x 2 -> 3 x 2 x 2 (1, 2 swap)
            .transpose_dim(0, 2)?; // 3 x 2 x 2 -> 2 x 2 x 3

        transposed.backward()?;

        let grad = tensor.grad()?.unwrap().to_vec()?;

        println!("Original Tensor Grad: {:?}", grad);

        assert_eq!(grad, vec![1.0; 12]);
        Ok(())
    }

    #[test]
    fn test_transpose_invalid_dims() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(tensor.transpose_dim(-1, 1).is_err());
        assert!(tensor.transpose_dim(1, 2).is_err());
        assert!(tensor.transpose_dim(2, 1).is_err());
    }

    #[test]
    fn test_transpose_non_2d() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(tensor.transpose().is_err());

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 1, 2]).unwrap();
        assert!(tensor.transpose().is_err());
    }

    #[test]
    fn test_transpose_chain() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
        let transposed1 = tensor.transpose()?;
        let transposed2 = transposed1.transpose()?;

        // Double transpose should return the original tensor
        assert_eq!(transposed2.shape(), tensor.shape());
        assert_eq!(transposed2.to_vec()?, tensor.to_vec()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_transpose_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        let result = tensor.transpose()?;

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_transpose_dim_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor = Tensor::from_vec_with_device(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            &device,
        )?;
        let result = tensor.transpose_dim(0, 2)?;

        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(
            result.to_vec()?,
            vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]
        );
        Ok(())
    }
}
