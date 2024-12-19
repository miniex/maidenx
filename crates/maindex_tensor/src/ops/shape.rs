// TRANSPOSE, TRANSPOSE DIM, RESHAPE

use crate::{
    error::{TensorError, TensorResult},
    gradient::Node,
    tape::TENSOR_TAPE,
    Tensor,
};
use maidenx_cpu::tensor_ops::shape::{cpu_tensor_transpose_2d, cpu_tensor_transpose_dim};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::shape::{cuda_tensor_transpose_2d, cuda_tensor_transpose_dim};
use maidenx_device::Device;
use std::sync::{Arc, Mutex};

impl Tensor {
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

    pub fn reshape(&self, new_shape: &[usize]) -> TensorResult<Self> {
        let total_size = self.shape.iter().product::<usize>();
        let new_total_size = new_shape.iter().product::<usize>();

        if total_size != new_total_size {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Cannot reshape tensor with size {} to shape {:?} (size {}).",
                    total_size, new_shape, new_total_size
                ),
            });
        }

        let new_strides = Self::compute_strides(new_shape);

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let old_shape = self.shape.clone();
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        vec![grad_output
                            .reshape(&old_shape)
                            .expect("Reshape back to original shape failed")]
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

        let output = Self {
            buffer: self.buffer.clone(),
            shape: new_shape.to_vec(),
            strides: new_strides,
            device: self.device,
            requires_grad,
            node,
        };

        if requires_grad {
            TENSOR_TAPE.with(|tape| tape.borrow_mut().add(output.clone()));
        }

        Ok(output)
    }
}
