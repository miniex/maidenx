// MEAN, SUM, SUM WITH DIM, SUM TO SHAPE

use crate::{
    error::{TensorError, TensorResult},
    gradient::Node,
    tape::TENSOR_TAPE,
    Tensor,
};
use maidenx_cpu::tensor_ops::reduce::{cpu_tensor_mean, cpu_tensor_sum, cpu_tensor_sum_with_dim};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::reduce::{
    cuda_tensor_mean, cuda_tensor_sum, cuda_tensor_sum_with_dim,
};
use maidenx_device::Device;
use std::sync::{Arc, Mutex};

impl Tensor {
    pub fn mean(&self) -> TensorResult<Self> {
        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; 1], &[1], &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_mean(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_mean(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let size = self.size();
                    let shape = self.shape.clone();
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let scale = 1.0 / (size as f32);
                        let grad = Tensor::from_vec_with_device(
                            vec![grad_output.to_vec().unwrap()[0] * scale; size],
                            &shape,
                            &grad_output.device,
                        )
                        .unwrap();
                        vec![grad]
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

    pub fn sum(&self) -> TensorResult<Self> {
        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; 1], &[1], &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_sum(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_sum(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let size = self.size();
                    let shape = self.shape.clone();
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let grad = Tensor::from_vec_with_device(
                            vec![grad_output.to_vec().unwrap()[0]; size],
                            &shape,
                            &grad_output.device,
                        )
                        .unwrap();
                        vec![grad]
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

    pub fn sum_with_dim(&self, reduction_dim: usize) -> TensorResult<Self> {
        let device = self.device;
        let input_shape = self.shape.clone();
        let output_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != reduction_dim)
            .map(|(_, &dim)| dim)
            .collect();

        let output_size = output_shape.iter().product();
        let mut output =
            Self::from_vec_with_device(vec![0.0; output_size], &output_shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_sum_with_dim(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    &input_shape.iter().map(|&d| d as i32).collect::<Vec<_>>(),
                    reduction_dim as i32,
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_sum_with_dim(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    &input_shape.iter().map(|&d| d as i32).collect::<Vec<_>>(),
                    reduction_dim as i32,
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let shape = self.shape.clone();
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut expanded_shape = grad_output.shape.clone();
                        expanded_shape.insert(reduction_dim, shape[reduction_dim]);
                        let grad = Tensor::from_vec_with_device(
                            grad_output.to_vec().unwrap().repeat(shape[reduction_dim]),
                            &expanded_shape,
                            &grad_output.device,
                        )
                        .unwrap();
                        vec![grad]
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

    // Gradient is calculated by sum_with_dim
    pub fn sum_to_shape(&self, target_shape: &[usize]) -> TensorResult<Self> {
        let mut current_tensor = self.clone();

        // Ensure dimensions match by padding with 1s if needed
        let mut padded_shape = current_tensor.shape.clone();
        while padded_shape.len() < target_shape.len() {
            padded_shape.insert(0, 1);
            current_tensor.strides.insert(0, 0);
        }
        current_tensor.shape = padded_shape;

        // Process each dimension
        for dim in (0..current_tensor.shape.len()).rev() {
            let target_dim = target_shape.get(dim).copied().unwrap_or(1);
            if current_tensor.shape[dim] != target_dim {
                if current_tensor.shape[dim] > 1 && target_dim == 1 {
                    // Reduce this dimension
                    current_tensor = current_tensor.sum_with_dim(dim)?;
                } else if current_tensor.shape[dim] != 1 {
                    return Err(TensorError::InvalidShape {
                        reason: format!(
                            "Cannot sum shape {:?} to target shape {:?} at dimension {}",
                            current_tensor.shape, target_shape, dim
                        ),
                    });
                }
            }
        }

        // Ensure final shape matches target shape
        current_tensor.shape = target_shape.to_vec();

        Ok(current_tensor)
    }
}
