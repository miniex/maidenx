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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.mean()?;
        assert_eq!(result.to_vec()?, vec![2.5]);
        Ok(())
    }

    #[test]
    fn test_mean_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        let b = a.mean()?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![0.25, 0.25, 0.25, 0.25]); // grad is 1/N for each element
        Ok(())
    }

    #[test]
    fn test_sum() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.sum()?;
        assert_eq!(result.to_vec()?, vec![10.0]);
        Ok(())
    }

    #[test]
    fn test_sum_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        let b = a.sum()?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![1.0, 1.0, 1.0, 1.0]); // grad is 1 for each element
        Ok(())
    }

    #[test]
    fn test_sum_with_dim() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        // Reduction along dimension 1 (columns)
        let result = tensor.sum_with_dim(1)?;
        assert_eq!(result.to_vec()?, vec![6.0, 15.0]);

        Ok(())
    }

    #[test]
    fn test_sum_with_dim_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        tensor.with_grad();

        let result = tensor.sum_with_dim(1)?;
        result.backward()?;

        let grad = tensor.grad()?.unwrap().to_vec()?;
        assert_eq!(grad, vec![1.0; 6]); // Grad is 1 for each element since it's a sum

        Ok(())
    }

    #[test]
    fn test_sum_to_shape() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        // Test reduction to [1, 3]
        let reduced = tensor.sum_to_shape(&[1, 3])?;
        assert_eq!(reduced.shape(), &[1, 3]);
        assert_eq!(reduced.to_vec()?, vec![5.0, 7.0, 9.0]);

        // Test reduction to [2, 1]
        let reduced = tensor.sum_to_shape(&[2, 1])?;
        assert_eq!(reduced.shape(), &[2, 1]);
        assert_eq!(reduced.to_vec()?, vec![6.0, 15.0]);

        Ok(())
    }

    #[test]
    fn test_sum_to_shape_with_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        tensor.with_grad();

        // Test reduction to [1, 3]
        let reduced = tensor.sum_to_shape(&[1, 3])?;
        reduced.backward()?;

        // Gradient should be evenly distributed back to original tensor
        let grad = tensor.grad()?.unwrap().to_vec()?;
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_mean_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.mean()?;
        assert_eq!(result.to_vec()?, vec![2.5]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sum_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
        let result = tensor.sum()?;
        assert_eq!(result.to_vec()?, vec![10.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sum_with_dim_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        // Reduction along dimension 0 (columns)
        let result = tensor.sum_with_dim(0)?;

        println!("Input shape: {:?}", tensor.shape());
        println!("Output shape: {:?}", result.shape());

        assert_eq!(result.to_vec()?, vec![5.0, 7.0, 9.0]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sum_to_shape_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

        // Test reduction to [1, 3]
        let reduced = tensor.sum_to_shape(&[1, 3])?;
        assert_eq!(reduced.shape(), &[1, 3]);
        assert_eq!(reduced.to_vec()?, vec![5.0, 7.0, 9.0]);

        // Test reduction to [2, 1]
        let reduced = tensor.sum_to_shape(&[2, 1])?;
        assert_eq!(reduced.shape(), &[2, 1]);
        assert_eq!(reduced.to_vec()?, vec![6.0, 15.0]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sum_to_shape_cuda_with_backward() -> TensorResult<()> {
        let device = Device::cuda(0);
        let mut tensor =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
        tensor.with_grad();

        // Test reduction to [1, 3]
        let reduced = tensor.sum_to_shape(&[1, 3])?;
        reduced.backward()?;

        // Gradient should be evenly distributed back to original tensor
        let grad = tensor.grad()?.unwrap().to_vec()?;
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        Ok(())
    }
}
