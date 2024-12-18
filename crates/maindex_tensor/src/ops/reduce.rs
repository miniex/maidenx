use crate::{error::TensorResult, gradient::Node, tape::TENSOR_TAPE, Tensor};
use maidenx_cpu::tensor_ops::reduce::{cpu_tensor_mean, cpu_tensor_sum};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::reduce::{cuda_tensor_mean, cuda_tensor_sum};
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
}
