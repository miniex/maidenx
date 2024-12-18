// NEG, POW

use crate::{error::TensorResult, gradient::Node, tape::TENSOR_TAPE, Tensor};
use maidenx_cpu::tensor_ops::scalar::cpu_tensor_pow;
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::scalar::cuda_tensor_pow;
use maidenx_device::Device;
use std::sync::{Arc, Mutex};

impl Tensor {
    // Gradient is calculated by scalar_mul
    pub fn neg(&self) -> TensorResult<Self> {
        self.scalar_mul(-1.0f32)
    }

    pub fn pow(&self, exponent: f32) -> TensorResult<Self> {
        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_pow(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    exponent,
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_pow(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    exponent,
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let self_clone = self.clone();
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let grad = self_clone
                            .pow(exponent - 1.0)
                            .unwrap()
                            .scalar_mul(exponent)
                            .unwrap()
                            .mul(grad_output)
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
    fn test_neg() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![1.0, -2.0, 3.0], &device)?;
        let result = tensor.neg()?;
        assert_eq!(result.to_vec()?, vec![-1.0, 2.0, -3.0]);
        Ok(())
    }

    #[test]
    fn test_pow() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        let result = tensor.pow(2.0)?;
        assert_eq!(result.to_vec()?, vec![4.0, 9.0, 16.0]);
        Ok(())
    }

    #[test]
    fn test_pow_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        let b = a.pow(2.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        // gradient of x^2 is 2x
        assert_eq!(a_grad, vec![4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_pow_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        let b = a.pow(2.0)?.pow(2.0)?; // (x^2)^2 = x^4
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        // gradient of x^4 is 4x^3
        assert_eq!(a_grad, vec![32.0, 108.0, 256.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_neg_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor = Tensor::from_device(vec![1.0, -2.0, 3.0], &device)?;
        let result = tensor.neg()?;
        assert_eq!(result.to_vec()?, vec![-1.0, 2.0, -3.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pow_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        let result = tensor.pow(2.0)?;
        assert_eq!(result.to_vec()?, vec![4.0, 9.0, 16.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pow_cuda_backward() -> TensorResult<()> {
        let device = Device::cuda(0);
        let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        let b = a.pow(2.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        // gradient of x^2 is 2x
        assert_eq!(a_grad, vec![4.0, 6.0, 8.0]);
        Ok(())
    }
}
