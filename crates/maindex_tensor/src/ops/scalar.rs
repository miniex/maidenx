// SCALAR ADD, SCALAR SUB, SCALAR MUL, SCALAR DIV, NEG

use std::sync::{Arc, Mutex};

use crate::{
    error::{TensorError, TensorResult},
    gradient::Node,
    tape::TENSOR_TAPE,
    Tensor,
};
use maidenx_cpu::ops::tensor_scalar::{
    cpu_tensor_scalar_add, cpu_tensor_scalar_div, cpu_tensor_scalar_mul, cpu_tensor_scalar_sub,
};
#[cfg(feature = "cuda")]
use maidenx_cuda::ops::tensor_scalar::{
    cuda_tensor_scalar_add, cuda_tensor_scalar_div, cuda_tensor_scalar_mul, cuda_tensor_scalar_sub,
};
use maidenx_device::Device;

impl Tensor {
    pub fn scalar_add(&self, scalar: f32) -> TensorResult<Self> {
        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_scalar_add(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_scalar_add(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    move |grad_output: &Tensor| -> Vec<Tensor> { vec![grad_output.clone()] }
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

    pub fn scalar_sub(&self, scalar: f32) -> TensorResult<Self> {
        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_scalar_sub(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_scalar_sub(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    move |grad_output: &Tensor| -> Vec<Tensor> { vec![grad_output.clone()] }
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

    pub fn scalar_mul(&self, scalar: f32) -> TensorResult<Self> {
        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_scalar_mul(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_scalar_mul(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        vec![grad_output.scalar_mul(scalar).unwrap()]
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

    pub fn scalar_div(&self, scalar: f32) -> TensorResult<Self> {
        if scalar == 0.0 {
            return Err(TensorError::OperationError {
                reason: "Cannot divide tensor by zero scalar value".to_string(),
            });
        }

        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_scalar_div(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_scalar_div(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    scalar,
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let scalar_inv = 1.0 / scalar;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        vec![grad_output.scalar_mul(scalar_inv).unwrap()]
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

    pub fn neg(&self) -> TensorResult<Self> {
        self.scalar_mul(-1.0f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_add() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let result = tensor.scalar_add(5.0)?;
        assert_eq!(result.to_vec()?, vec![6.0, 7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_add_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        a.with_grad();

        let b = a.scalar_add(5.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]); // dL/da = 1
        Ok(())
    }

    #[test]
    fn test_scalar_add_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        a.with_grad();

        let b = a.scalar_add(5.0)?.scalar_add(3.0)?.scalar_add(2.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_sub() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
        let result = tensor.scalar_sub(5.0)?;
        assert_eq!(result.to_vec()?, vec![5.0, 15.0, 25.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_sub_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        a.with_grad();

        let b = a.scalar_sub(5.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]); // dL/da = 1
        Ok(())
    }

    #[test]
    fn test_scalar_sub_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
        a.with_grad();

        let b = a.scalar_sub(5.0)?.scalar_sub(3.0)?.scalar_sub(2.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_mul() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![2.0, 4.0, 6.0], &device)?;
        let result = tensor.scalar_mul(3.0)?;
        assert_eq!(result.to_vec()?, vec![6.0, 12.0, 18.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_mul_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        a.with_grad();

        let b = a.scalar_mul(5.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![5.0, 5.0, 5.0]); // dL/da = scalar
        Ok(())
    }

    #[test]
    fn test_scalar_mul_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        let b = a.scalar_mul(2.0)?.scalar_mul(3.0)?.scalar_mul(4.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![24.0, 24.0, 24.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_div() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
        let result = tensor.scalar_div(10.0)?;
        assert_eq!(result.to_vec()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_div_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
        a.with_grad();

        let b = a.scalar_div(5.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![0.2, 0.2, 0.2]); // dL/da = 1/scalar
        Ok(())
    }

    #[test]
    fn test_scalar_div_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![8.0, 16.0, 32.0], &device)?;
        a.with_grad();

        let b = a.scalar_div(2.0)?.scalar_div(2.0)?.scalar_div(2.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        assert_eq!(a_grad, vec![0.125, 0.125, 0.125]);
        Ok(())
    }

    #[test]
    fn test_neg() -> TensorResult<()> {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![1.0, -2.0, 3.0], &device)?;
        let result = tensor.neg()?;
        assert_eq!(result.to_vec()?, vec![-1.0, 2.0, -3.0]);
        Ok(())
    }

    #[test]
    fn test_scalar_mix_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();
        let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
        a.with_grad();

        // (((a + 2) * 3) - 4) / 2
        let b = a
            .scalar_add(2.0)?
            .scalar_mul(3.0)?
            .scalar_sub(4.0)?
            .scalar_div(2.0)?;
        b.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        // dL/da = ((1 * 3) * 1) / 2 = 1.5
        assert_eq!(a_grad, vec![1.5, 1.5, 1.5]);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Cannot divide tensor by zero scalar value")]
    fn test_scalar_div_zero() {
        let device = Device::cpu();
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device).unwrap();
        tensor.scalar_div(0.0).unwrap(); // This should panic
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_scalar_add_cuda() -> TensorResult<()> {
        let device = Device::cuda(0);
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let result = tensor.scalar_add(5.0)?;
        assert_eq!(result.to_vec()?, vec![6.0, 7.0, 8.0]);
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
}
