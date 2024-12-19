// SCALAR ADD, SCALAR SUB, SCALAR MUL, SCALAR DIV

use crate::{
    error::{TensorError, TensorResult},
    gradient::Node,
    tape::TENSOR_TAPE,
    Tensor,
};
use maidenx_cpu::tensor_ops::scalar::{
    cpu_tensor_pow, cpu_tensor_scalar_add, cpu_tensor_scalar_div, cpu_tensor_scalar_mul,
    cpu_tensor_scalar_sub,
};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::scalar::{
    cuda_tensor_pow, cuda_tensor_scalar_add, cuda_tensor_scalar_div, cuda_tensor_scalar_mul,
    cuda_tensor_scalar_sub,
};
use maidenx_device::Device;
use std::sync::{Arc, Mutex};

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
