// ADD, SUB, MUL, DIV, MAT_MUL

use crate::{
    error::TensorResult,
    gradient::Node,
    tape::TENSOR_TAPE,
    utils::validate::{assert_device_match, assert_mat_mul_shape_match, assert_shape_match},
    Tensor,
};
use maidenx_cpu::tensor_ops::tensor_basic::{
    cpu_tensor_add, cpu_tensor_div, cpu_tensor_mat_mul, cpu_tensor_mul, cpu_tensor_sub,
};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::tensor_basic::{
    cuda_tensor_add, cuda_tensor_div, cuda_tensor_mat_mul, cuda_tensor_mul, cuda_tensor_sub,
};
use maidenx_device::Device;
use std::sync::{Arc, Mutex};

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> TensorResult<Self> {
        assert_shape_match(self, rhs)?;
        assert_device_match(self, rhs)?;

        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_add(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_add(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            grad_inputs.push(grad_output.clone());
                        }
                        if rhs_requires_grad {
                            grad_inputs.push(grad_output.clone());
                        }
                        grad_inputs
                    }
                })),
                inputs: vec![
                    self.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    rhs.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
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

    pub fn sub(&self, rhs: &Tensor) -> TensorResult<Self> {
        assert_shape_match(self, rhs)?;
        assert_device_match(self, rhs)?;

        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_sub(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_sub(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            grad_inputs.push(grad_output.clone());
                        }
                        if rhs_requires_grad {
                            grad_inputs.push(grad_output.neg().unwrap_or(grad_output.clone()));
                        }
                        grad_inputs
                    }
                })),
                inputs: vec![
                    self.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    rhs.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
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

    pub fn mul(&self, rhs: &Tensor) -> TensorResult<Self> {
        assert_shape_match(self, rhs)?;
        assert_device_match(self, rhs)?;

        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_mul(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_mul(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs = self.clone();
                    let rhs = rhs.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            grad_inputs.push(grad_output.mul(&rhs).unwrap_or(grad_output.clone()));
                        }
                        if rhs_requires_grad {
                            grad_inputs.push(grad_output.mul(&lhs).unwrap_or(grad_output.clone()));
                        }
                        grad_inputs
                    }
                })),
                inputs: vec![
                    self.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    rhs.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
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

    pub fn div(&self, rhs: &Tensor) -> TensorResult<Self> {
        assert_shape_match(self, rhs)?;
        assert_device_match(self, rhs)?;

        let device = self.device;
        let mut output = Self::from_vec_with_device(vec![0.0; self.size()], &self.shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_div(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_div(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs = self.clone();
                    let rhs = rhs.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            grad_inputs.push(grad_output.div(&rhs).unwrap_or(grad_output.clone()));
                        }
                        if rhs_requires_grad {
                            let rhs_squared = rhs.mul(&rhs).unwrap_or(rhs.clone());
                            let grad = grad_output
                                .neg()
                                .unwrap_or_else(|_| grad_output.clone())
                                .mul(&lhs)
                                .unwrap_or_else(|_| grad_output.clone())
                                .div(&rhs_squared)
                                .unwrap_or_else(|_| grad_output.clone());
                            grad_inputs.push(grad);
                        }
                        grad_inputs
                    }
                })),
                inputs: vec![
                    self.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    rhs.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
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

    pub fn mat_mul(&self, rhs: &Tensor) -> TensorResult<Self> {
        assert_device_match(self, rhs)?;
        assert_mat_mul_shape_match(self, rhs)?;

        let (m, k) = (self.shape[0] as i32, self.shape[1] as i32);
        let (k2, n) = (rhs.shape[0] as i32, rhs.shape[1] as i32);
        assert_eq!(k, k2, "Incompatible matrix dimensions for multiplication");

        let device = self.device;
        let output_shape = vec![m as usize, n as usize];
        let mut output =
            Self::from_vec_with_device(vec![0.0; (m * n) as usize], &output_shape, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_mat_mul(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    m,
                    n,
                    k,
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_mat_mul(
                    output.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    m,
                    n,
                    k,
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs = self.clone();
                    let rhs = rhs.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();

                        if lhs_requires_grad {
                            let rhs_t = rhs.transpose().unwrap();
                            let grad_lhs = grad_output.mat_mul(&rhs_t).unwrap();
                            grad_inputs.push(grad_lhs.clone());
                        }

                        if rhs_requires_grad {
                            let lhs_t = lhs.transpose().unwrap();
                            let grad_rhs = lhs_t.mat_mul(grad_output).unwrap();
                            grad_inputs.push(grad_rhs.clone());
                        }

                        grad_inputs
                    }
                })),

                inputs: vec![
                    self.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                    rhs.node
                        .clone()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
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
    fn test_cpu_add() -> TensorResult<()> {
        let device = Device::cpu();

        // 1D tensor addition
        let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.add(&b)?;
        assert_eq!(c.to_vec()?, vec![5.0, 7.0, 9.0]);

        // 2D tensor addition
        let a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
        let c = a.add(&b)?;
        assert_eq!(c.to_vec()?, vec![6.0, 8.0, 10.0, 12.0]);

        Ok(())
    }

    #[test]
    fn test_add_shape_mismatch() -> TensorResult<()> {
        let device = Device::cpu();

        let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![1.0, 2.0], &device)?;
        assert!(a.add(&b).is_err());
        Ok(())
    }

    #[test]
    fn test_add_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.with_grad();
        b.with_grad();

        let c = a.add(&b)?;
        c.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        assert_eq!(b_grad, vec![1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_add_chain_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

        a.with_grad();
        b.with_grad();
        c.with_grad();

        let d = a.add(&b)?.add(&c)?;
        d.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;
        let c_grad = c.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        assert_eq!(b_grad, vec![1.0, 1.0, 1.0]);
        assert_eq!(c_grad, vec![1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_cpu_sub() -> TensorResult<()> {
        let device = Device::cpu();

        let a = Tensor::from_device(vec![5.0, 7.0, 9.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.sub(&b)?;
        assert_eq!(c.to_vec()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_sub_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.with_grad();
        b.with_grad();

        let c = a.sub(&b)?;
        c.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        assert_eq!(b_grad, vec![-1.0, -1.0, -1.0]);

        Ok(())
    }

    #[test]
    fn test_sub_chain_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

        a.with_grad();
        b.with_grad();
        c.with_grad();

        let d = a.sub(&b)?.sub(&c)?;
        d.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;
        let c_grad = c.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
        assert_eq!(b_grad, vec![-1.0, -1.0, -1.0]);
        assert_eq!(c_grad, vec![-1.0, -1.0, -1.0]);

        Ok(())
    }

    #[test]
    fn test_cpu_mul() -> TensorResult<()> {
        let device = Device::cpu();

        let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.mul(&b)?;
        assert_eq!(c.to_vec()?, vec![4.0, 10.0, 18.0]);
        Ok(())
    }

    #[test]
    fn test_mul_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.with_grad();
        b.with_grad();

        let c = a.mul(&b)?;
        c.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![4.0, 5.0, 6.0]);
        assert_eq!(b_grad, vec![1.0, 2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_mul_chain_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

        a.with_grad();
        b.with_grad();
        c.with_grad();

        let d = a.mul(&b)?.mul(&c)?;
        d.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;
        let c_grad = c.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![28.0, 40.0, 54.0]);
        assert_eq!(b_grad, vec![7.0, 16.0, 27.0]);
        assert_eq!(c_grad, vec![4.0, 10.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_cpu_div() -> TensorResult<()> {
        let device = Device::cpu();

        let a = Tensor::from_device(vec![4.0, 10.0, 18.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.div(&b)?;
        let tolerance = 1e-6;
        assert!(c
            .to_vec()?
            .iter()
            .zip([1.0, 2.0, 3.0].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));
        Ok(())
    }

    #[test]
    fn test_div_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.with_grad();
        b.with_grad();

        let c = a.div(&b)?;
        c.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;

        let tolerance = 1e-6;

        // Check gradients with tolerance
        assert!(a_grad
            .iter()
            .zip([0.25, 0.2, 0.16666667].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));
        assert!(b_grad
            .iter()
            .zip([-0.0625, -0.08, -0.08333334].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));

        Ok(())
    }

    #[test]
    fn test_div_chain_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

        a.with_grad();
        b.with_grad();
        c.with_grad();

        let d = a.div(&b)?.div(&c)?;
        d.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;
        let c_grad = c.grad()?.unwrap().to_vec()?;

        // ! very big tolerance
        let tolerance = 1e-2;

        // Check gradients with tolerance
        assert!(a_grad
            .iter()
            .zip([0.03571429, 0.025, 0.01851852].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));
        assert!(b_grad
            .iter()
            .zip([-0.00892857, -0.01, -0.01028807].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));
        assert!(c_grad
            .iter()
            .zip([-0.00510204, -0.005, -0.00462963].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));

        Ok(())
    }

    #[test]
    fn test_cpu_mat_mul() -> TensorResult<()> {
        let device = Device::cpu();

        let a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
        let c = a.mat_mul(&b)?;

        // Result should be:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(c.to_vec()?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    fn test_mat_mul_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let mut b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

        a.with_grad();
        b.with_grad();

        let c = a.mat_mul(&b)?;
        c.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![11.0, 15.0, 11.0, 15.0]);
        assert_eq!(b_grad, vec![4.0, 4.0, 6.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_mat_mul_chain_backward() -> TensorResult<()> {
        let device = Device::Cpu;

        let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let mut b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
        let mut c = Tensor::from_device(vec![vec![1.0, 0.0], vec![0.0, 1.0]], &device)?;

        a.with_grad();
        b.with_grad();
        c.with_grad();

        let d = a.mat_mul(&b)?.mat_mul(&c)?;
        d.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;
        let c_grad = c.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![11.0, 15.0, 11.0, 15.0]);
        assert_eq!(b_grad, vec![4.0, 4.0, 6.0, 6.0]);
        assert_eq!(c_grad, vec![62.0, 62.0, 72.0, 72.0]);

        Ok(())
    }

    #[test]
    fn test_mix_chain_backward() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;
        let mut d = Tensor::from_device(vec![10.0, 11.0, 12.0], &device)?;

        a.with_grad();
        b.with_grad();
        c.with_grad();
        d.with_grad();

        // ((a + b) * c - d) / 2
        let result = a
            .add(&b)? // (a + b)
            .mul(&c)? // * c
            .sub(&d)? // - d
            .scalar_div(2.0)?; // / 2

        result.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;
        let c_grad = c.grad()?.unwrap().to_vec()?;
        let d_grad = d.grad()?.unwrap().to_vec()?;

        // result = (((a + b) * c) - d) / 2
        // ∂L/∂a = (1 * c) / 2
        // ∂L/∂b = (1 * c) / 2
        // ∂L/∂c = ((a + b) * 1) / 2
        // ∂L/∂d = -1 / 2

        assert_eq!(a_grad, vec![3.5, 4.0, 4.5]); // (c / 2)
        assert_eq!(b_grad, vec![3.5, 4.0, 4.5]); // (c / 2)
        assert_eq!(c_grad, vec![2.5, 3.5, 4.5]); // ((a + b) / 2)
        assert_eq!(d_grad, vec![-0.5, -0.5, -0.5]); // -1 / 2

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_add() -> TensorResult<()> {
        let device = Device::cuda(0);

        let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.add(&b)?;
        assert_eq!(c.to_vec()?, vec![5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_sub() -> TensorResult<()> {
        let device = Device::cuda(0);

        let a = Tensor::from_device(vec![5.0, 7.0, 9.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.sub(&b)?;
        assert_eq!(c.to_vec()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_mul() -> TensorResult<()> {
        let device = Device::cuda(0);

        let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.mul(&b)?;
        assert_eq!(c.to_vec()?, vec![4.0, 10.0, 18.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_div() -> TensorResult<()> {
        let device = Device::cuda(0);

        let a = Tensor::from_device(vec![4.0, 10.0, 18.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
        let c = a.div(&b)?;
        let tolerance = 1e-6;
        assert!(c
            .to_vec()?
            .iter()
            .zip([1.0, 2.0, 3.0].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_mat_mul() -> TensorResult<()> {
        let device = Device::cuda(0);

        let a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
        let c = a.mat_mul(&b)?;

        // Result should be:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(c.to_vec()?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_mat_mul_backward() -> TensorResult<()> {
        let device = Device::cuda(0);

        let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let mut b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

        a.with_grad();
        b.with_grad();

        let c = a.mat_mul(&b)?;
        c.backward()?;

        let a_grad = a.grad()?.unwrap().to_vec()?;
        let b_grad = b.grad()?.unwrap().to_vec()?;

        assert_eq!(a_grad, vec![11.0, 15.0, 11.0, 15.0]);
        assert_eq!(b_grad, vec![4.0, 4.0, 6.0, 6.0]);

        Ok(())
    }
}
