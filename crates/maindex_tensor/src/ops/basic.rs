// ADD, SUB, MUL, DIV, MAT_MUL

use crate::{
    error::{TensorError, TensorResult},
    gradient::Node,
    tape::TENSOR_TAPE,
    utils::validate::{assert_device_match, assert_mat_mul_shape_match},
    Tensor,
};
use maidenx_cpu::tensor_ops::basic::{
    cpu_tensor_add, cpu_tensor_div, cpu_tensor_mat_mul, cpu_tensor_mul, cpu_tensor_sub,
};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::basic::{
    cuda_tensor_add, cuda_tensor_div, cuda_tensor_mat_mul, cuda_tensor_mul, cuda_tensor_sub,
};
use maidenx_device::Device;
use std::sync::{Arc, Mutex};

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> TensorResult<Self> {
        // Calculate output shape by comparing dimensions
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        // Pad shorter shape with ones on the left
        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

        // For each dimension, take the maximum of the two shapes
        for i in 0..max_rank {
            let dim1 = padded_lhs[i];
            let dim2 = padded_rhs[i];
            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                return Err(TensorError::InvalidShape {
                    reason: format!(
                        "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                        self.shape, rhs.shape, i
                    ),
                });
            }
            broadcasted_shape.push(dim1.max(dim2));
        }

        // Broadcast tensors to the calculated shape
        let broadcasted_lhs = self.broadcast_like(&Tensor {
            buffer: self.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: self.device,
            requires_grad: false,
            node: None,
        })?;
        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;
        let device = self.device;

        // Use the broadcasted_shape for output
        let mut output = Self::from_vec_with_device(
            vec![0.0; broadcasted_shape.iter().product()],
            &broadcasted_shape, // Here we use the calculated broadcasted_shape
            &device,
        )?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_add(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_add(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs_shape = self.shape.clone();
                    let rhs_shape = rhs.shape.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            let grad = grad_output
                                .sum_to_shape(&lhs_shape)
                                .unwrap_or(grad_output.clone());
                            grad_inputs.push(grad);
                        }
                        if rhs_requires_grad {
                            let grad = grad_output
                                .sum_to_shape(&rhs_shape)
                                .unwrap_or(grad_output.clone());
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

    pub fn sub(&self, rhs: &Tensor) -> TensorResult<Self> {
        // Calculate output shape by comparing dimensions
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        // Pad shorter shape with ones on the left
        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

        // For each dimension, take the maximum of the two shapes
        for i in 0..max_rank {
            let dim1 = padded_lhs[i];
            let dim2 = padded_rhs[i];
            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                return Err(TensorError::InvalidShape {
                    reason: format!(
                        "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                        self.shape, rhs.shape, i
                    ),
                });
            }
            broadcasted_shape.push(dim1.max(dim2));
        }

        // Broadcast tensors to the calculated shape
        let broadcasted_lhs = self.broadcast_like(&Tensor {
            buffer: self.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: self.device,
            requires_grad: false,
            node: None,
        })?;
        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;
        let device = self.device;

        // Use the broadcasted_shape for output
        let mut output = Self::from_vec_with_device(
            vec![0.0; broadcasted_shape.iter().product()],
            &broadcasted_shape, // Here we use the calculated broadcasted_shape
            &device,
        )?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_sub(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_sub(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs_shape = self.shape.clone();
                    let rhs_shape = rhs.shape.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            let grad = grad_output
                                .sum_to_shape(&lhs_shape)
                                .unwrap_or(grad_output.clone());
                            grad_inputs.push(grad);
                        }
                        if rhs_requires_grad {
                            let grad = grad_output
                                .neg()
                                .unwrap_or(grad_output.clone())
                                .sum_to_shape(&rhs_shape)
                                .unwrap_or(grad_output.clone());
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

    pub fn mul(&self, rhs: &Tensor) -> TensorResult<Self> {
        // Calculate output shape by comparing dimensions
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        // Pad shorter shape with ones on the left
        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

        // For each dimension, take the maximum of the two shapes
        for i in 0..max_rank {
            let dim1 = padded_lhs[i];
            let dim2 = padded_rhs[i];
            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                return Err(TensorError::InvalidShape {
                    reason: format!(
                        "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                        self.shape, rhs.shape, i
                    ),
                });
            }
            broadcasted_shape.push(dim1.max(dim2));
        }

        // Broadcast tensors to the calculated shape
        let broadcasted_lhs = self.broadcast_like(&Tensor {
            buffer: self.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: self.device,
            requires_grad: false,
            node: None,
        })?;
        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;
        let device = self.device;

        // Use the broadcasted_shape for output
        let mut output = Self::from_vec_with_device(
            vec![0.0; broadcasted_shape.iter().product()],
            &broadcasted_shape, // Here we use the calculated broadcasted_shape
            &device,
        )?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_mul(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_mul(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs = broadcasted_lhs.clone();
                    let rhs = broadcasted_rhs.clone();
                    let lhs_shape = self.shape.clone();
                    let rhs_shape = rhs.shape.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            let grad = grad_output
                                .mul(&rhs)
                                .unwrap_or(grad_output.clone())
                                .sum_to_shape(&lhs_shape)
                                .unwrap_or(grad_output.clone());
                            grad_inputs.push(grad);
                        }
                        if rhs_requires_grad {
                            let grad = grad_output
                                .mul(&lhs)
                                .unwrap_or(grad_output.clone())
                                .sum_to_shape(&rhs_shape)
                                .unwrap_or(grad_output.clone());
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

    pub fn div(&self, rhs: &Tensor) -> TensorResult<Self> {
        // Calculate output shape by comparing dimensions
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        // Pad shorter shape with ones on the left
        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

        // For each dimension, take the maximum of the two shapes
        for i in 0..max_rank {
            let dim1 = padded_lhs[i];
            let dim2 = padded_rhs[i];
            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                return Err(TensorError::InvalidShape {
                    reason: format!(
                        "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                        self.shape, rhs.shape, i
                    ),
                });
            }
            broadcasted_shape.push(dim1.max(dim2));
        }

        // Broadcast tensors to the calculated shape
        let broadcasted_lhs = self.broadcast_like(&Tensor {
            buffer: self.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: self.device,
            requires_grad: false,
            node: None,
        })?;
        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;
        let device = self.device;

        // Use the broadcasted_shape for output
        let mut output = Self::from_vec_with_device(
            vec![0.0; broadcasted_shape.iter().product()],
            &broadcasted_shape, // Here we use the calculated broadcasted_shape
            &device,
        )?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_div(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_div(
                    output.buffer.as_mut_ptr(),
                    broadcasted_lhs.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    output.size(),
                    None,
                )?;
            },
        }

        let requires_grad = self.requires_grad || rhs.requires_grad;
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let lhs = broadcasted_lhs.clone();
                    let rhs = broadcasted_rhs.clone();
                    let lhs_shape = self.shape.clone();
                    let rhs_shape = rhs.shape.clone();
                    let lhs_requires_grad = self.requires_grad;
                    let rhs_requires_grad = rhs.requires_grad;
                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let mut grad_inputs = Vec::new();
                        if lhs_requires_grad {
                            let grad = grad_output
                                .div(&rhs)
                                .unwrap_or(grad_output.clone())
                                .sum_to_shape(&lhs_shape)
                                .unwrap_or(grad_output.clone());
                            grad_inputs.push(grad);
                        }
                        if rhs_requires_grad {
                            let rhs_squared = rhs.mul(&rhs).unwrap_or(rhs.clone());
                            let grad = grad_output
                                .neg()
                                .unwrap_or_else(|_| grad_output.clone())
                                .mul(&lhs)
                                .unwrap_or_else(|_| grad_output.clone())
                                .div(&rhs_squared)
                                .unwrap_or_else(|_| grad_output.clone())
                                .sum_to_shape(&rhs_shape)
                                .unwrap_or(grad_output.clone());
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
