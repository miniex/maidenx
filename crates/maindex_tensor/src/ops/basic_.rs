// ADD_, SUB_, MUL_, DIV_, MAT_MUL_

use crate::{
    error::{TensorError, TensorResult},
    utils::validate::{assert_device_match, assert_mat_mul_shape_match},
    Tensor,
};
use maidenx_core::buffer::Buffer;
use maidenx_cpu::tensor_ops::basic::{
    cpu_tensor_add, cpu_tensor_div, cpu_tensor_mat_mul, cpu_tensor_mul, cpu_tensor_sub,
};
#[cfg(feature = "cuda")]
use maidenx_cuda::tensor_ops::basic::{
    cuda_tensor_add, cuda_tensor_div, cuda_tensor_mat_mul, cuda_tensor_mul, cuda_tensor_sub,
};
use maidenx_device::Device;

impl Tensor {
    pub fn add_(&mut self, rhs: &Tensor) -> TensorResult<()> {
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

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

        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;

        if self.shape != broadcasted_shape {
            *self = self.broadcast_like(&Tensor {
                buffer: self.buffer.clone(),
                shape: broadcasted_shape.clone(),
                strides: vec![0; broadcasted_shape.len()],
                device: self.device,
                requires_grad: false,
                node: None,
            })?;
        }

        match &self.device {
            Device::Cpu => unsafe {
                cpu_tensor_add(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_add(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        Ok(())
    }

    pub fn sub_(&mut self, rhs: &Tensor) -> TensorResult<()> {
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

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

        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;

        if self.shape != broadcasted_shape {
            *self = self.broadcast_like(&Tensor {
                buffer: self.buffer.clone(),
                shape: broadcasted_shape.clone(),
                strides: vec![0; broadcasted_shape.len()],
                device: self.device,
                requires_grad: false,
                node: None,
            })?;
        }

        match &self.device {
            Device::Cpu => unsafe {
                cpu_tensor_sub(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_sub(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        Ok(())
    }

    pub fn mul_(&mut self, rhs: &Tensor) -> TensorResult<()> {
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

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

        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;

        if self.shape != broadcasted_shape {
            *self = self.broadcast_like(&Tensor {
                buffer: self.buffer.clone(),
                shape: broadcasted_shape.clone(),
                strides: vec![0; broadcasted_shape.len()],
                device: self.device,
                requires_grad: false,
                node: None,
            })?;
        }

        match &self.device {
            Device::Cpu => unsafe {
                cpu_tensor_mul(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_mul(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        Ok(())
    }

    pub fn div_(&mut self, rhs: &Tensor) -> TensorResult<()> {
        let max_rank = self.shape.len().max(rhs.shape.len());
        let mut broadcasted_shape = Vec::with_capacity(max_rank);

        let mut padded_lhs = vec![1; max_rank - self.shape.len()];
        padded_lhs.extend(&self.shape);
        let mut padded_rhs = vec![1; max_rank - rhs.shape.len()];
        padded_rhs.extend(&rhs.shape);

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

        let broadcasted_rhs = rhs.broadcast_like(&Tensor {
            buffer: rhs.buffer.clone(),
            shape: broadcasted_shape.clone(),
            strides: vec![0; broadcasted_shape.len()],
            device: rhs.device,
            requires_grad: false,
            node: None,
        })?;

        assert_device_match(self, rhs)?;

        if self.shape != broadcasted_shape {
            *self = self.broadcast_like(&Tensor {
                buffer: self.buffer.clone(),
                shape: broadcasted_shape.clone(),
                strides: vec![0; broadcasted_shape.len()],
                device: self.device,
                requires_grad: false,
                node: None,
            })?;
        }

        match &self.device {
            Device::Cpu => unsafe {
                cpu_tensor_div(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                )?;
            },
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                cuda_tensor_div(
                    self.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    broadcasted_rhs.buffer.as_ptr(),
                    self.size(),
                    None,
                )?;
            },
        }

        Ok(())
    }

    pub fn mat_mul_(&mut self, rhs: &Tensor) -> TensorResult<()> {
        assert_device_match(self, rhs)?;
        assert_mat_mul_shape_match(self, rhs)?;

        let (m, k) = (self.shape[0] as i32, self.shape[1] as i32);
        let (k2, n) = (rhs.shape[0] as i32, rhs.shape[1] as i32);
        assert_eq!(k, k2, "Incompatible matrix dimensions for multiplication");

        let device = self.device;
        let output_shape = vec![m as usize, n as usize];

        let result_buffer = Buffer::new((m * n) as usize, &device)?;

        match &device {
            Device::Cpu => unsafe {
                cpu_tensor_mat_mul(
                    result_buffer.as_mut_ptr(),
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
                    result_buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    rhs.buffer.as_ptr(),
                    m,
                    n,
                    k,
                    None,
                )?;
            },
        }

        // Update self.buffer and shape
        self.buffer = result_buffer;
        self.shape = output_shape;

        Ok(())
    }
}
