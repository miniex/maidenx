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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_in_place() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.add_(&b)?;

        // Check if `a` is updated in place
        assert_eq!(a.to_vec()?, vec![5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_sub_in_place() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![5.0, 7.0, 9.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.sub_(&b)?;

        // Check if `a` is updated in place
        assert_eq!(a.to_vec()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_mul_in_place() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.mul_(&b)?;

        // Check if `a` is updated in place
        assert_eq!(a.to_vec()?, vec![4.0, 10.0, 18.0]);
        Ok(())
    }

    #[test]
    fn test_div_in_place() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![4.0, 10.0, 18.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.div_(&b)?;

        // Check if `a` is updated in place
        let tolerance = 1e-6;
        assert!(a
            .to_vec()?
            .iter()
            .zip([1.0, 2.0, 3.0].iter())
            .all(|(a, b)| (a - b).abs() < tolerance));
        Ok(())
    }

    #[test]
    fn test_mat_mul_in_place() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
        let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

        a.mat_mul_(&b)?;

        // Check if `a` is updated in place
        // Result should be:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(a.to_vec()?, vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_in_place() -> TensorResult<()> {
        let device = Device::cpu();

        let mut a = Tensor::from_device(vec![1.0], &device)?; // Shape: [1]
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?; // Shape: [3]

        a.add_(&b)?;

        // After broadcasting, `a` should match the shape and values
        assert_eq!(a.to_vec()?, vec![5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_add_in_place() -> TensorResult<()> {
        let device = Device::cuda(0);

        let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
        let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

        a.add_(&b)?;

        // Check if `a` is updated in place
        assert_eq!(a.to_vec()?, vec![5.0, 7.0, 9.0]);
        Ok(())
    }
}
