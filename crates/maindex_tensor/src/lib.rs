mod adapter;
mod creation;
pub mod error;
mod gradient;
mod ops;
mod tape;
pub mod utils;

use adapter::TensorAdapter;
use error::{TensorError, TensorResult};
use gradient::Node;
use maidenx_core::buffer::Buffer;
use maidenx_device::{get_current_device, Device};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Tensor {
    buffer: Buffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
    device: Device,
    // gradient
    requires_grad: bool,
    node: Option<Arc<Mutex<Node>>>,
}

impl Tensor {
    pub fn new<T>(data: T) -> TensorResult<Self>
    where
        T: TensorAdapter,
    {
        let device = get_current_device();
        Self::from_device(data, &device)
    }

    pub fn from_device<T>(data: T, device: &Device) -> TensorResult<Self>
    where
        T: TensorAdapter,
    {
        let shape = data.to_shape();
        let data = data.to_flat_vec();
        let strides = Self::compute_strides(&shape);

        let mut buffer = Buffer::new(data.len(), device)?;
        buffer.copy_from_host(&data)?;

        Ok(Self {
            buffer,
            shape,
            strides,
            device: *device,
            // gradient
            requires_grad: false,
            node: None,
        })
    }

    pub fn from_vec<T: Into<Vec<f32>>>(vec: T, shape: &[usize]) -> TensorResult<Self> {
        let device = get_current_device();
        Self::from_vec_with_device(vec, shape, &device)
    }

    pub fn from_vec_with_device<T: Into<Vec<f32>>>(
        vec: T,
        shape: &[usize],
        device: &Device,
    ) -> TensorResult<Self> {
        let data = vec.into();
        let strides = Self::compute_strides(shape);
        let total_size: usize = shape.iter().product();

        if data.len() != total_size {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Data size: {}, but got size from shape: {}",
                    data.len(),
                    total_size
                ),
            });
        }

        let mut buffer = Buffer::new(data.len(), device)?;
        buffer.copy_from_host(&data)?;

        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            strides,
            device: *device,
            requires_grad: false,
            node: None,
        })
    }

    pub fn with_grad(&mut self) {
        self.requires_grad = true;
        self.node = Some(Arc::new(Mutex::new(Node {
            grad_fn: None,
            inputs: vec![],
            grad: None,
        })));
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // get
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    pub fn buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }

    pub fn to_vec(&self) -> TensorResult<Vec<f32>> {
        let num_elements = self.shape.iter().product::<usize>();
        let mut result = vec![0.0f32; num_elements];
        self.buffer.copy_to_host(&mut result)?;
        Ok(result)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn size_dim(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cpu_tensor() {
        let device = Device::cpu();

        // 1D tensor
        let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device).unwrap();
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        // 2D tensor
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tensor = Tensor::from_device(data, &device).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // Scalar
        let tensor = Tensor::from_device(5.0f32, &device).unwrap();
        assert_eq!(tensor.shape(), &[1]);
        assert_eq!(tensor.to_vec().unwrap(), vec![5.0]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_new_cuda_tensor() {
        let device = Device::cuda(0);

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tensor = Tensor::from_device(data, &device).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
