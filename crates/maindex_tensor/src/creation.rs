use maidenx_device::Device;

use super::Tensor;
use crate::error::{TensorError, TensorResult};
use rand_distr::{Distribution, Normal};

impl Tensor {
    pub fn zeros(shape: &[usize]) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::from_vec(data, shape)
    }
    pub fn zeros_with_device(shape: &[usize], device: &Device) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::from_vec_with_device(data, shape, device)
    }

    pub fn ones(shape: &[usize]) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![1.0f32; size];
        Self::from_vec(data, shape)
    }
    pub fn ones_with_device(shape: &[usize], device: &Device) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![1.0f32; size];
        Self::from_vec_with_device(data, shape, device)
    }

    pub fn randn(shape: &[usize]) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).map_err(|_e| TensorError::OperationError {
            reason: "Failed to create normal distribution with mean=0.0 and std=1.0".to_string(),
        })?;
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self::from_vec(data, shape)
    }
    pub fn randn_with_device(shape: &[usize], device: &Device) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).map_err(|_e| TensorError::OperationError {
            reason: "Failed to create normal distribution with mean=0.0 and std=1.0".to_string(),
        })?;
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self::from_vec_with_device(data, shape, device)
    }
}
