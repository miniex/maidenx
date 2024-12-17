use super::Tensor;
use crate::error::TensorResult;

impl Tensor {
    pub fn zeros(shape: &[usize]) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::from_vec(data, shape)
    }

    pub fn ones(shape: &[usize]) -> TensorResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![1.0f32; size];
        Self::from_vec(data, shape)
    }
}
