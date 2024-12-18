use crate::error::NnResult;
pub use maidenx_nn_macros::Optimizer;
use maidenx_tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Tensor]) -> NnResult<()>;
    fn zero_grad(&mut self, parameters: &mut [Tensor]) -> NnResult<()>;
    fn set_learning_rate(&mut self, learning_rate: f32);
}
