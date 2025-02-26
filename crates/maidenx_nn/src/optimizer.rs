use maidenx_core::{error::Result, scalar::Scalar};
pub use maidenx_nn_macros::Optimizer;
use maidenx_tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;
    fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>);
}
