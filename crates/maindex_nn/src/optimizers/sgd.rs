use crate::{error::NnResult, optimizer::Optimizer};
use maidenx_tensor::Tensor;

#[derive(Optimizer)]
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub fn step(&mut self, parameters: &mut [Tensor]) -> NnResult<()> {
        for param in parameters.iter_mut() {
            if let Some(grad) = param.grad()? {
                param.sub_(&grad.scalar_mul(self.learning_rate)?)?;
            }
        }
        Ok(())
    }

    pub fn zero_grad(&mut self, parameters: &mut [Tensor]) -> NnResult<()> {
        for param in parameters.iter_mut() {
            param.zero_grad()?;
        }
        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}
