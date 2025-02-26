use crate::optimizer::Optimizer;
use maidenx_core::{error::Result, scalar::Scalar};
use maidenx_tensor::Tensor;

#[derive(Optimizer)]
pub struct SGD {
    learning_rate: Scalar,
}

impl SGD {
    pub fn new(learning_rate: impl Into<Scalar>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
        }
    }

    pub fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        for param in parameters.iter_mut() {
            if let Some(grad) = param.grad()? {
                let lr = self.learning_rate;

                param.sub_(&grad.mul_scalar(lr)?)?;
            }
        }
        Ok(())
    }

    pub fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        for param in parameters.iter_mut() {
            param.zero_grad()?;
        }
        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }
}
