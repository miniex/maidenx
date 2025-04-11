use crate::layer::{Layer, LayerState};
use maidenx_core::{error::Result, scalar::Scalar};
use maidenx_tensor::Tensor;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Layer, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[layer(inputs = 2)]
pub struct CrossEntropyLoss {
    state: LayerState,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            state: LayerState::new(),
        }
    }

    pub fn forward(&self, (logits, targets): (&Tensor, &Tensor)) -> Result<Tensor> {
        let batch_size = Scalar::new(logits.shape()[0]);

        let softmax = crate::Softmax::new(1)?;
        let probs = softmax.forward(&logits)?;

        let epsilon = Scalar::new(1e-10);
        let probs_safe = probs.add_scalar(epsilon)?;
        let log_probs = probs_safe.log()?;

        let targets_reshaped = if targets.ndim() == 1 {
            targets.reshape(&[targets.shape()[0], 1])?
        } else {
            targets.clone()
        };

        let target_log_probs = log_probs.gather(1, &targets_reshaped)?;
        let nll_loss = target_log_probs.neg()?.sum_all()?.div_scalar(batch_size)?;

        Ok(nll_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_core::device::{set_default_device, Device};

    fn setup_device() {
        #[cfg(feature = "cuda")]
        set_default_device(Device::CUDA(0));
        #[cfg(not(any(feature = "cuda")))]
        set_default_device(Device::CPU);
    }

    #[test]
    fn forward() -> Result<()> {
        setup_device();

        let logits = Tensor::new(vec![vec![1.0f32, 2.0, 0.1], vec![0.1f32, 2.0, 3.0]])?;

        let targets = Tensor::new(vec![0i64, 2])?;

        let ce_loss = CrossEntropyLoss::new();
        let loss = ce_loss.forward((&logits, &targets))?;

        let expected_loss = 0.884864;

        assert!((loss.to_flatten_vec::<f32>()?[0] - expected_loss).abs() < 1e-2);

        Ok(())
    }

    #[test]
    fn backward() -> Result<()> {
        setup_device();

        let mut logits = Tensor::new(vec![vec![1.0f32, 2.0, 0.1], vec![0.1f32, 2.0, 3.0]])?;
        logits.with_grad()?;

        let targets = Tensor::new(vec![0i64, 2])?;

        let ce_loss = CrossEntropyLoss::new();
        let loss = ce_loss.forward((&logits, &targets))?;
        loss.backward()?;

        if let Some(grad) = logits.grad()? {
            let grad_values = grad.to_flatten_vec::<f32>()?;
            let expected_grads = [-0.3788, 0.3295, 0.0493, 0.0193, 0.1293, -0.1486];

            for (g, e) in grad_values.iter().zip(expected_grads.iter()) {
                assert!((g - e).abs() < 1e-2);
            }
        }

        Ok(())
    }
}
