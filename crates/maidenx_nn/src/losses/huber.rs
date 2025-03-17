use crate::layer::{Layer, LayerState};
use maidenx_core::{error::Result, scalar::Scalar};
use maidenx_tensor::Tensor;

#[derive(Layer, Clone)]
#[layer(inputs = 2)]
pub struct HuberLoss {
    delta: Scalar,

    state: LayerState,
}

impl HuberLoss {
    pub fn new(delta: impl Into<Scalar>) -> Self {
        Self {
            delta: delta.into(),
            state: LayerState::new(),
        }
    }

    pub fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        // Compute quadratic terms for small differences
        let quadratic_loss = diff.pow(2.0)?.div_scalar(2.0)?;

        // Compute linear terms for large differences
        let linear_loss = abs_diff.mul_scalar(self.delta)?.sub_scalar(self.delta.powi(2) / Scalar::new(2))?;

        // Create mask for selecting between quadratic and linear terms
        let mask = abs_diff.le_scalar(self.delta)?.to_dtype(abs_diff.dtype())?;

        // Combine losses using mask
        let loss = mask
            .mul(&quadratic_loss)?
            .add(&mask.logical_not()?.to_dtype(mask.dtype())?.mul(&linear_loss)?)?;

        // Compute mean loss
        let batch_size = Scalar::new(pred.shape()[0]);
        let mean_loss = loss.sum_all()?.div_scalar(batch_size)?;

        Ok(mean_loss)
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

        let pred = Tensor::new(vec![2.0f32, 3.0, 4.0])?;
        let target = Tensor::new(vec![1.0f32, 2.0, 3.0])?;
        let huber_loss = HuberLoss::new(1.0);
        let loss = huber_loss.forward((&pred, &target))?;

        let expected_loss = (0.5 + 0.5 + 0.5) / 3.0;
        assert!((loss.to_flatten_vec::<f32>()?[0] - expected_loss).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn backward() -> Result<()> {
        setup_device();

        let mut pred = Tensor::new(vec![2.0f32, 3.0, 4.0])?;
        pred.with_grad()?;
        let target = Tensor::new(vec![1.0f32, 2.0, 3.0])?;
        let huber_loss = HuberLoss::new(1.0);
        let loss = huber_loss.forward((&pred, &target))?;
        loss.backward()?;

        if let Some(grad) = pred.grad()? {
            let pred_grad = grad.to_flatten_vec::<f32>()?;
            let expected_grad = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
            for (g, e) in pred_grad.iter().zip(expected_grad.iter()) {
                assert!((g - e).abs() < 1e-6);
            }
        }

        Ok(())
    }
}
