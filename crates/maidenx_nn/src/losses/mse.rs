use crate::layer::{Layer, LayerState};
use maidenx_core::{error::Result, scalar::Scalar};
use maidenx_tensor::Tensor;

#[derive(Layer, Clone, Default)]
#[layer(inputs = 2)]
pub struct MSE {
    state: LayerState,
}

impl MSE {
    pub fn new() -> Self {
        Self { state: LayerState::new() }
    }

    pub fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let squared = diff.pow(2.0)?;

        let num_elements = Scalar::new(squared.size());
        let mean = squared.sum_all()?.div_scalar(num_elements)?;

        Ok(mean)
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
        let mse_loss = MSE::new();

        let loss = mse_loss.forward((&pred, &target))?;

        assert_eq!(loss.to_flatten_vec::<f32>()?, vec![1.0]);

        Ok(())
    }

    #[test]
    fn backward() -> Result<()> {
        setup_device();

        let mut pred = Tensor::new(vec![2.0f32, 3.0, 4.0])?;
        pred.with_grad()?;
        let target = Tensor::new(vec![1.0f32, 2.0, 3.0])?;
        let mse_loss = MSE::new();
        let loss = mse_loss.forward((&pred, &target))?;

        loss.backward()?;

        if let Some(grad) = pred.grad()? {
            let pred_grad = grad.to_flatten_vec::<f32>()?;
            let expected_grad = vec![2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
            assert_eq!(pred_grad, expected_grad);
        }

        Ok(())
    }
}
