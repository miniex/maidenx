use crate::layer::Layer;
use maidenx_core::{error::Result, scalar::Scalar};
use maidenx_tensor::Tensor;

#[derive(Layer, Clone, Default)]
#[layer(inputs = 2)]
pub struct MAE {}

impl MAE {
    pub fn new() -> Self {
        MAE::default()
    }

    pub fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        let batch_size = Scalar::new(pred.shape()[0]);
        let mean = abs_diff.sum_all()?.div_scalar(batch_size)?;

        Ok(mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() -> Result<()> {
        let pred = Tensor::new(vec![2.0f32, 3.0, 4.0])?;
        let target = Tensor::new(vec![1.0f32, 2.0, 3.0])?;

        let mae_loss = MAE::new();
        let loss = mae_loss.forward((&pred, &target))?;

        assert_eq!(loss.to_flatten_vec::<f32>()?, vec![1.0]);

        Ok(())
    }

    #[test]
    fn backward() -> Result<()> {
        let mut pred = Tensor::new(vec![2.0f32, 3.0, 4.0])?;
        pred.with_grad()?;

        let target = Tensor::new(vec![1.0f32, 2.0, 3.0])?;

        let mae_loss = MAE::new();
        let loss = mae_loss.forward((&pred, &target))?;
        loss.backward()?;

        if let Some(grad) = pred.grad()? {
            let pred_grad = grad.to_flatten_vec::<f32>()?;
            let expected_grad = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
            assert_eq!(pred_grad, expected_grad);
        }

        Ok(())
    }
}
