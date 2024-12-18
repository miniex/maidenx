use crate::{error::NnResult, module::Module};
use maidenx_tensor::{gradient::Node, tape::TENSOR_TAPE, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Default, Module)]
#[module(inputs = 2)]
pub struct MSELoss {}

impl MSELoss {
    pub fn new() -> Self {
        MSELoss::default()
    }

    pub fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> NnResult<Tensor> {
        let diff = pred.sub(target)?;
        let squared = diff.pow(2.0)?;
        let batch_size = pred.shape()[0] as f32;

        let mean = squared.sum()?.scalar_div(batch_size)?;

        let requires_grad = pred.is_requires_grad() || target.is_requires_grad();
        let node = if requires_grad {
            Some(Arc::new(Mutex::new(Node {
                grad_fn: Some(Box::new({
                    let pred = pred.clone();
                    let target = target.clone();

                    move |grad_output: &Tensor| -> Vec<Tensor> {
                        let batch_size = pred.shape()[0] as f32;
                        let grad_pred = grad_output
                            .mul(&pred.sub(&target).unwrap()) // (pred - target)
                            .unwrap()
                            .scalar_mul(2.0 / batch_size) // (2/N)
                            .unwrap();

                        vec![grad_pred]
                    }
                })),
                inputs: vec![
                    pred.node().map(|n| Arc::downgrade(&n)).unwrap_or_default(),
                    target
                        .node()
                        .map(|n| Arc::downgrade(&n))
                        .unwrap_or_default(),
                ],
                grad: None,
            })))
        } else {
            None
        };

        let mut output = mean.clone();
        output.set_requires_grad(requires_grad);
        output.set_node(node);

        TENSOR_TAPE.with(|tape| tape.borrow_mut().add(output.clone()));

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_device::Device;
    use maidenx_tensor::Tensor;

    #[test]
    fn test_mse_loss_forward() -> NnResult<()> {
        let device = Device::cpu();

        let pred = Tensor::from_vec_with_device(vec![2.0, 3.0, 4.0], &[3], &device)?;
        let target = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[3], &device)?;

        let mse_loss = MSELoss::new();
        let loss = mse_loss.forward((&pred, &target))?;

        println!("Loss: {:?}", loss.to_vec()?);
        assert_eq!(loss.to_vec()?, vec![1.0]);

        Ok(())
    }

    #[test]
    fn test_mse_loss_backward() -> NnResult<()> {
        let device = Device::cpu();

        let mut pred = Tensor::from_vec_with_device(vec![2.0, 3.0, 4.0], &[3], &device)?;
        pred.with_grad();

        let target = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[3], &device)?;

        let mse_loss = MSELoss::new();
        let loss = mse_loss.forward((&pred, &target))?;
        loss.backward()?;

        let pred_grad = pred.grad()?.unwrap().to_vec()?;
        println!("Gradient w.r.t pred: {:?}", pred_grad);

        // MSE Loss Gradient: 2 * (pred - target) / N
        let expected_grad = vec![2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        assert_eq!(pred_grad, expected_grad);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_mse_loss_forward_cuda() -> NnResult<()> {
        let device = Device::cuda(0);

        let pred = Tensor::from_vec_with_device(vec![2.0, 3.0, 4.0], &[3], &device)?;
        let target = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[3], &device)?;

        let mse_loss = MSELoss::new();
        let loss = mse_loss.forward((&pred, &target))?;

        println!("Loss (CUDA): {:?}", loss.to_vec()?);
        assert_eq!(loss.to_vec()?, vec![1.0]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_mse_loss_backward_cuda() -> NnResult<()> {
        let device = Device::cuda(0);

        let mut pred = Tensor::from_vec_with_device(vec![2.0, 3.0, 4.0], &[3], &device)?;
        pred.with_grad();

        let target = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[3], &device)?;

        let mse_loss = MSELoss::new();
        let loss = mse_loss.forward((&pred, &target))?;
        loss.backward()?;

        let pred_grad = pred.grad()?.unwrap().to_vec()?;
        println!("Gradient w.r.t pred (CUDA): {:?}", pred_grad);

        let expected_grad = vec![2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        assert_eq!(pred_grad, expected_grad);

        Ok(())
    }
}
