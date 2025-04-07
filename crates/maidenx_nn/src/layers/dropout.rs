use crate::layer::{Layer, LayerState};
use maidenx_core::{dtype::DType, error::Result};
use maidenx_tensor::{Tensor, TensorNode};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dropout {
    p: f32,

    state: LayerState,
}

impl Dropout {
    pub fn new(p: f32) -> Result<Self> {
        Ok(Self { p, state: LayerState::new() })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.state.is_training() {
            let random = Tensor::randn_like(input)?;
            let mut mask = random.gt_scalar(self.p)?;
            mask.with_dtype(DType::U8)?;
            let scale = 1.0 / (1.0 - self.p);
            let scaled_mask = mask.mul_scalar(scale)?;

            if input.requires_grad() {
                let input_clone = input.clone();
                let scaled_mask_clone = scaled_mask.clone();

                let mut output = input.mul(&scaled_mask)?;

                let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                    let grad_input = grad_out.mul(&scaled_mask_clone)?;

                    Ok(vec![grad_input])
                });

                let node = TensorNode::new("dropout".to_string(), vec![input_clone], Some(backward_fn));
                output.set_node(node);

                Ok(output)
            } else {
                input.mul(&scaled_mask)
            }
        } else {
            Ok(input.clone())
        }
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
    fn forward_with_training() -> Result<()> {
        setup_device();
        let mut dropout = Dropout::new(0.5)?;
        dropout.train();

        let input = Tensor::new(vec![vec![1.0f32; 100]; 10])?;
        let output = dropout.forward(&input)?;

        assert_eq!(output.shape(), &[10, 100]);

        let output_vec = output.to_flatten_vec::<f32>()?;

        for &val in &output_vec {
            assert!(val == 0.0 || (val - 2.0).abs() < 1e-5);
        }

        let zeros_count = output_vec.iter().filter(|&&x| x == 0.0).count();
        let non_zeros_count = output_vec.iter().filter(|&&x| (x - 2.0).abs() < 1e-5).count();

        assert!(zeros_count > 0, "There should be at least some zeros");
        assert!(non_zeros_count > 0, "There should be at least some non-zero values");
        assert_eq!(zeros_count + non_zeros_count, 1000, "All values should be either 0 or 2.0");

        Ok(())
    }

    #[test]
    fn forward_with_eval() -> Result<()> {
        setup_device();

        let mut dropout = Dropout::new(0.5)?;
        dropout.eval();

        let input = Tensor::new(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]])?;
        let output = dropout.forward(&input)?;

        assert_eq!(output.shape(), &[2, 2]);
        let output_vec = output.to_flatten_vec::<f32>()?;
        let input_vec = input.to_flatten_vec::<f32>()?;

        for i in 0..output_vec.len() {
            assert!((output_vec[i] - input_vec[i]).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn backward() -> Result<()> {
        setup_device();

        let mut dropout = Dropout::new(0.5)?;
        dropout.train();

        let mut input = Tensor::new(vec![vec![1.0f32; 10]; 10])?;
        input.with_grad()?;

        let output = dropout.forward(&input)?;
        let loss = output.sum_all()?;
        loss.backward()?;

        let input_grad = input.grad()?.expect("Input gradient should exist");
        assert_eq!(input_grad.shape(), &[10, 10]);

        let grad_vec = input_grad.to_flatten_vec::<f32>()?;
        let output_vec = output.to_flatten_vec::<f32>()?;

        for i in 0..grad_vec.len() {
            if output_vec[i] == 0.0 {
                assert!((grad_vec[i]).abs() < 1e-5);
            } else {
                assert!((grad_vec[i] - 2.0).abs() < 1e-5);
            }
        }

        Ok(())
    }
}
