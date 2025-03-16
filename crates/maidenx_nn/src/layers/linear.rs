use crate::layer::Layer;
use maidenx_core::{
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::Result,
};
use maidenx_tensor::Tensor;

#[derive(Layer, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, with_bias: bool) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::new_with_spec(in_features, out_features, with_bias, device, dtype)
    }

    pub fn new_with_spec(in_features: usize, out_features: usize, with_bias: bool, device: Device, dtype: DType) -> Result<Self> {
        let k: f32 = 1.0 / (in_features as f32).sqrt();

        // weight
        let mut w = Tensor::randn_with_spec(&[out_features, in_features], device, dtype)?;
        w.with_grad()?;
        w.mul_scalar(k)?;

        // bias
        let b = if with_bias {
            let mut b = Tensor::randn_with_spec(&[], device, dtype)?;
            b.with_grad()?;
            b.mul_scalar(k)?;

            Some(b)
        } else {
            None
        };

        Ok(Self { weight: w, bias: b })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let batch_dims = if input_shape.len() >= 2 {
            &input_shape[..input_shape.len() - 2]
        } else {
            &[]
        };

        let broadcasted_weight = if !batch_dims.is_empty() {
            self.weight.broadcast_left(batch_dims)?
        } else {
            self.weight.clone()
        };

        let output = input.matmul(&broadcasted_weight.transpose(-1, -2)?)?;

        if let Some(ref bias) = self.bias {
            Ok(output.add(bias)?)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![];
        params.push(&mut self.weight);
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_core::device::set_default_device;

    fn setup_device() {
        #[cfg(feature = "cuda")]
        set_default_device(Device::CUDA(0));
        #[cfg(not(any(feature = "cuda")))]
        set_default_device(Device::CPU);
    }

    #[test]
    fn linear_forward() -> Result<()> {
        setup_device();

        let linear = Linear::new(2, 3, true)?;

        let input = Tensor::new(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]])?;
        let output = linear.forward(&input)?;

        assert_eq!(output.shape(), &[2, 3]);

        let output_vec = output.to_flatten_vec::<f32>()?;
        assert_eq!(output_vec.len(), 6);

        Ok(())
    }

    #[test]
    fn linear_backward() -> Result<()> {
        setup_device();

        let linear = Linear::new(2, 3, true)?;

        let mut input = Tensor::new(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]])?;
        input.with_grad()?;
        let output = linear.forward(&input)?;
        let loss = output.sum_all()?;
        loss.backward()?;

        let input_grad = input.grad()?.expect("Input gradient should exist");
        assert_eq!(input_grad.shape(), &[2, 2]);

        let weight_grad = linear.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[3, 2]);

        if let Some(bias) = linear.bias() {
            let bias_grad = bias.grad()?.expect("Bias gradient should exist");
            assert_eq!(bias_grad.shape(), &[]);
        }

        Ok(())
    }
}
