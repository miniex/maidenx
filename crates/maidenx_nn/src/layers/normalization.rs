use crate::layer::{Layer, LayerState};
use maidenx_core::{
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::{Error, Result},
};
use maidenx_tensor::Tensor;

#[derive(Layer, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    normalized_shape: Vec<usize>,
    eps: f32,

    state: LayerState,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, with_bias: bool, eps: f32) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();
        Self::new_with_spec(normalized_shape, with_bias, eps, device, dtype)
    }

    pub fn new_with_spec(normalized_shape: Vec<usize>, with_bias: bool, eps: f32, device: Device, dtype: DType) -> Result<Self> {
        // Initialize weight with ones
        let mut w = Tensor::ones_with_spec(&normalized_shape, device, dtype)?;
        w.with_grad()?;

        // Initialize bias with zeros if needed
        let b = if with_bias {
            let mut b = Tensor::zeros_with_spec(&normalized_shape, device, dtype)?;
            b.with_grad()?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight: w,
            bias: b,
            normalized_shape,
            eps,
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let input_ndim = input_shape.len();
        let normalized_ndim = self.normalized_shape.len();

        // Check that the last dimensions match the normalized shape
        if input_ndim < normalized_ndim {
            return Err(Error::ShapeMismatch {
                expected: normalized_ndim,
                got: input_ndim,
                msg: format!(
                    "Input must have at least {} dimensions for LayerNorm with normalized_shape {:?}",
                    normalized_ndim, self.normalized_shape
                ),
            });
        }

        for i in 0..normalized_ndim {
            let norm_dim = self.normalized_shape[normalized_ndim - 1 - i];
            let input_dim = input_shape[input_ndim - 1 - i];
            if norm_dim != input_dim {
                return Err(Error::DimensionMismatch {
                    expected: self.normalized_shape.clone(),
                    got: input_shape[input_ndim - normalized_ndim..].to_vec(),
                });
            }
        }

        // Compute mean
        let mut mean = input.clone();
        for dim in (input_ndim - normalized_ndim..input_ndim).rev() {
            mean = mean.mean(dim, true)?;
        }

        // Compute variance
        let centered = input.sub(&mean)?;
        let mut variance = centered.pow(2.0)?;
        for dim in (input_ndim - normalized_ndim..input_ndim).rev() {
            variance = variance.mean(dim, true)?;
        }

        // Normalize
        let denom = variance.add_scalar(self.eps)?.sqrt()?;
        let normalized = centered.div(&denom)?;

        // Apply scale and shift (weight and bias)
        let scaled = normalized.mul(&self.weight)?;

        if let Some(ref bias) = self.bias {
            Ok(scaled.add(bias)?)
        } else {
            Ok(scaled)
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

    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    pub fn eps(&self) -> f32 {
        self.eps
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
    fn layer_norm_forward() -> Result<()> {
        setup_device();

        // Create a layer norm for normalizing the last dimension (size 3)
        let layer_norm = LayerNorm::new(vec![3], true, 1e-5)?;

        // Input tensor with shape [2, 3]
        let input = Tensor::new(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;

        let output = layer_norm.forward(&input)?;

        // Output should have the same shape as input
        assert_eq!(output.shape(), &[2, 3]);

        // Verify output values (should be normalized along the last dimension)
        let output_data = output.to_flatten_vec::<f32>()?;
        assert_eq!(output_data.len(), 6);

        // For each row, mean should be close to 0 and standard deviation close to 1
        // before applying weight and bias
        Ok(())
    }

    #[test]
    fn layer_norm_backward() -> Result<()> {
        setup_device();

        let layer_norm = LayerNorm::new(vec![3], true, 1e-5)?;

        let mut input = Tensor::new(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        input.with_grad()?;

        let output = layer_norm.forward(&input)?;
        let loss = output.sum_all()?;
        loss.backward()?;

        // Check gradients
        let input_grad = input.grad()?.expect("Input gradient should exist");
        assert_eq!(input_grad.shape(), &[2, 3]);

        let weight_grad = layer_norm.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[3]);

        if let Some(bias) = layer_norm.bias() {
            let bias_grad = bias.grad()?.expect("Bias gradient should exist");
            assert_eq!(bias_grad.shape(), &[3]);
        }

        Ok(())
    }

    #[test]
    fn layer_norm_multi_dim() -> Result<()> {
        setup_device();

        // Test layer norm with multi-dimensional normalized shape [2, 3]
        let layer_norm = LayerNorm::new(vec![2, 3], true, 1e-5)?;

        // Input tensor with shape [2, 2, 3]
        let input = Tensor::new(vec![
            vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]],
        ])?;

        let output = layer_norm.forward(&input)?;

        // Output should have the same shape as input
        assert_eq!(output.shape(), &[2, 2, 3]);

        Ok(())
    }
}
