use crate::layer::{Layer, LayerState};
use maidenx_core::{
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::{Error, Result},
};
use maidenx_tensor::Tensor;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Embedding {
    weight: Tensor,
    padding_idx: Option<usize>,
    max_norm: Option<f32>,
    norm_type: f32,
    scale_grad_by_freq: bool,
    state: LayerState,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::new_with_spec(num_embeddings, embedding_dim, None, None, 2.0, false, device, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_spec(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: f32,
        scale_grad_by_freq: bool,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let k: f32 = 1.0 / (embedding_dim as f32).sqrt();
        let mut w = Tensor::randn_with_spec(&[num_embeddings, embedding_dim], device, dtype)?;
        w.with_grad()?;
        w.mul_scalar(k)?;

        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(Error::IndexOutOfBounds {
                    index: idx,
                    size: num_embeddings,
                });
            }

            let zero_emb = Tensor::zeros_with_spec(&[embedding_dim], device, dtype)?;
            w.index_put_(&[idx], &zero_emb)?;
        }

        Ok(Self {
            weight: w,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !input.dtype().is_int() {
            return Err(Error::InvalidArgument(format!(
                "Embedding requires integer input, got {}",
                input.dtype()
            )));
        }

        let mut output = self.weight.index_select(0, input)?;

        if let Some(max_norm) = self.max_norm {
            let norms = output.norm(self.norm_type, -1, true)?;
            let mask = norms.gt_scalar(max_norm)?;

            if mask.any()? {
                let scale = norms.div_scalar(max_norm)?;

                let ones = Tensor::ones_like(&scale)?;
                let scale_factors = scale.maximum(&ones)?;

                let normed_output = output.div(&scale_factors)?;

                output = normed_output;
            }
        }

        {
            let input_clone = input.clone();
            let weight_clone = self.weight.clone();
            let scale_grad = self.scale_grad_by_freq;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let mut grad_weight = Tensor::zeros_like(&weight_clone)?;

                if scale_grad {
                    let counts = input_clone.bincount(None, None)?;

                    for i in 0..input_clone.size() {
                        let idx = input_clone.get(&[i])?.as_i32() as usize;
                        let count = counts.get(&[idx])?.as_f32().max(1.0);
                        let scale = 1.0 / count;

                        let pos_grad = grad_out.slice(0, i, Some(i + 1), 1)?;
                        let scaled_grad = pos_grad.mul_scalar(scale)?;

                        let idx_tensor = Tensor::new(vec![idx as i32])?;

                        grad_weight.index_add_(0, &idx_tensor, &scaled_grad)?;
                    }
                } else {
                    grad_weight.index_add_(0, &input_clone, grad_out)?;
                }

                Ok(vec![grad_weight])
            });

            let node =
                maidenx_tensor::TensorNode::new("embedding".to_string(), vec![self.weight.clone()], Some(backward_fn));

            output.set_node(node);
        }

        Ok(output)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn num_embeddings(&self) -> usize {
        self.weight.shape()[0]
    }

    pub fn embedding_dim(&self) -> usize {
        self.weight.shape()[1]
    }

    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    pub fn max_norm(&self) -> Option<f32> {
        self.max_norm
    }

    pub fn norm_type(&self) -> f32 {
        self.norm_type
    }

    pub fn scale_grad_by_freq(&self) -> bool {
        self.scale_grad_by_freq
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
    fn init_with_padding_idx() -> Result<()> {
        setup_device();

        let embedding = Embedding::new_with_spec(
            10,
            5,
            Some(2),
            None,
            2.0,
            false,
            get_default_device(),
            get_default_dtype(),
        )?;

        let padding_weights = embedding.weight().slice(0, 2, Some(3), 1)?.reshape(&[5])?;
        let padding_sum = padding_weights.sum_all()?.item()?.as_f32();

        assert!((padding_sum - 0.0).abs() < 1e-6, "Padding embedding should be zeros");

        Ok(())
    }

    #[test]
    fn forward() -> Result<()> {
        setup_device();

        let embedding = Embedding::new(10, 5)?;
        let input = Tensor::new(vec![1i64, 3, 5, 7, 2])?;
        let output = embedding.forward(&input)?;

        assert_eq!(output.shape(), &[5, 5]);

        Ok(())
    }

    #[test]
    fn forward_with_max_norm() -> Result<()> {
        setup_device();

        let embedding = Embedding::new_with_spec(
            10,
            5,
            None,
            Some(1.0),
            2.0,
            false,
            get_default_device(),
            get_default_dtype(),
        )?;
        let input = Tensor::new(vec![0i64, 1, 2, 3, 4])?;
        let output = embedding.forward(&input)?;

        let norms = output.norm(2.0, 1, false)?;

        let norms_vec = norms.to_flatten_vec::<f32>()?;
        for norm in norms_vec {
            assert!(norm <= 1.0 + 1e-5, "Embedding norm should not exceed max_norm");
        }

        Ok(())
    }

    #[test]
    fn backward() -> Result<()> {
        setup_device();

        let embedding = Embedding::new(10, 5)?;

        let input = Tensor::new(vec![1i64, 3, 1, 5, 3])?;

        let output = embedding.forward(&input)?;

        let loss = output.sum_all()?;

        loss.backward()?;

        let weight_grad = embedding.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[10, 5]);

        let grad_1 = weight_grad.slice(0, 1, Some(2), 1)?.reshape(&[5])?;
        let grad_3 = weight_grad.slice(0, 3, Some(4), 1)?.reshape(&[5])?;
        let grad_5 = weight_grad.slice(0, 5, Some(6), 1)?.reshape(&[5])?;

        let grad_1_sum = grad_1.sum_all()?.item()?.as_f32();
        let grad_3_sum = grad_3.sum_all()?.item()?.as_f32();
        let grad_5_sum = grad_5.sum_all()?.item()?.as_f32();

        assert!(
            (grad_1_sum - 2.0 * grad_5_sum).abs() < 1e-5,
            "Gradient for index 1 should be accumulated across its two occurrences"
        );
        assert!(
            (grad_3_sum - 2.0 * grad_5_sum).abs() < 1e-5,
            "Gradient for index 3 should be accumulated across its two occurrences"
        );

        Ok(())
    }

    #[test]
    fn backward_with_scale_grad_by_freq() -> Result<()> {
        setup_device();

        let embedding =
            Embedding::new_with_spec(10, 5, None, None, 2.0, true, get_default_device(), get_default_dtype())?;

        let input = Tensor::new(vec![1i64, 3, 1, 5, 3])?;

        let output = embedding.forward(&input)?;

        let loss = output.sum_all()?;

        loss.backward()?;

        let weight_grad = embedding.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[10, 5]);

        let grad_1 = weight_grad.slice(0, 1, Some(2), 1)?.reshape(&[5])?;
        let grad_3 = weight_grad.slice(0, 3, Some(4), 1)?.reshape(&[5])?;
        let grad_5 = weight_grad.slice(0, 5, Some(6), 1)?.reshape(&[5])?;

        let grad_1_sum = grad_1.sum_all()?.item()?.as_f32();
        let grad_3_sum = grad_3.sum_all()?.item()?.as_f32();
        let grad_5_sum = grad_5.sum_all()?.item()?.as_f32();

        println!("{}", grad_1_sum);
        println!("{}", grad_3_sum);
        println!("{}", grad_5_sum);

        assert!(
            (grad_1_sum - grad_5_sum).abs() < 1e-5,
            "With scale_grad_by_freq, index 1 gradient should be scaled by 1/frequency"
        );
        assert!(
            (grad_3_sum - grad_5_sum).abs() < 1e-5,
            "With scale_grad_by_freq, index 3 gradient should be scaled by 1/frequency"
        );

        Ok(())
    }

    #[test]
    fn backward_with_max_norm() -> Result<()> {
        setup_device();

        let embedding = Embedding::new_with_spec(
            10,
            5,
            None,
            Some(1.0),
            2.0,
            false,
            get_default_device(),
            get_default_dtype(),
        )?;

        let input = Tensor::new(vec![0i64, 1, 2, 3, 4])?;

        let output = embedding.forward(&input)?;

        let norms = output.norm(2.0, 1, false)?;
        let norms_vec = norms.to_flatten_vec::<f32>()?;
        for norm in norms_vec {
            assert!(norm <= 1.0 + 1e-5, "Forward: Embedding norm should not exceed max_norm");
        }

        let loss = output.sum_all()?;

        loss.backward()?;

        let weight_grad = embedding.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[10, 5]);

        for idx in 0..5 {
            let grad_idx = weight_grad.slice(0, idx, Some(idx + 1), 1)?.reshape(&[5])?;
            let grad_sum = grad_idx.sum_all()?.item()?.as_f32();
            assert!(grad_sum.abs() > 1e-6, "Gradient for used index should be non-zero");
        }

        Ok(())
    }

    #[test]
    fn backward_with_padding_idx() -> Result<()> {
        setup_device();

        let embedding = Embedding::new_with_spec(
            10,
            5,
            Some(2),
            None,
            2.0,
            false,
            get_default_device(),
            get_default_dtype(),
        )?;

        let input = Tensor::new(vec![1i64, 2, 3, 2, 4])?;

        let output = embedding.forward(&input)?;

        let padding_indices = Tensor::new(vec![2i64])?;
        let padding_embed = embedding.weight().index_select(0, &padding_indices)?;
        let padding_sum = padding_embed.sum_all()?.item()?.as_f32();
        assert!((padding_sum - 0.0).abs() < 1e-6, "Padding embedding should be zeros");

        let loss = output.sum_all()?;

        loss.backward()?;

        let weight_grad = embedding.weight().grad()?.expect("Weight gradient should exist");
        assert_eq!(weight_grad.shape(), &[10, 5]);

        let padding_grad = weight_grad.slice(0, 2, Some(3), 1)?.reshape(&[5])?;
        let padding_grad_sum = padding_grad.sum_all()?.item()?.as_f32();
        assert!(
            padding_grad_sum.abs() > 1e-6,
            "Padding index gradient should be non-zero from backprop"
        );

        Ok(())
    }
}
