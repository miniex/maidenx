use crate::{utils::promotion::promote_tensor, Tensor, TensorNode};
use maidenx_core::{dtype::DType, error::Result, scalar::Scalar};

impl Tensor {
    pub fn softmax(&self, dim: impl Into<Scalar>) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };
        let input = promote_tensor(self, target_dtype)?;
        let mut result = Tensor::empty_with_spec(input.shape(), input.device(), input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::nn::activation::softmax(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    input.ndim(),
                    dim,
                    Some(&crate::ops::unary::prepare_metadata(&input)),
                )?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let output = result.clone();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let s_t = output.clone();
                let s_grad = grad_out.mul(&s_t)?;

                let mut target_shape = output.shape().to_vec();
                target_shape[dim] = 1;

                let s_grad_sum = s_grad.sum_to_shape(&target_shape)?;
                let broadcasted_sum = s_grad_sum.broadcast_like(&output)?;
                let diff = grad_out.sub(&broadcasted_sum)?;
                let grad_input = output.mul(&diff)?;

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("softmax".to_string(), vec![input.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }
}

