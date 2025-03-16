use crate::layer::Layer;
use maidenx_core::{dtype::DType, error::Result, scalar::Scalar};
use maidenx_tensor::{Tensor, TensorNode};

#[derive(Layer, Clone)]
pub struct Softmax {
    dim: Scalar,
}

impl Softmax {
    pub fn new(dim: impl Into<Scalar>) -> Result<Self> {
        Ok(Self { dim: dim.into() })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dim_i32 = self.dim.as_i32();
        let dim: usize = if dim_i32 < 0 {
            (input.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let target_dtype = if input.dtype().is_int() { DType::F32 } else { input.dtype() };
        let promoted_input = maidenx_tensor::utils::promotion::promote_tensor(input, target_dtype)?;
        let mut result = Tensor::empty_with_spec(promoted_input.shape(), promoted_input.device(), promoted_input.dtype())?;

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::nn::activation::softmax(
                    out_buf,
                    promoted_input.buffer(),
                    promoted_input.size(),
                    promoted_input.ndim(),
                    dim,
                    Some(&prepare_metadata(&promoted_input)),
                )?;
                Ok(())
            })?;
        }

        if input.requires_grad() {
            result.with_grad()?;

            let output = result.clone();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let s_t = output.clone();
                let s_grad = grad_out.mul(&s_t)?;

                let mut target_shape = output.shape().to_vec();
                let last_dim = target_shape.len() - 1;
                target_shape.remove(last_dim);

                let s_grad_sum = s_grad.sum_to_shape(&target_shape)?;
                let broadcasted_sum = s_grad_sum.unsqueeze(last_dim)?.broadcast_like(&output)?;
                let diff = grad_out.sub(&broadcasted_sum)?;
                let grad_input = output.mul(&diff)?;

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("softmax".to_string(), vec![input.clone()], Some(backward_fn));
            result.set_node(node);
        }

        Ok(result)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

fn prepare_metadata(tensor: &Tensor) -> Vec<usize> {
    let mut info = Vec::new();

    // Add dimensions
    info.extend_from_slice(tensor.shape());
    // Add strides
    info.extend_from_slice(tensor.strides());
    // Add offset
    info.push(tensor.offset());

    info
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_core::error::Result;
    use maidenx_tensor::Tensor;

    #[test]
    fn softmax_forward_2d() -> Result<()> {
        // Create a 2D tensor and apply softmax along the last dimension
        let input = Tensor::new(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let softmax = Softmax::new(-1)?;
        let output = softmax.forward(&input)?;

        // Check shape is preserved
        assert_eq!(output.shape(), &[2, 3]);

        // Get output tensor as vector
        let output_vec = output.to_flatten_vec::<f32>()?;
        assert_eq!(output_vec.len(), 6);

        // Sum along each row should be close to 1
        let row1_sum = output_vec[0] + output_vec[1] + output_vec[2];
        let row2_sum = output_vec[3] + output_vec[4] + output_vec[5];

        assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 sum should be close to 1, got {}", row1_sum);
        assert!((row2_sum - 1.0).abs() < 1e-5, "Row 2 sum should be close to 1, got {}", row2_sum);

        // Values should be in increasing order within each row
        assert!(
            output_vec[0] < output_vec[1] && output_vec[1] < output_vec[2],
            "First row should have increasing probabilities"
        );
        assert!(
            output_vec[3] < output_vec[4] && output_vec[4] < output_vec[5],
            "Second row should have increasing probabilities"
        );

        Ok(())
    }

    #[test]
    fn softmax_forward_different_dimension() -> Result<()> {
        // Create a 2D tensor and apply softmax along the first dimension
        let input = Tensor::new(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let softmax = Softmax::new(0)?; // Apply along first dimension
        let output = softmax.forward(&input)?;

        // Check shape is preserved
        assert_eq!(output.shape(), &[2, 3]);

        // Get output tensor as vector
        let output_vec = output.to_flatten_vec::<f32>()?;

        // Check column sums (approximately 1)
        let col1_sum = output_vec[0] + output_vec[3];
        let col2_sum = output_vec[1] + output_vec[4];
        let col3_sum = output_vec[2] + output_vec[5];

        assert!((col1_sum - 1.0).abs() < 1e-5, "Column 1 sum should be close to 1, got {}", col1_sum);
        assert!((col2_sum - 1.0).abs() < 1e-5, "Column 2 sum should be close to 1, got {}", col2_sum);
        assert!((col3_sum - 1.0).abs() < 1e-5, "Column 3 sum should be close to 1, got {}", col3_sum);

        // First row values should be smaller than second row (since inputs are smaller)
        assert!(output_vec[0] < output_vec[3], "Column 1: First row < Second row");
        assert!(output_vec[1] < output_vec[4], "Column 2: First row < Second row");
        assert!(output_vec[2] < output_vec[5], "Column 3: First row < Second row");

        Ok(())
    }

    #[test]
    fn softmax_forward_3d() -> Result<()> {
        // Create a 3D tensor and apply softmax along the last dimension
        let input = Tensor::new(vec![
            vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]],
        ])?;
        let softmax = Softmax::new(-1)?;
        let output = softmax.forward(&input)?;

        // Check shape is preserved
        assert_eq!(output.shape(), &[2, 2, 3]);

        // Each slice (last dimension) should sum to approximately 1
        let output_data = output.to_flatten_vec::<f32>()?;

        for i in 0..4 {
            let slice_sum = output_data[i * 3] + output_data[i * 3 + 1] + output_data[i * 3 + 2];
            assert!((slice_sum - 1.0).abs() < 1e-5, "Slice {} sum should be close to 1, got {}", i, slice_sum);
        }

        Ok(())
    }

    #[test]
    fn softmax_backward() -> Result<()> {
        // Create a simple input tensor with gradients enabled
        let mut input = Tensor::new(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]])?;
        input.with_grad()?;

        // Apply softmax and compute loss
        let softmax = Softmax::new(-1)?;
        let output = softmax.forward(&input)?;

        // Sum all elements (simple loss function)
        let loss = output.sum_all()?;

        // Backward pass
        loss.backward()?;

        // Get input gradient
        let input_grad = input.grad()?.expect("Input gradient should exist");

        // Check gradient shape
        assert_eq!(input_grad.shape(), &[2, 2]);

        // Check that gradients exist and are not all zero
        let grad_vec = input_grad.to_flatten_vec::<f32>()?;
        let has_nonzero = grad_vec.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "Gradient should not be all zeros");

        Ok(())
    }

    #[test]
    fn softmax_integer_input() -> Result<()> {
        // Test that softmax works with integer inputs by promoting to float
        let input = Tensor::new(vec![vec![1i32, 2, 3], vec![4, 5, 6]])?;
        let softmax = Softmax::new(-1)?;
        let output = softmax.forward(&input)?;

        // Output should be float (f32)
        assert_eq!(output.dtype(), DType::F32);

        // Check shape is preserved
        assert_eq!(output.shape(), &[2, 3]);

        // Each row should sum to approximately 1
        let output_vec = output.to_flatten_vec::<f32>()?;
        let row1_sum = output_vec[0] + output_vec[1] + output_vec[2];
        let row2_sum = output_vec[3] + output_vec[4] + output_vec[5];

        assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 sum should be close to 1, got {}", row1_sum);
        assert!((row2_sum - 1.0).abs() < 1e-5, "Row 2 sum should be close to 1, got {}", row2_sum);

        Ok(())
    }
}
