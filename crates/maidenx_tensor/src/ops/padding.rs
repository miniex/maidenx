use crate::{utils::promotion::promote_tensor, Tensor, TensorNode};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
    pub fn pad(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor> {
        self.pad_with_constant(paddings, pad_value)
    }

    pub fn pad_with_constant(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() { DType::U8 } else { self.dtype() };
        let input = promote_tensor(self, target_dtype)?;

        // Calculate output shape
        let mut output_shape = Vec::with_capacity(input.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(input.shape()[i] + pad_before + pad_after);
        }

        // Convert pad_value to scalar
        let pad_scalar = pad_value.into();

        // Create output tensor
        let mut result = Self::empty_with_spec(&output_shape, input.device(), input.dtype())?;

        // Prepare metadata for padding operation
        let metadata = prepare_metadata_for_padding(&input, paddings);

        unsafe {
            let result_size = result.size();
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::padding::pad_with_constant(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    result_size,
                    input.ndim(),
                    Some(&metadata),
                    pad_scalar,
                )?;
                Ok(())
            })?;
        }

        if input.requires_grad() {
            result.with_grad()?;

            let input_shape = input.shape().to_vec();
            let input_strides = input.strides().to_vec();
            let paddings_vec = paddings.to_vec();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let mut grad_in = Tensor::empty_with_spec(&input_shape, grad_out.device(), grad_out.dtype())?;

                let metadata = prepare_metadata_for_padding_backward(&input_shape, &input_strides, 0, &paddings_vec);

                unsafe {
                    let grad_in_size = grad_in.size();
                    grad_in.with_buffer_mut(|in_buf| {
                        maidenx_core::be::ops::padding::pad_with_constant_backward(
                            in_buf,
                            grad_out.buffer(),
                            grad_in_size,
                            grad_out.size(),
                            grad_out.ndim(),
                            Some(&metadata),
                        )
                    })?;
                }

                Ok(vec![grad_in])
            });

            let node = TensorNode::new("pad_with_constant".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn pad_with_reflection(&self, paddings: &[(usize, usize)]) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() { DType::U8 } else { self.dtype() };
        let input = promote_tensor(self, target_dtype)?;

        // Calculate output shape
        let mut output_shape = Vec::with_capacity(input.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            let dim_size = input.shape()[i];

            if pad_before >= dim_size || pad_after >= dim_size {
                return Err(Error::InvalidArgument(format!(
                    "Reflection padding width ({}, {}) must be less than the dimension size ({})",
                    pad_before, pad_after, dim_size
                )));
            }

            if dim_size <= 1 && (pad_before > 0 || pad_after > 0) {
                return Err(Error::InvalidArgument(format!(
                    "Reflection padding requires input dimension > 1 for non-zero padding, but got dim {} with size {}",
                    i, dim_size
                )));
            }
            output_shape.push(dim_size + pad_before + pad_after);
        }

        // Create output tensor
        let mut result = Self::empty_with_spec(&output_shape, input.device(), input.dtype())?;

        // Prepare metadata for padding operation
        let metadata = prepare_metadata_for_padding(&input, paddings);

        unsafe {
            let result_size = result.size();
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::padding::pad_with_reflection(out_buf, input.buffer(), input.size(), result_size, input.ndim(), Some(&metadata))
            })?;
        }

        if input.requires_grad() {
            result.with_grad()?;

            let input_shape = input.shape().to_vec();
            let input_strides = input.strides().to_vec();
            let paddings_vec = paddings.to_vec();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let mut grad_in = Tensor::empty_with_spec(&input_shape, grad_out.device(), grad_out.dtype())?;

                let metadata = prepare_metadata_for_padding_backward(&input_shape, &input_strides, 0, &paddings_vec);

                unsafe {
                    let grad_in_size = grad_in.size();
                    grad_in.with_buffer_mut(|in_buf| {
                        maidenx_core::be::ops::padding::pad_with_reflection_backward(
                            in_buf,
                            grad_out.buffer(),
                            grad_in_size,
                            grad_out.size(),
                            grad_out.ndim(),
                            Some(&metadata),
                        )
                    })?;
                }

                Ok(vec![grad_in])
            });

            let node = TensorNode::new("pad_with_reflection".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn pad_with_replication(&self, paddings: &[(usize, usize)]) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() { DType::U8 } else { self.dtype() };
        let input = promote_tensor(self, target_dtype)?;

        // Calculate output shape
        let mut output_shape = Vec::with_capacity(input.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(input.shape()[i] + pad_before + pad_after);
        }

        // Create output tensor
        let mut result = Self::empty_with_spec(&output_shape, input.device(), input.dtype())?;

        // Prepare metadata for padding operation
        let metadata = prepare_metadata_for_padding(&input, paddings);

        unsafe {
            let result_size = result.size();
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::padding::pad_with_replication(
                    out_buf,
                    input.buffer(),
                    input.size(),
                    result_size,
                    input.ndim(),
                    Some(&metadata),
                )
            })?;
        }

        if input.requires_grad() {
            result.with_grad()?;

            let input_shape = input.shape().to_vec();
            let input_strides = input.strides().to_vec();
            let paddings_vec = paddings.to_vec();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let mut grad_in = Tensor::empty_with_spec(&input_shape, grad_out.device(), grad_out.dtype())?;

                let metadata = prepare_metadata_for_padding_backward(&input_shape, &input_strides, 0, &paddings_vec);

                unsafe {
                    let grad_in_size = grad_in.size();
                    grad_in.with_buffer_mut(|in_buf| {
                        maidenx_core::be::ops::padding::pad_with_replication_backward(
                            in_buf,
                            grad_out.buffer(),
                            grad_in_size,
                            grad_out.size(),
                            grad_out.ndim(),
                            Some(&metadata),
                        )
                    })?;
                }

                Ok(vec![grad_in])
            });

            let node = TensorNode::new("pad_with_replication".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }
}

fn prepare_metadata_for_padding(tensor: &Tensor, paddings: &[(usize, usize)]) -> Vec<usize> {
    let mut info = Vec::new();

    // Input dimensions
    info.extend_from_slice(tensor.shape());
    // Input strides
    info.extend_from_slice(tensor.strides());
    // Input offset
    info.push(tensor.offset());

    // Output dimensions
    for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
        info.push(tensor.shape()[i] + pad_before + pad_after);
    }

    // Padding values (before, after for each dimension)
    for &(pad_before, pad_after) in paddings {
        info.push(pad_before);
        info.push(pad_after);
    }

    info
}

fn prepare_metadata_for_padding_backward(
    input_shape: &[usize],
    input_strides: &[usize],
    input_offset: usize,
    paddings: &[(usize, usize)],
) -> Vec<usize> {
    let mut info = Vec::new();

    // Input dimensions (original tensor shape)
    info.extend_from_slice(input_shape);
    // Input strides
    info.extend_from_slice(input_strides);
    // Input offset
    info.push(input_offset);

    // Output dimensions (padded tensor shape)
    for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
        info.push(input_shape[i] + pad_before + pad_after);
    }

    // Padding values (before, after for each dimension)
    for &(pad_before, pad_after) in paddings {
        info.push(pad_before);
        info.push(pad_after);
    }

    info
}

