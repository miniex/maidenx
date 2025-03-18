use crate::{utils::promotion::promote_tensor, Tensor, TensorNode};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
    pub fn sum(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let mut result = Self::zeros_with_spec(&shape, self.device(), self.dtype())?;

        let metadata = prepare_metadata(self, dim);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::reduction::sum(out_buf, self.buffer(), self.size(), self.ndim(), 1, Some(&metadata))?;

                Ok(())
            })?;
        }

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result = result.view(&keep_dim_shape)?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_out = grad_out.clone();

                if keep_dim {
                    Ok(vec![grad_out.broadcast(&input_shape)?])
                } else {
                    let mut grad_shape = input_shape.clone();
                    grad_shape[dim] = 1;
                    let viewed_grad = grad_out.view(&grad_shape)?;
                    Ok(vec![viewed_grad.broadcast(&input_shape)?])
                }
            });

            let node = TensorNode::new("sum".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn sum_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.sum(dim, false)?;
        }

        Ok(result)
    }

    pub fn sum_to_shape(&self, shape: &[usize]) -> Result<Tensor> {
        if shape.len() != self.ndim() {
            if self.ndim() > shape.len() {
                let mut result = self.clone();
                while result.ndim() > shape.len() {
                    result = result.sum(0, false)?;
                }
                return result.sum_to_shape(shape);
            } else {
                return Err(Error::ShapeMismatch {
                    expected: self.ndim(),
                    got: shape.len(),
                    msg: "Target shape has more dimensions than input".to_string(),
                });
            }
        }

        for (i, &dim) in shape.iter().enumerate() {
            if self.shape()[i] % dim != 0 {
                return Err(Error::ShapeMismatch {
                    expected: self.shape()[i],
                    got: dim,
                    msg: format!("Dimension {} is not divisible: {} -> {}", i, self.shape()[i], dim),
                });
            }
        }

        let mut result = Self::zeros_with_spec(shape, self.device(), self.dtype())?;

        let metadata = prepare_metadata_for_shape(self, shape);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::reduction::sum_to_shape(out_buf, self.buffer(), self.size(), self.ndim(), Some(&metadata))?;

                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_shape = self.shape().to_vec();
            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.broadcast(&input_shape)?]) });

            let node = TensorNode::new("sum_to_shape".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn mean(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let target_dtype = if self.dtype().is_int() { DType::F32 } else { self.dtype() };
        let input = promote_tensor(self, target_dtype)?;

        let mut result = Self::zeros_with_spec(&shape, input.device(), input.dtype())?;

        let metadata = prepare_metadata(self, dim);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::reduction::mean(out_buf, input.buffer(), input.size(), input.ndim(), 1, Some(&metadata))?;

                Ok(())
            })?;
        }

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result = result.view(&keep_dim_shape)?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_shape = self.shape().to_vec();
            let dim_size = self.shape()[dim];
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let mut grad_out = grad_out.clone();

                if !keep_dim {
                    let mut grad_shape = input_shape.clone();
                    grad_shape[dim] = 1;
                    grad_out = grad_out.view(&grad_shape)?;
                }

                let broadcasted_grad = grad_out.broadcast(&input_shape)?;

                Ok(vec![broadcasted_grad.div_scalar(dim_size)?])
            });

            let node = TensorNode::new("sum".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn mean_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.mean(dim, false)?;
        }

        Ok(result)
    }

    pub fn fold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let size_i32 = size.into().as_i32();
        let step_i32 = step.into().as_i32();

        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim >= self.ndim() - 1 {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let window_dim = dim + 1;
        let window_size = self.shape()[window_dim];
        if size_i32 as usize != 0 && size_i32 as usize != window_size {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch: window size is {}, but requested size is {}",
                window_size, size_i32
            )));
        }

        let n_windows = self.shape()[dim];
        let orig_dim_size = (n_windows - 1) * (step_i32 as usize) + window_size;

        let mut output_shape = Vec::with_capacity(self.ndim() - 1);
        for d in 0..self.ndim() {
            if d == dim {
                output_shape.push(orig_dim_size);
            } else if d == window_dim {
                continue;
            } else {
                output_shape.push(self.shape()[d]);
            }
        }

        let mut result = Self::zeros_with_spec(&output_shape, self.device(), self.dtype())?;

        let metadata = prepare_metadata_for_fold(self, dim, window_dim, orig_dim_size, step_i32 as usize, window_size);

        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::reduction::fold(out_buf, self.buffer(), self.size(), self.ndim(), Some(&metadata))?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_shape = self.shape().to_vec();
            let orig_dim = dim;
            let orig_size = size_i32 as usize;
            let orig_step = step_i32 as usize;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_in = grad_out.unfold(orig_dim, orig_size, orig_step)?;

                if grad_in.shape() != input_shape {
                    let mut adj_grad = Tensor::zeros_with_spec(&input_shape, grad_in.device(), grad_in.dtype())?;

                    for indices in grad_in.index_iter()? {
                        let value = grad_in.get(&indices)?;
                        adj_grad.set(&indices, value)?;
                    }

                    Ok(vec![adj_grad])
                } else {
                    Ok(vec![grad_in])
                }
            });

            let node = TensorNode::new("fold".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn max(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let mut result = Self::empty_with_spec(&shape, self.device(), self.dtype())?;
        let metadata = prepare_metadata(self, dim);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::reduction::max(out_buf, self.buffer(), self.size(), self.ndim(), 1, Some(&metadata))?;

                Ok(())
            })?;
        }

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result = result.view(&keep_dim_shape)?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_shape = self.shape().to_vec();
            let input_clone = self.clone();
            let result_clone = result.clone();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let input_ndim = input_clone.ndim();
                let grad_ndim = grad_out.ndim();

                let max_values = if keep_dim {
                    result_clone.broadcast(&input_shape)?
                } else if grad_ndim == input_ndim - 1 {
                    let mut expanded_shape = vec![0; input_ndim];
                    let mut result_idx = 0;
                    (0..input_ndim).for_each(|i| {
                        if i == dim {
                            expanded_shape[i] = 1;
                        } else {
                            expanded_shape[i] = result_clone.shape()[result_idx];
                            result_idx += 1;
                        }
                    });

                    result_clone.view(&expanded_shape)?.broadcast(&input_shape)?
                } else {
                    let broadcast_shape = input_shape.clone();
                    result_clone.broadcast(&broadcast_shape)?
                };

                let mut mask = input_clone.eq(&max_values)?;
                mask.with_dtype(grad_out.dtype())?;

                let count = mask.sum(dim, keep_dim)?;
                let safe_count = count.maximum_scalar(1.0)?;

                let grad_expanded = if keep_dim {
                    grad_out.broadcast(&input_shape)?
                } else if grad_ndim == input_ndim - 1 {
                    let mut expanded_shape = vec![0; input_ndim];

                    let mut grad_idx = 0;
                    (0..input_ndim).for_each(|i| {
                        if i == dim {
                            expanded_shape[i] = 1;
                        } else {
                            expanded_shape[i] = grad_out.shape()[grad_idx];
                            grad_idx += 1;
                        }
                    });

                    grad_out.view(&expanded_shape)?.broadcast(&input_shape)?
                } else {
                    let broadcast_shape = input_shape.clone();
                    grad_out.broadcast(&broadcast_shape)?
                };

                let count_expanded = if keep_dim {
                    safe_count.broadcast(&input_shape)?
                } else if grad_ndim == input_ndim - 1 {
                    let mut expanded_shape = vec![0; input_ndim];

                    let mut count_idx = 0;
                    (0..input_ndim).for_each(|i| {
                        if i == dim {
                            expanded_shape[i] = 1;
                        } else {
                            expanded_shape[i] = safe_count.shape()[count_idx];
                            count_idx += 1;
                        }
                    });

                    safe_count.view(&expanded_shape)?.broadcast(&input_shape)?
                } else {
                    let broadcast_shape = input_shape.clone();
                    safe_count.broadcast(&broadcast_shape)?
                };

                let distributed_grad = grad_expanded.div(&count_expanded)?;
                let grad_input = mask.mul(&distributed_grad)?;

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("max".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn max_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.max(dim, false)?;
        }

        Ok(result)
    }

    pub fn min(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let original_shape = shape.clone();
        shape.remove(dim);

        let mut result = Self::empty_with_spec(&shape, self.device(), self.dtype())?;
        let metadata = prepare_metadata(self, dim);
        unsafe {
            result.with_buffer_mut(|out_buf| {
                maidenx_core::be::ops::reduction::min(out_buf, self.buffer(), self.size(), self.ndim(), 1, Some(&metadata))?;

                Ok(())
            })?;
        }

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result = result.view(&keep_dim_shape)?;
        }

        if self.requires_grad() {
            result.with_grad()?;

            let input_shape = self.shape().to_vec();
            let input_clone = self.clone();
            let result_clone = result.clone();

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let input_ndim = input_clone.ndim();
                let grad_ndim = grad_out.ndim();

                let min_values = if keep_dim {
                    result_clone.broadcast(&input_shape)?
                } else if grad_ndim == input_ndim - 1 {
                    let mut expanded_shape = vec![0; input_ndim];
                    let mut result_idx = 0;
                    (0..input_ndim).for_each(|i| {
                        if i == dim {
                            expanded_shape[i] = 1;
                        } else {
                            expanded_shape[i] = result_clone.shape()[result_idx];
                            result_idx += 1;
                        }
                    });
                    result_clone.view(&expanded_shape)?.broadcast(&input_shape)?
                } else {
                    result_clone.broadcast(&input_shape)?
                };

                let mut mask = input_clone.eq(&min_values)?;
                mask.with_dtype(grad_out.dtype())?;

                let count = mask.sum(dim, keep_dim)?;
                let safe_count = count.maximum_scalar(1.0)?;

                let grad_expanded = if keep_dim {
                    grad_out.broadcast(&input_shape)?
                } else if grad_ndim == input_ndim - 1 {
                    let mut expanded_shape = vec![0; input_ndim];
                    let mut grad_idx = 0;
                    (0..input_ndim).for_each(|i| {
                        if i == dim {
                            expanded_shape[i] = 1;
                        } else {
                            expanded_shape[i] = grad_out.shape()[grad_idx];
                            grad_idx += 1;
                        }
                    });
                    grad_out.view(&expanded_shape)?.broadcast(&input_shape)?
                } else {
                    grad_out.broadcast(&input_shape)?
                };

                let count_expanded = if keep_dim {
                    safe_count.broadcast(&input_shape)?
                } else if grad_ndim == input_ndim - 1 {
                    let mut expanded_shape = vec![0; input_ndim];
                    let mut count_idx = 0;
                    (0..input_ndim).for_each(|i| {
                        if i == dim {
                            expanded_shape[i] = 1;
                        } else {
                            expanded_shape[i] = safe_count.shape()[count_idx];
                            count_idx += 1;
                        }
                    });
                    safe_count.view(&expanded_shape)?.broadcast(&input_shape)?
                } else {
                    safe_count.broadcast(&input_shape)?
                };

                let distributed_grad = grad_expanded.div(&count_expanded)?;
                let grad_input = mask.mul(&distributed_grad)?;

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("min".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn min_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.min(dim, false)?;
        }

        Ok(result)
    }
}

fn prepare_metadata(tensor: &Tensor, dim: usize) -> Vec<usize> {
    let mut info = Vec::new();
    let shape = tensor.shape();
    let strides = tensor.strides();
    info.extend_from_slice(shape);
    info.extend_from_slice(strides);
    info.push(shape[dim]);
    info.push(strides[dim]);
    info.push(tensor.offset());
    info
}

fn prepare_metadata_for_shape(tensor: &Tensor, target_shape: &[usize]) -> Vec<usize> {
    let mut info = Vec::new();
    let input_shape = tensor.shape();
    let input_strides = tensor.strides();
    info.extend_from_slice(input_shape);
    info.extend_from_slice(input_strides);
    info.extend_from_slice(target_shape);
    info.push(tensor.offset());
    info
}

fn prepare_metadata_for_fold(tensor: &Tensor, fold_dim: usize, window_dim: usize, fold_size: usize, step: usize, window_size: usize) -> Vec<usize> {
    let mut info = Vec::new();
    let input_shape = tensor.shape();
    let input_strides = tensor.strides();

    info.extend_from_slice(input_shape);
    info.extend_from_slice(input_strides);

    info.push(fold_dim);
    info.push(window_dim);
    info.push(fold_size);
    info.push(step);
    info.push(window_size);

    info.push(tensor.offset());

    info
}
