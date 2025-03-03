use crate::{utils::broadcast::pad_shape, Tensor, TensorNode};
use maidenx_core::{
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
    // ==== view ops ====

    pub fn view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let mut result = Self::from_tensor(self)?;
        result.layout_mut().view(&computed_shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.view(&orig_shape)?]) });

            let node = TensorNode::new("view".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn squeeze(&self, dim: impl Into<Scalar>) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();

        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim >= self.ndim() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        if self.size_dim(dim) != Some(1) {
            return Ok(self.clone());
        }

        let mut shape: Vec<usize> = self.shape().to_vec();
        shape.remove(dim);
        if shape.is_empty() {
            shape.push(1);
        }

        let mut result = Self::from_tensor(self)?;
        result.layout_mut().view(&shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.view(&orig_shape)?]) });

            let node = TensorNode::new("squeeze".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }
        Ok(result)
    }

    pub fn squeeze_all(&self) -> Result<Self> {
        let shape: Vec<usize> = self.shape().iter().filter(|&&dim| dim != 1).cloned().collect();

        let mut result = Self::from_tensor(self)?;
        result.layout_mut().view(&shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.view(&orig_shape)?]) });

            let node = TensorNode::new("squeeze_all".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn unsqueeze(&self, dim: impl Into<Scalar>) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();

        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let mut shape = self.shape().to_vec();
        if dim > shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }
        shape.insert(dim, 1);

        let mut result = Self::from_tensor(self)?;
        result.layout_mut().view(&shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.view(&orig_shape)?]) });

            let node = TensorNode::new("unsqueeze".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn transpose(&self, dim0: impl Into<Scalar>, dim1: impl Into<Scalar>) -> Result<Self> {
        let dim0_i32 = dim0.into().as_i32();
        let dim1_i32 = dim1.into().as_i32();

        let dim0: usize = if dim0_i32 < 0 {
            (self.ndim() as i32 + dim0_i32) as usize
        } else {
            dim0_i32 as usize
        };
        let dim1: usize = if dim1_i32 < 0 {
            (self.ndim() as i32 + dim1_i32) as usize
        } else {
            dim1_i32 as usize
        };

        let mut result = Self::from_tensor(self)?;
        result.layout_mut().transpose(dim0, dim1)?;

        if self.requires_grad() {
            result.with_grad()?;

            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.transpose(dim0, dim1)?]) });

            let node = TensorNode::new("transpose".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    // ==== reshape ops ====

    pub fn reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let mut result = Self::from_tensor(self)?;
        result.layout_mut().view(&computed_shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.view(&orig_shape)?]) });

            let node = TensorNode::new("reshape".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        result.contiguous()?;

        Ok(result)
    }

    pub fn broadcast(&self, shape: &[usize]) -> Result<Self> {
        if self.shape() == shape {
            return Ok(self.clone());
        }

        let old_shape = self.shape();
        let rank = shape.len();

        if old_shape.is_empty() {
            return self.broadcast_scalar_to(shape);
        }

        if shape.is_empty() {
            if !old_shape.is_empty() {
                return Err(Error::InvalidShape {
                    message: format!("Cannot broadcast non-scalar shape {:?} to scalar ()", old_shape),
                });
            } else {
                return Ok(self.clone());
            }
        }

        let padded_old = pad_shape(old_shape, rank);

        // Validate broadcast compatibility
        for i in 0..rank {
            let dim_old = padded_old[i];
            let dim_new = shape[i];
            if dim_old != 1 && dim_old != dim_new {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Cannot broadcast dimension {} -> {} (shape {:?} -> {:?})",
                        dim_old, dim_new, old_shape, shape
                    ),
                });
            }
        }

        let mut result = Self::empty_with_spec(shape, self.device(), self.dtype())?;

        let mut temp_buf = vec![0u8; self.size() * self.dtype().size_in_bytes()];
        unsafe {
            self.buffer()?
                .copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, temp_buf.len())?;
        }

        let mut result_buf = vec![0u8; result.size() * result.dtype().size_in_bytes()];

        let src_strides = self.strides();
        let padded_src_strides = pad_shape(src_strides, rank);
        let padded_src_shape = padded_old;

        let mut effective_strides = padded_src_strides.clone();
        for i in 0..rank {
            if padded_src_shape[i] == 1 {
                effective_strides[i] = 0;
            }
        }

        let mut pos = vec![0; rank];
        for i in 0..result.size() {
            let mut remainder = i;
            for d in (0..rank).rev() {
                pos[d] = remainder % shape[d];
                remainder /= shape[d];
            }

            let mut src_idx = 0;
            for d in 0..rank {
                src_idx += pos[d] * effective_strides[d];
            }

            let src_offset = src_idx * self.dtype().size_in_bytes();
            let dst_offset = i * self.dtype().size_in_bytes();
            result_buf[dst_offset..dst_offset + self.dtype().size_in_bytes()]
                .copy_from_slice(&temp_buf[src_offset..src_offset + self.dtype().size_in_bytes()]);
        }

        unsafe {
            result.with_buffer_mut(|buf| {
                buf.copy_from_host(result_buf.as_ptr() as *const std::ffi::c_void, result_buf.len())?;
                Ok(())
            })?;
        }

        if self.requires_grad() {
            result.with_grad()?;
            let orig_shape = self.shape().to_vec();
            let backward_fn =
                Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> { Ok(vec![grad_out.sum_to_shape(&orig_shape)?]) });
            let node = TensorNode::new("broadcast".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn broadcast_left(&self, batch_dims: &[usize]) -> Result<Self> {
        let mut new_shape = batch_dims.as_ref().to_vec();
        new_shape.extend(self.shape());
        self.broadcast(&new_shape)
    }

    // ==== helper ====

    fn compute_shape_with_auto<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Vec<usize>> {
        let total_elements = self.size();
        let mut product: i64 = 1;
        let mut auto_dim_idx = None;

        let shape_i64: Vec<i64> = shape.iter().map(|x| x.clone().into().as_i32() as i64).collect();

        for (i, &dim) in shape_i64.iter().enumerate() {
            if dim == -1 {
                if auto_dim_idx.is_some() {
                    return Err(Error::InvalidShape {
                        message: "Only one dimension can be -1".to_string(),
                    });
                }
                auto_dim_idx = Some(i);
            } else if dim <= 0 && dim != -1 {
                return Err(Error::InvalidShape {
                    message: format!("Invalid dimension size: {}", dim),
                });
            } else {
                product *= dim;
            }
        }

        let mut result: Vec<usize> = Vec::with_capacity(shape.len());
        if let Some(idx) = auto_dim_idx {
            if product == 0 {
                return Err(Error::InvalidShape {
                    message: "Cannot infer size for dimension -1".to_string(),
                });
            }

            let auto_dim = total_elements as i64 / product;
            if auto_dim * product != total_elements as i64 {
                return Err(Error::InvalidShape {
                    message: "Cannot reshape tensor: incompatible dimensions".to_string(),
                });
            }

            for (i, &dim) in shape_i64.iter().enumerate() {
                if i == idx {
                    result.push(auto_dim as usize);
                } else {
                    result.push(dim as usize);
                }
            }
        } else {
            if product != total_elements as i64 {
                return Err(Error::InvalidShape {
                    message: "Total size of new shape must be equal to old size".to_string(),
                });
            }
            result = shape_i64.iter().map(|&x| x as usize).collect();
        }

        Ok(result)
    }

    fn broadcast_scalar_to(&self, shape: &[usize]) -> Result<Self> {
        let result = Self::empty_with_spec(shape, self.device(), self.dtype())?;
        let mut scalar_buf = vec![0u8; self.dtype().size_in_bytes()];

        unsafe {
            self.buffer()?
                .copy_to_host(scalar_buf.as_mut_ptr() as *mut std::ffi::c_void, scalar_buf.len())?;

            let mut result_buf = vec![0u8; result.size() * result.dtype().size_in_bytes()];
            for i in 0..result.size() {
                let offset = i * result.dtype().size_in_bytes();
                result_buf[offset..offset + scalar_buf.len()].copy_from_slice(&scalar_buf);
            }

            result.with_buffer_mut(|buf| {
                buf.copy_from_host(result_buf.as_ptr() as *const std::ffi::c_void, result_buf.len())?;

                Ok(())
            })?;
        }

        Ok(result)
    }
}
