use crate::{utils::broadcast::pad_shape, Tensor, TensorNode};
use maidenx_core::error::{Error, Result};

impl Tensor {
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
            self.buffer()
                .copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, temp_buf.len(), self.offset(), 0)?;
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
                buf.copy_from_host(result_buf.as_ptr() as *const std::ffi::c_void, result_buf.len(), 0, 0)?;
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

    pub fn broadcast_like(&self, other: &Self) -> Result<Self> {
        self.broadcast(other.shape())
    }

    pub fn broadcast_left(&self, batch_dims: &[usize]) -> Result<Self> {
        let mut new_shape = batch_dims.as_ref().to_vec();
        new_shape.extend(self.shape());
        self.broadcast(&new_shape)
    }

    // ==== helper ====

    fn broadcast_scalar_to(&self, shape: &[usize]) -> Result<Self> {
        let mut result = Self::empty_with_spec(shape, self.device(), self.dtype())?;
        let mut scalar_buf = vec![0u8; self.dtype().size_in_bytes()];

        unsafe {
            self.buffer()
                .copy_to_host(scalar_buf.as_mut_ptr() as *mut std::ffi::c_void, scalar_buf.len(), self.offset(), 0)?;

            let mut result_buf = vec![0u8; result.size() * result.dtype().size_in_bytes()];
            for i in 0..result.size() {
                let offset = i * result.dtype().size_in_bytes();
                result_buf[offset..offset + scalar_buf.len()].copy_from_slice(&scalar_buf);
            }

            result.with_buffer_mut(|buf| {
                buf.copy_from_host(result_buf.as_ptr() as *const std::ffi::c_void, result_buf.len(), 0, 0)?;

                Ok(())
            })?;
        }

        Ok(result)
    }
}
