use crate::{utils::indexing::set_index, Tensor};
use maidenx_core::{
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
    #[allow(clippy::needless_range_loop)]
    pub fn index_put_(&mut self, indices: &[usize], src: &Tensor) -> Result<()> {
        // Validate that the indices specify a valid position in the selfination tensor
        if indices.len() > self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "Indices length ({}) exceeds selfination tensor dimensions ({})",
                indices.len(),
                self.ndim()
            )));
        }

        // Check if we are indexing a subset of dimensions
        let remaining_dims = self.ndim() - indices.len();

        // If we're not indexing all dimensions, the source tensor should match
        // the remaining dimensions of the selfination
        if remaining_dims > 0 {
            // Get the shape of the remaining dimensions in the selfination tensor
            let mut self_remaining_shape = Vec::with_capacity(remaining_dims);
            for i in indices.len()..self.ndim() {
                self_remaining_shape.push(self.dim_size(i).unwrap_or(0));
            }

            // Check if source tensor shape matches the remaining dimensions
            if src.ndim() != remaining_dims {
                return Err(Error::InvalidArgument(format!(
                    "Source tensor dimensions ({}) don't match selfination's remaining dimensions ({})",
                    src.ndim(),
                    remaining_dims
                )));
            }

            for i in 0..remaining_dims {
                if src.dim_size(i).unwrap_or(0) != self_remaining_shape[i] {
                    return Err(Error::InvalidArgument(format!(
                        "Dimension mismatch at index {}: source has size {}, selfination expects {}",
                        i,
                        src.dim_size(i).unwrap_or(0),
                        self_remaining_shape[i]
                    )));
                }
            }

            // Calculate the base offset in the selfination tensor
            let mut base_offset = 0;
            for (dim, &idx) in indices.iter().enumerate() {
                let dim_size = self.dim_size(dim).unwrap_or(0);
                if idx >= dim_size {
                    return Err(Error::IndexOutOfBounds { index: idx, size: dim_size });
                }
                base_offset += idx * self.strides()[dim];
            }

            fn copy_elements(
                src: &Tensor,
                dest: &mut Tensor,
                base_offset: usize,
                base_dims: usize,
                curr_indices: &mut Vec<usize>,
                curr_dim: usize,
            ) -> Result<()> {
                if curr_dim == src.ndim() {
                    let mut dest_flat_idx = base_offset;
                    for dim in 0..curr_indices.len() {
                        dest_flat_idx += curr_indices[dim] * dest.strides()[base_dims + dim];
                    }

                    // Calculate the source flat index
                    let mut src_flat_idx = src.offset();
                    for dim in 0..curr_indices.len() {
                        src_flat_idx += curr_indices[dim] * src.strides()[dim];
                    }

                    // Read from source and write to destination
                    let value = src.buffer().read_scalar(src_flat_idx)?;
                    dest.with_buffer_mut(|buffer| buffer.write_scalar(dest_flat_idx, value))?;

                    return Ok(());
                }

                // Recursively iterate through all possible values for the current dimension
                let dim_size = src.dim_size(curr_dim).unwrap_or(0);
                for i in 0..dim_size {
                    curr_indices.push(i);
                    copy_elements(src, dest, base_offset, base_dims, curr_indices, curr_dim + 1)?;
                    curr_indices.pop();
                }

                Ok(())
            }

            let mut curr_indices = Vec::new();
            copy_elements(src, self, base_offset + self.offset(), indices.len(), &mut curr_indices, 0)?;
        } else {
            // If we're indexing all dimensions (or the selfination is 1D)
            // We can simply set the scalar value from the source
            if src.ndim() == 0 {
                // Source is a scalar
                let scalar = src.buffer().read_scalar(src.offset())?;
                set_index(self, indices, scalar)?;
            } else if src.size() > 0 {
                // Source is a 1D tensor or has flattened elements that we need to copy over
                let self_dim = indices[0];
                if self_dim + src.size() > self.size() {
                    return Err(Error::InvalidArgument(format!(
                        "Source tensor size ({}) exceeds available space in selfination starting at index {} (available: {})",
                        src.size(),
                        self_dim,
                        self.size() - self_dim
                    )));
                }

                // Copy each element from source to the selfination
                for i in 0..src.size() {
                    let value = src.buffer().read_scalar(src.offset() + i * src.strides()[0])?;
                    let self_idx = vec![self_dim + i]; // For a 1D tensor
                    set_index(self, &self_idx, value)?;
                }
            }
        }

        Ok(())
    }

    pub fn gather(&self, dim: impl Into<Scalar>, index: &Tensor) -> Result<Tensor> {
        let dim_i32 = dim.into().as_i32();
        let dim_usize: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim_usize >= self.ndim() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim_usize as i32,
                ndim: self.ndim(),
            });
        }

        for d in 0..self.ndim() {
            if d != dim_usize && self.shape()[d] != index.shape()[d] {
                return Err(Error::InvalidArgument(format!(
                    "Index tensor must have the same shape as self in all dimensions except for dimension {}, but got {} and {}",
                    dim_usize,
                    self.shape()[d],
                    index.shape()[d]
                )));
            }
        }

        let mut output = Tensor::zeros_like(index)?;

        let mut index_vec = vec![0; self.ndim()];
        let mut output_vec = vec![0; output.ndim()];

        let index_iter = output.index_iter()?;
        for output_indices in index_iter {
            output_indices.iter().enumerate().for_each(|(d, &idx)| {
                output_vec[d] = idx;
                index_vec[d] = idx;
            });

            let index_value = index.get(&output_indices)?.as_i32();

            let dim_size = self.shape()[dim_usize] as i32;
            let actual_index = if index_value < 0 { dim_size + index_value } else { index_value };

            if actual_index < 0 || actual_index >= dim_size {
                return Err(Error::IndexOutOfBounds {
                    index: actual_index as usize,
                    size: dim_size as usize,
                });
            }

            index_vec[dim_usize] = actual_index as usize;

            let value = self.get(&index_vec)?;
            output.set(&output_vec, value)?;
        }

        if self.requires_grad() {
            output.with_grad()?;

            let self_clone = self.clone();
            let index_clone = index.clone();
            let dim_clone = dim_i32;

            let backward_fn = Box::new(move |_grads: &[Tensor], grad_output: &Tensor| -> Result<Vec<Tensor>> {
                let mut grad_self = Tensor::zeros_like(&self_clone)?;
                grad_self.scatter_add_(dim_clone, &index_clone, grad_output)?;

                let grad_index = Tensor::zeros_like(&index_clone)?;

                Ok(vec![grad_self, grad_index])
            });

            let node = crate::TensorNode::new("gather".to_string(), vec![self.clone(), index.clone()], Some(backward_fn));

            output.set_node(node);
        }

        Ok(output)
    }

    pub fn scatter_add_(&mut self, dim: impl Into<Scalar>, index: &Tensor, src: &Tensor) -> Result<()> {
        let dim_i32 = dim.into().as_i32();
        let dim_usize: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim_usize >= self.ndim() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim_usize as i32,
                ndim: self.ndim(),
            });
        }

        for d in 0..self.ndim() {
            if d != dim_usize && index.shape()[d] != src.shape()[d] {
                return Err(Error::InvalidArgument(format!(
                    "Index and source tensors must have the same shape, but got {:?} and {:?}",
                    index.shape(),
                    src.shape()
                )));
            }
        }

        if src.shape() != index.shape() {
            return Err(Error::InvalidArgument(format!(
                "Source and index tensors must have the same shape, but got {:?} and {:?}",
                src.shape(),
                index.shape()
            )));
        }

        let src_iter = src.index_iter()?;
        for src_indices in src_iter {
            let value = src.get(&src_indices)?;
            let target_index = index.get(&src_indices)?.as_i32();
            let dim_size = self.shape()[dim_usize] as i32;
            let actual_index = if target_index < 0 { dim_size + target_index } else { target_index };

            if actual_index < 0 || actual_index >= dim_size {
                return Err(Error::IndexOutOfBounds {
                    index: actual_index as usize,
                    size: dim_size as usize,
                });
            }

            let mut target_indices = src_indices.clone();
            target_indices[dim_usize] = actual_index as usize;

            crate::utils::indexing::add_at_index(self, &target_indices, value)?;
        }

        Ok(())
    }
}
