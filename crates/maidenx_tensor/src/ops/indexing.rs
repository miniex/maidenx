use crate::Tensor;
use maidenx_core::{
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
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
