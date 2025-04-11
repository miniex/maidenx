use crate::{Tensor, TensorNode};
use maidenx_core::{
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
    // ==== view ops ====

    pub fn view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let mut result = Self::share_buffer(self)?;
        result.layout_mut().view(&computed_shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.reshape(&orig_shape)?])
            });

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

        if self.dim_size(dim) != Some(1) {
            return Ok(self.clone());
        }

        let mut shape: Vec<usize> = self.shape().to_vec();
        shape.remove(dim);
        if shape.is_empty() {
            shape.push(1);
        }

        let mut result = Self::share_buffer(self)?;
        result.layout_mut().view(&shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.reshape(&orig_shape)?])
            });

            let node = TensorNode::new("squeeze".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }
        Ok(result)
    }

    pub fn squeeze_all(&self) -> Result<Self> {
        let shape: Vec<usize> = self.shape().iter().filter(|&&dim| dim != 1).cloned().collect();

        let mut result = Self::share_buffer(self)?;
        result.layout_mut().view(&shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.reshape(&orig_shape)?])
            });

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

        let mut result = Self::share_buffer(self)?;
        result.layout_mut().view(&shape)?;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.reshape(&orig_shape)?])
            });

            let node = TensorNode::new("unsqueeze".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    // ==== layout ops ====

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

        let mut result = Self::share_buffer(self)?;
        result.layout_mut().transpose(dim0, dim1)?;

        if self.requires_grad() {
            result.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.transpose(dim0, dim1)?])
            });

            let node = TensorNode::new("transpose".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    pub fn slice(
        &self,
        dim: impl Into<Scalar>,
        start: impl Into<Scalar>,
        end: Option<impl Into<Scalar>>,
        step: impl Into<Scalar>,
    ) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let start_i32 = start.into().as_i32();
        let end_i32 = end.map(|e| e.into().as_i32());
        let step_i32 = step.into().as_i32();

        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let new_layout =
            self.layout()
                .slice(dim, start_i32 as isize, end_i32.map(|e| e as isize), step_i32 as isize)?;
        let mut result = Self::share_buffer(self)?;
        *result.layout_mut() = new_layout;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let orig_dim = dim;
            let orig_start = start_i32;
            let orig_end = end_i32;
            let orig_step = step_i32;

            let backward_fn = Box::new(move |inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let input = &inputs[0];
                let mut grad_input = Tensor::zeros_with_spec(&orig_shape, input.device(), input.dtype())?;

                let actual_end = orig_end.unwrap_or(input.dim_size(orig_dim).unwrap_or(0) as i32);

                let mut i = 0;
                for idx in (orig_start..actual_end).step_by(orig_step as usize) {
                    if idx < 0 || idx >= input.dim_size(orig_dim).unwrap_or(0) as i32 {
                        continue;
                    }

                    let idx_usize = idx as usize;
                    let grad_slice = grad_out.slice(orig_dim, i, Some(i + 1), 1)?;

                    for idx_tuple in grad_slice.index_iter()? {
                        let mut indices = idx_tuple.to_vec();
                        indices[orig_dim] = idx_usize;

                        let value = grad_slice.get(&idx_tuple)?;
                        crate::utils::indexing::add_at_index(&mut grad_input, &indices, value)?;
                    }

                    i += 1;
                }

                Ok(vec![grad_input])
            });

            let node = TensorNode::new("slice".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }
        Ok(result)
    }

    pub fn unfold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let size_i32 = size.into().as_i32();
        let step_i32 = step.into().as_i32();

        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let new_layout = self.layout().unfold(dim, size_i32 as usize, step_i32 as usize)?;
        let mut result = Self::share_buffer(self)?;
        *result.layout_mut() = new_layout;

        if self.requires_grad() {
            result.with_grad()?;

            let orig_shape = self.shape().to_vec();
            let orig_dim = dim;
            let orig_size = size_i32 as usize;
            let orig_step = step_i32 as usize;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                let grad_input = grad_out.fold(orig_dim, orig_size, orig_step)?;

                if grad_input.shape() != orig_shape {
                    return Ok(vec![grad_input.reshape(&orig_shape)?]);
                }

                Ok(vec![grad_input])
            });
            let node = TensorNode::new("unfold".to_string(), vec![self.clone()], Some(backward_fn));
            result.node = Some(node);
        }

        Ok(result)
    }

    // ==== broadcast ops ====

    pub fn broadcast(&self, shape: &[usize]) -> Result<Self> {
        if self.shape() == shape {
            return Ok(self.clone());
        }

        let old_shape = self.shape();

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

        let new_layout = self.layout().broadcast_to(shape)?;
        let mut result = Self::share_buffer(self)?;
        *result.layout_mut() = new_layout;

        if self.requires_grad() {
            result.with_grad()?;
            let orig_shape = self.shape().to_vec();
            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.sum_to_shape(&orig_shape)?])
            });
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

    fn broadcast_scalar_to(&self, shape: &[usize]) -> Result<Self> {
        let scalar_value = self.item()?;
        let result = Self::fill_with_spec(shape, scalar_value, self.device(), self.dtype())?;

        if self.requires_grad() {
            let mut result_with_grad = result;
            result_with_grad.with_grad()?;

            let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                Ok(vec![grad_out.sum_all()?])
            });

            let node = TensorNode::new("broadcast_scalar".to_string(), vec![self.clone()], Some(backward_fn));
            result_with_grad.node = Some(node);

            Ok(result_with_grad)
        } else {
            Ok(result)
        }
    }

    // ==== reshape ops ====

    pub fn reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;

        if self.is_contiguous() {
            let mut result = Self::share_buffer(self)?;
            result.layout_mut().view(&computed_shape)?;

            if self.requires_grad() {
                result.with_grad()?;

                let orig_shape = self.shape().to_vec();
                let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                    Ok(vec![grad_out.reshape(&orig_shape)?])
                });

                let node = TensorNode::new("reshape".to_string(), vec![self.clone()], Some(backward_fn));
                result.node = Some(node);
            }

            Ok(result)
        } else {
            let mut result = self.contiguous()?;
            result.layout_mut().view(&computed_shape)?;

            if self.requires_grad() {
                let orig_shape = self.shape().to_vec();
                let backward_fn = Box::new(move |_inputs: &[Tensor], grad_out: &Tensor| -> Result<Vec<Tensor>> {
                    Ok(vec![grad_out.reshape(&orig_shape)?])
                });

                let node = TensorNode::new("reshape".to_string(), vec![self.clone()], Some(backward_fn));
                result.node = Some(node);
            }

            Ok(result)
        }
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
}
