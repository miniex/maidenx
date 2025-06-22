use crate::{
    get_mode, insert_metadata, link_tensor_to_storage, next_tensor_id, utils::graph::add_to_graph, Tensor,
    TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};

impl Tensor {
    pub fn try_view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let mut new_layout = self.layout();
        new_layout.view(&computed_shape)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("view", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_squeeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Result<Self> {
        let current_shape = self.shape();
        let rank = current_shape.len();

        let mut normalized_dims = Vec::new();
        for dim in dims {
            let dim_i32 = dim.clone().into().as_i32();
            let normalized_dim: usize = if dim_i32 < 0 {
                (rank as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };

            if normalized_dim >= rank {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Dimension {} is out of bounds for tensor with {} dimensions",
                        normalized_dim, rank
                    ),
                });
            }

            if current_shape[normalized_dim] != 1 {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Cannot squeeze dimension {} with size {}",
                        normalized_dim, current_shape[normalized_dim]
                    ),
                });
            }

            normalized_dims.push(normalized_dim);
        }

        let mut new_shape = Vec::new();
        normalized_dims.sort_unstable();
        normalized_dims.dedup();

        for (i, &size) in current_shape.iter().enumerate() {
            if !normalized_dims.contains(&i) {
                new_shape.push(size);
            }
        }

        let mut new_layout = self.layout();
        new_layout.view(&new_shape)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("squeeze", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_squeeze_all(&self) -> Result<Self> {
        let current_shape = self.shape();
        let mut new_shape = Vec::new();

        for &size in &current_shape {
            if size != 1 {
                new_shape.push(size);
            }
        }

        if new_shape.len() == current_shape.len() {
            return self.try_view(&current_shape);
        }

        let mut new_layout = self.layout();
        new_layout.view(&new_shape)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("squeeze_all", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_unsqueeze<T: Into<Scalar> + Clone>(&self, dims: &[T]) -> Result<Self> {
        let current_shape = self.shape();
        let target_rank = current_shape.len() + dims.len();

        let mut normalized_dims = Vec::new();
        for dim in dims {
            let dim_i32 = dim.clone().into().as_i32();
            let normalized_dim: usize = if dim_i32 < 0 {
                (target_rank as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };

            if normalized_dim >= target_rank {
                return Err(Error::InvalidShape {
                    message: format!(
                        "Dimension {} is out of bounds for target tensor with {} dimensions",
                        normalized_dim, target_rank
                    ),
                });
            }

            normalized_dims.push(normalized_dim);
        }

        let mut new_shape = vec![1; target_rank];
        normalized_dims.sort_unstable();
        normalized_dims.dedup();

        let mut src_idx = 0;
        for i in 0..target_rank {
            if !normalized_dims.contains(&i) {
                if src_idx >= current_shape.len() {
                    return Err(Error::InvalidShape {
                        message: "Internal error: source index out of bounds".to_string(),
                    });
                }
                new_shape[i] = current_shape[src_idx];
                src_idx += 1;
            }
        }

        let mut new_layout = self.layout();
        new_layout.view(&new_shape)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("unsqueeze", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_transpose(&self, dim0: impl Into<Scalar>, dim1: impl Into<Scalar>) -> Result<Self> {
        let rank = self.shape().len();

        let dim0_i32 = dim0.into().as_i32();
        let dim1_i32 = dim1.into().as_i32();

        let dim0_normalized: usize = if dim0_i32 < 0 {
            (rank as i32 + dim0_i32) as usize
        } else {
            dim0_i32 as usize
        };

        let dim1_normalized: usize = if dim1_i32 < 0 {
            (rank as i32 + dim1_i32) as usize
        } else {
            dim1_i32 as usize
        };

        if dim0_normalized >= rank {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim0_normalized, rank
                ),
            });
        }
        if dim1_normalized >= rank {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim1_normalized, rank
                ),
            });
        }

        let mut new_layout = self.layout();
        new_layout.transpose(dim0_normalized, dim1_normalized)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("transpose", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_slice(
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

        let dim_normalized: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim_normalized >= self.ndim() {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim_normalized,
                    self.ndim()
                ),
            });
        }

        let new_layout = self.layout().slice(
            dim_normalized,
            start_i32 as isize,
            end_i32.map(|e| e as isize),
            step_i32 as isize,
        )?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("slice", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_unfold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let size_i32 = size.into().as_i32();
        let step_i32 = step.into().as_i32();

        let dim_usize: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        if dim_usize >= self.ndim() {
            return Err(Error::InvalidShape {
                message: format!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim_usize,
                    self.ndim()
                ),
            });
        }

        if size_i32 <= 0 {
            return Err(Error::InvalidShape {
                message: format!("Unfold size must be positive, got {}", size_i32),
            });
        }

        if step_i32 <= 0 {
            return Err(Error::InvalidShape {
                message: format!("Unfold step must be positive, got {}", step_i32),
            });
        }

        let new_layout = self.layout().unfold(dim_usize, size_i32 as usize, step_i32 as usize)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("unfold", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_broadcast(&self, shape: &[usize]) -> Result<Self> {
        if self.shape() == shape {
            return Ok(self.clone());
        }

        let old_shape = self.shape();

        if old_shape.is_empty() {
            return self.try_broadcast_scalar_to(shape);
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

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: new_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("broadcast", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_broadcast_like(&self, other: &Self) -> Result<Self> {
        self.try_broadcast(&other.shape())
    }

    pub fn try_broadcast_left(&self, batch_dims: &[usize]) -> Result<Self> {
        let mut new_shape = batch_dims.to_vec();
        new_shape.extend(self.shape());
        self.try_broadcast(&new_shape)
    }

    fn try_broadcast_scalar_to(&self, shape: &[usize]) -> Result<Self> {
        if self.size() != 1 {
            return Err(Error::InvalidShape {
                message: "Cannot broadcast non-scalar tensor using scalar broadcast".to_string(),
            });
        }

        let layout = self.layout().broadcast_to(shape)?;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_view(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("broadcast_scalar_to", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_view(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Self> {
        let computed_shape = self.compute_shape_with_auto(shape)?;
        let target_layout = Layout::from_shape(&computed_shape);

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: target_layout,
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_reshape(&output)?;
            },
            TensorMode::Lazy => {
                add_to_graph("reshape", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_reshape(&outputs[0])?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    fn execute_view(&self, target: &Tensor) -> Result<()> {
        let src_storage_id = crate::get_storage_id(self.id())
            .ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;

        link_tensor_to_storage(target.id(), src_storage_id);

        crate::utils::tensor::update_tensor_status(target.id(), TensorUpdateStatus::Materialized)?;

        Ok(())
    }

    fn execute_reshape(&self, target: &Tensor) -> Result<()> {
        if self.is_contiguous() {
            self.execute_view(target)
        } else {
            let contiguous = self.try_contiguous()?;

            let src_storage_id = crate::get_storage_id(contiguous.id())
                .ok_or_else(|| Error::InvalidState("contiguous tensor storage id not found".into()))?;
            crate::link_tensor_to_storage(target.id(), src_storage_id);

            crate::utils::tensor::update_tensor_status(target.id(), TensorUpdateStatus::Materialized)?;

            Ok(())
        }
    }

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
