use crate::{
    get_mode, insert_metadata, next_tensor_id, utils::graph::add_to_graph, Tensor, TensorMetadata, TensorMode,
    TensorUpdateStatus,
};
use maidenx_core::{
    buffer::BufferManager,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};
use std::sync::Arc;

macro_rules! impl_reduction_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(&self, target: &Tensor, target_dtype: DType, target_layout: Layout, dim: usize) -> Result<()> {
            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };

                let metadata = input_tensor.prepare_reduction_metadata(dim);

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        input_tensor.ndim(),
                        1,
                        Some(&metadata),
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target.id(), sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));

            crate::utils::tensor::update_tensor_status(target.id(), TensorUpdateStatus::Materialized)?;

            Ok(())
        }
    };
}

macro_rules! impl_reduction_execute_special {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target: &Tensor,
            target_dtype: DType,
            target_layout: Layout,
            metadata: Vec<usize>,
        ) -> Result<()> {
            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        input_tensor.ndim(),
                        Some(&metadata),
                    )?;
                }
            }

            let sid = crate::next_storage_id();
            crate::link_tensor_to_storage(target.id(), sid);
            crate::insert_storage(sid, crate::TensorStorage::new(buffer));

            crate::utils::tensor::update_tensor_status(target.id(), TensorUpdateStatus::Materialized)?;

            Ok(())
        }
    };
}

impl Tensor {
    pub fn try_sum(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
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

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&shape);

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        let result = match get_mode() {
            TensorMode::Eager => {
                self.execute_sum(&output, target_dtype, target_layout, dim)?;
                output
            },
            TensorMode::Lazy => {
                add_to_graph("sum", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sum(&outputs[0], target_dtype, target_layout.clone(), dim)?;
                    Ok(())
                })?;
                output
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    pub fn try_sum_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_sum(dim, false)?;
        }
        Ok(result)
    }

    pub fn try_sum_to_shape(&self, shape: &[usize]) -> Result<Tensor> {
        if shape.len() != self.ndim() {
            if self.ndim() > shape.len() {
                let mut result = self.clone();
                while result.ndim() > shape.len() {
                    result = result.try_sum(0, false)?;
                }
                return result.try_sum_to_shape(shape);
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

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(shape);
        let metadata = self.prepare_metadata_for_shape(shape);

        let output_tid = next_tensor_id();
        let metadata_tensor = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata_tensor);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_sum_to_shape(&output, target_dtype, target_layout, metadata)?;
                Ok(output)
            },
            TensorMode::Lazy => {
                add_to_graph("sum_to_shape", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sum_to_shape(
                        &outputs[0],
                        target_dtype,
                        target_layout.clone(),
                        metadata.clone(),
                    )?;
                    Ok(())
                })?;
                Ok(output)
            },
        }
    }

    pub fn try_mean(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Self> {
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

        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };
        let target_layout = Layout::from_shape(&shape);

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        let result = match get_mode() {
            TensorMode::Eager => {
                self.execute_mean(&output, target_dtype, target_layout, dim)?;
                output
            },
            TensorMode::Lazy => {
                add_to_graph("mean", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_mean(&outputs[0], target_dtype, target_layout.clone(), dim)?;
                    Ok(())
                })?;
                output
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    pub fn try_mean_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_mean(dim, false)?;
        }
        Ok(result)
    }

    pub fn try_fold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Tensor> {
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

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&output_shape);
        let metadata = self.prepare_metadata_for_fold(dim, window_dim, orig_dim_size, step_i32 as usize, window_size);

        let output_tid = next_tensor_id();
        let metadata_tensor = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata_tensor);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_fold(&output, target_dtype, target_layout, metadata)?;
                Ok(output)
            },
            TensorMode::Lazy => {
                add_to_graph("fold", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_fold(&outputs[0], target_dtype, target_layout.clone(), metadata.clone())?;
                    Ok(())
                })?;
                Ok(output)
            },
        }
    }

    pub fn try_max(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
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

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&shape);

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        let result = match get_mode() {
            TensorMode::Eager => {
                self.execute_max(&output, target_dtype, target_layout, dim)?;
                output
            },
            TensorMode::Lazy => {
                add_to_graph("max", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_max(&outputs[0], target_dtype, target_layout.clone(), dim)?;
                    Ok(())
                })?;
                output
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    pub fn try_max_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_max(dim, false)?;
        }
        Ok(result)
    }

    pub fn try_min(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor> {
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

        let target_dtype = self.dtype();
        let target_layout = Layout::from_shape(&shape);

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        let result = match get_mode() {
            TensorMode::Eager => {
                self.execute_min(&output, target_dtype, target_layout, dim)?;
                output
            },
            TensorMode::Lazy => {
                add_to_graph("min", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_min(&outputs[0], target_dtype, target_layout.clone(), dim)?;
                    Ok(())
                })?;
                output
            },
        };

        if keep_dim {
            let mut keep_dim_shape = original_shape;
            keep_dim_shape[dim] = 1;
            result.try_view(&keep_dim_shape)
        } else {
            Ok(result)
        }
    }

    pub fn try_min_all(&self) -> Result<Self> {
        let mut result = self.clone();
        for dim in (0..self.ndim()).rev() {
            result = result.try_min(dim, false)?;
        }
        Ok(result)
    }

    pub fn try_norm(&self, p: impl Into<Scalar>, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Self> {
        let p_f32 = p.into().as_f32();
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        if p_f32 == 1.0 {
            // L1 norm: sum(abs(x))
            let abs_input = self.try_abs()?;
            let result = abs_input.try_sum(dim, keep_dim)?;
            Ok(result)
        } else if p_f32 == 2.0 {
            // L2 norm: sqrt(sum(x^2))
            let squared = self.try_pow(2.0)?;
            let sum_squared = squared.try_sum(dim, keep_dim)?;
            let result = sum_squared.try_sqrt()?;
            Ok(result)
        } else {
            // Generalized p-norm: (sum(abs(x)^p))^(1/p)
            let abs_input = self.try_abs()?;
            let pow_input = abs_input.try_pow(p_f32)?;
            let sum_result = pow_input.try_sum(dim, keep_dim)?;
            let result = sum_result.try_pow(1.0 / p_f32)?;
            Ok(result)
        }
    }

    pub fn try_norm_all(&self, p: impl Into<Scalar>) -> Result<Tensor> {
        let p_f32 = p.into().as_f32();

        if p_f32 == 1.0 {
            self.try_abs()?.try_sum_all()
        } else if p_f32 == 2.0 {
            let squared = self.try_pow(2.0)?;
            let sum_squared = squared.try_sum_all()?;
            sum_squared.try_sqrt()
        } else {
            let abs_values = self.try_abs()?;
            let pow_values = abs_values.try_pow(p_f32)?;
            let sum_result = pow_values.try_sum_all()?;
            sum_result.try_pow(1.0 / p_f32)
        }
    }

    pub fn try_var(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Result<Self> {
        let dim_i32 = dim.into().as_i32();
        let dim: usize = if dim_i32 < 0 {
            (self.ndim() as i32 + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        let shape: Vec<usize> = self.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::DimensionOutOfBounds {
                dim: dim as i32,
                ndim: self.ndim(),
            });
        }

        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };
        let input = self.try_to_dtype(target_dtype)?;

        let mean = input.try_mean(dim, true)?;
        let centered = input.try_sub(&mean)?;
        let squared_diff = centered.try_pow(2.0)?;
        let sum_squared_diff = squared_diff.try_sum(dim, keep_dim)?;
        let n = input.shape()[dim] as f32;
        let divisor = if unbiased { n - 1.0 } else { n };
        if divisor == 0.0 {
            return Err(Error::InvalidArgument(
                "Cannot compute unbiased variance for dimension with 1 element.".to_string(),
            ));
        }
        let result = sum_squared_diff.try_div_scalar(divisor)?;

        Ok(result)
    }

    pub fn try_var_all(&self, unbiased: bool) -> Result<Tensor> {
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let input = self.try_to_dtype(target_dtype)?;
        let mean = input.try_mean_all()?;
        let centered = input.try_sub(&mean)?;
        let squared_diff = centered.try_pow(2.0)?;
        let sum_squared_diff = squared_diff.try_sum_all()?;

        let n = self.shape().iter().product::<usize>() as f32;
        let divisor = if unbiased { n - 1.0 } else { n };
        if divisor == 0.0 {
            return Err(Error::InvalidArgument(
                "Cannot compute unbiased variance for a tensor with 1 element.".to_string(),
            ));
        }
        let result = sum_squared_diff.try_div_scalar(divisor)?;

        Ok(result)
    }

    pub fn try_std(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Result<Tensor> {
        let var_result = self.try_var(dim, keep_dim, unbiased)?;
        let result = var_result.try_sqrt()?;
        Ok(result)
    }

    pub fn try_std_all(&self, unbiased: bool) -> Result<Tensor> {
        let var_result = self.try_var_all(unbiased)?;
        let result = var_result.try_sqrt()?;
        Ok(result)
    }

    impl_reduction_execute!(execute_sum, maidenx_core::be::ops::reduction::sum);
    impl_reduction_execute_special!(execute_sum_to_shape, maidenx_core::be::ops::reduction::sum_to_shape);
    impl_reduction_execute!(execute_mean, maidenx_core::be::ops::reduction::mean);
    impl_reduction_execute_special!(execute_fold, maidenx_core::be::ops::reduction::fold);
    impl_reduction_execute!(execute_max, maidenx_core::be::ops::reduction::max);
    impl_reduction_execute!(execute_min, maidenx_core::be::ops::reduction::min);

    fn prepare_reduction_metadata(&self, dim: usize) -> Vec<usize> {
        let mut info = Vec::new();
        let shape = self.shape();
        let strides = self.strides();
        info.extend_from_slice(&shape);
        info.extend_from_slice(&strides);
        info.push(shape[dim]);
        info.push(strides[dim]);
        info.push(self.offset());
        info
    }

    fn prepare_metadata_for_shape(&self, target_shape: &[usize]) -> Vec<usize> {
        let mut info = Vec::new();
        let input_shape = self.shape();
        let input_strides = self.strides();
        info.extend_from_slice(&input_shape);
        info.extend_from_slice(&input_strides);
        info.extend_from_slice(target_shape);
        info.push(self.offset());
        info
    }

    fn prepare_metadata_for_fold(
        &self,
        fold_dim: usize,
        window_dim: usize,
        fold_size: usize,
        step: usize,
        window_size: usize,
    ) -> Vec<usize> {
        let mut info = Vec::new();
        let input_shape = self.shape();
        let input_strides = self.strides();

        info.extend_from_slice(&input_shape);
        info.extend_from_slice(&input_strides);

        info.push(fold_dim);
        info.push(window_dim);
        info.push(fold_size);
        info.push(step);
        info.push(window_size);

        info.push(self.offset());

        info
    }
}
