use crate::{
    get_mode, insert_metadata, is_grad_enabled, lazy, next_tensor_id, no_grad,
    utils::graph::{accumulate, add_to_graph},
    Tensor, TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};

macro_rules! impl_unary_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(&self, target: &Tensor, target_dtype: DType) -> Result<()> {
            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let shape = self.shape();
            let size = shape.iter().product();
            let mut buffer = BufferManager::create(size, self.device(), target_dtype)?;

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
                        size,
                        shape.len(),
                        Some(&self.prepare_metadata()),
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

macro_rules! impl_unary_with_scalar_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(&self, target: &Tensor, target_dtype: DType, scalar: Scalar) -> Result<()> {
            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let shape = self.shape();
            let size = shape.iter().product();
            let mut buffer = BufferManager::create(size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };
                let input_scalar = if scalar.dtype() != target_dtype {
                    scalar.to_dtype(target_dtype)
                } else {
                    scalar.clone()
                };

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_scalar,
                        size,
                        shape.len(),
                        Some(&self.prepare_metadata()),
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
    pub fn try_neg(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8.to_signed()
        } else {
            self.dtype().to_signed()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_neg(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("neg", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_neg(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_neg()?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_abs(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        // For unsigned types, absolute value is identity
        if target_dtype.is_uint() && !self.dtype().is_bool() {
            return Ok(self.clone());
        }

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_abs(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("abs", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_abs(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul(&self.try_sign()?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_sign(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_sign(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("sign", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sign(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_square(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_square(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("square", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_square(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul_scalar(2.0)?.try_mul(self)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_sqrt(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_sqrt(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("sqrt", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sqrt(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_div(&output.try_mul_scalar(2.0)?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_relu(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_relu(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("relu", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_relu(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output
                .grad()
                .try_mul(&self.try_gt_scalar(0.0)?.try_to_dtype(output.dtype())?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_sigmoid(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_sigmoid(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("sigmoid", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sigmoid(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output
                .grad()
                .try_mul(&output)?
                .try_mul(&output.try_sub_scalar(1.0)?.try_neg()?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_tanh(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_tanh(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("tanh", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_tanh(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output
                .grad()
                .try_mul(&output.try_pow(2.0)?.try_neg()?.try_add_scalar(1.0)?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_gelu(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_gelu(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("gelu", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_gelu(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let sqrt_2_over_pi = 0.7978845608028654;
            let coeff = 0.044715;

            let x_squared = self.try_mul(self)?;
            let x_cubed = x_squared.try_mul(self)?;
            let tanh_arg = self
                .try_add(&x_cubed.try_mul_scalar(coeff)?)?
                .try_mul_scalar(sqrt_2_over_pi)?;

            let tanh_val = tanh_arg.try_tanh()?;
            let sech_squared = tanh_val.try_mul(&tanh_val)?.try_mul_scalar(-1.0)?.try_add_scalar(1.0)?;
            let inner_derivative = x_squared
                .try_mul_scalar(3.0 * coeff)?
                .try_add_scalar(1.0)?
                .try_mul_scalar(sqrt_2_over_pi)?;

            let term1 = tanh_val.try_add_scalar(1.0)?.try_mul_scalar(0.5)?;
            let term2 = self
                .try_mul(&sech_squared)?
                .try_mul(&inner_derivative)?
                .try_mul_scalar(0.5)?;
            let g1 = output.grad().try_mul(&term1.try_add(&term2)?)?;

            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_sin(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_sin(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("sin", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sin(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul(&self.try_cos()?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_cos(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_cos(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("cos", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_cos(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul(&self.try_sin()?.try_neg()?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_tan(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_tan(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("tan", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_tan(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let cos_x = self.try_cos()?;
            let sec_squared = cos_x.try_mul(&cos_x)?.try_pow(-1.0)?;
            let g1 = output.grad().try_mul(&sec_squared)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_ln(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_ln(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("ln", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_ln(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_div(self)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_log(&self) -> Result<Self> {
        self.try_ln()
    }

    pub fn try_log10(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_log10(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("log10", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_log10(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let ln10 = std::f32::consts::LN_10;
            let g1 = output.grad().try_div(self)?.try_div_scalar(ln10)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_log2(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_log2(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("log2", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_log2(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let ln2 = std::f32::consts::LN_2;
            let g1 = output.grad().try_div(self)?.try_div_scalar(ln2)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_exp(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_exp(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("exp", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_exp(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul(&output)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_exp10(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_exp10(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("exp10", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_exp10(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let ln10 = std::f32::consts::LN_10;
            let g1 = output.grad().try_mul(&output)?.try_mul_scalar(ln10)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_exp2(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_exp2(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("exp2", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_exp2(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let ln2 = std::f32::consts::LN_2;
            let g1 = output.grad().try_mul(&output)?.try_mul_scalar(ln2)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_softplus(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_softplus(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("softplus", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_softplus(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul(&self.try_sigmoid()?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_recip(&self) -> Result<Self> {
        let target_dtype = if self.dtype().is_float() {
            self.dtype()
        } else {
            DType::F32
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_recip(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("recip", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_recip(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let input_squared = self.try_square()?;
            let neg_grad = output.grad().try_neg()?;
            let g1 = neg_grad.try_div(&input_squared)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_logical_not(&self) -> Result<Self> {
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_logical_not(&output, target_dtype)?;
            },
            TensorMode::Lazy => {
                add_to_graph("logical_not", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_logical_not(&outputs[0], target_dtype)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_add_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_add_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("add_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_add_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_sub_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_sub_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("sub_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sub_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_mul_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_mul_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("mul_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_mul_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_mul_scalar(scalar)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_div_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_div_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("div_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_div_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output.grad().try_div_scalar(scalar)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_maximum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_maximum_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("maximum_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_maximum_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_minimum_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = if scalar.is_float() && !scalar.is_integer_value() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_minimum_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("minimum_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_minimum_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_pow<T: Into<Scalar>>(&self, exponent: T) -> Result<Self> {
        let exponent = exponent.into();
        let target_dtype = if exponent.as_f32() < 0.0 {
            DType::F32
        } else if exponent.is_float() && !exponent.is_integer_value() {
            exponent.dtype()
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_pow(&output, target_dtype, exponent)?;
            },
            TensorMode::Lazy => {
                add_to_graph("pow", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_pow(&outputs[0], target_dtype, exponent)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let g1 = output
                .grad()
                .try_mul_scalar(exponent)?
                .try_mul(&self.try_pow(exponent.as_f32() - 1.0)?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_leaky_relu<T: Into<Scalar>>(&self, alpha: T) -> Result<Self> {
        let alpha = alpha.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_leaky_relu(&output, target_dtype, alpha)?;
            },
            TensorMode::Lazy => {
                add_to_graph("leaky_relu", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_leaky_relu(&outputs[0], target_dtype, alpha)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let ones = self.try_gt_scalar(0.0)?.try_to_dtype(output.dtype())?;
            let alpha_mask = ones.try_mul_scalar(-1.0)?.try_add_scalar(1.0)?;

            let g1 = output
                .grad()
                .try_mul(&ones.try_add(&alpha_mask.try_mul_scalar(alpha)?)?)?;
            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_elu<T: Into<Scalar>>(&self, alpha: T) -> Result<Self> {
        let alpha = alpha.into();
        let target_dtype = if self.dtype().is_int() {
            DType::F32
        } else {
            self.dtype()
        };

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_elu(&output, target_dtype, alpha)?;
            },
            TensorMode::Lazy => {
                add_to_graph("elu", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_elu(&outputs[0], target_dtype, alpha)?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && self.requires_grad() {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            let pos_mask = self.try_gt_scalar(0.0)?.try_to_dtype(output.dtype())?;
            let neg_mask = pos_mask.try_mul_scalar(-1.0)?.try_add_scalar(1.0)?;
            let neg_grad = output.try_mul(&neg_mask)?.try_add_scalar(alpha)?.try_mul(&neg_mask)?;
            let g1 = output.grad().try_mul(&pos_mask.try_add(&neg_grad)?)?;

            accumulate(&g1, &self.grad())?;
        }

        Ok(output)
    }

    pub fn try_eq_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_eq_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("eq_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_eq_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_ne_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_ne_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("ne_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_ne_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_lt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_lt_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("lt_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_lt_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_le_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_le_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("le_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_le_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_gt_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_gt_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("gt_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_gt_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_ge_scalar<T: Into<Scalar>>(&self, scalar: T) -> Result<Self> {
        let scalar = scalar.into();
        let target_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: self.device(),
            dtype: target_dtype,
            layout: self.layout().clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                self.execute_ge_scalar(&output, target_dtype, scalar)?;
            },
            TensorMode::Lazy => {
                add_to_graph("ge_scalar", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_ge_scalar(&outputs[0], target_dtype, scalar)?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    impl_unary_execute!(execute_neg, maidenx_core::be::ops::unary::neg);
    impl_unary_execute!(execute_abs, maidenx_core::be::ops::unary::abs);
    impl_unary_execute!(execute_sign, maidenx_core::be::ops::unary::sign);
    impl_unary_execute!(execute_square, maidenx_core::be::ops::unary::square);
    impl_unary_execute!(execute_sqrt, maidenx_core::be::ops::unary::sqrt);
    impl_unary_execute!(execute_relu, maidenx_core::be::ops::unary::relu);
    impl_unary_execute!(execute_sigmoid, maidenx_core::be::ops::unary::sigmoid);
    impl_unary_execute!(execute_tanh, maidenx_core::be::ops::unary::tanh);
    impl_unary_execute!(execute_gelu, maidenx_core::be::ops::unary::gelu);
    impl_unary_execute!(execute_sin, maidenx_core::be::ops::unary::sin);
    impl_unary_execute!(execute_cos, maidenx_core::be::ops::unary::cos);
    impl_unary_execute!(execute_tan, maidenx_core::be::ops::unary::tan);
    impl_unary_execute!(execute_ln, maidenx_core::be::ops::unary::ln);
    impl_unary_execute!(execute_log10, maidenx_core::be::ops::unary::log10);
    impl_unary_execute!(execute_log2, maidenx_core::be::ops::unary::log2);
    impl_unary_execute!(execute_exp, maidenx_core::be::ops::unary::exp);
    impl_unary_execute!(execute_exp10, maidenx_core::be::ops::unary::exp10);
    impl_unary_execute!(execute_exp2, maidenx_core::be::ops::unary::exp2);
    impl_unary_execute!(execute_softplus, maidenx_core::be::ops::unary::softplus);
    impl_unary_execute!(execute_recip, maidenx_core::be::ops::unary::recip);
    impl_unary_execute!(execute_logical_not, maidenx_core::be::ops::unary::logical_not);

    impl_unary_with_scalar_execute!(execute_add_scalar, maidenx_core::be::ops::unary::add_scalar);
    impl_unary_with_scalar_execute!(execute_sub_scalar, maidenx_core::be::ops::unary::sub_scalar);
    impl_unary_with_scalar_execute!(execute_mul_scalar, maidenx_core::be::ops::unary::mul_scalar);
    impl_unary_with_scalar_execute!(execute_div_scalar, maidenx_core::be::ops::unary::div_scalar);
    impl_unary_with_scalar_execute!(execute_maximum_scalar, maidenx_core::be::ops::unary::maximum_scalar);
    impl_unary_with_scalar_execute!(execute_minimum_scalar, maidenx_core::be::ops::unary::minimum_scalar);
    impl_unary_with_scalar_execute!(execute_pow, maidenx_core::be::ops::unary::pow);
    impl_unary_with_scalar_execute!(execute_leaky_relu, maidenx_core::be::ops::unary::leaky_relu);
    impl_unary_with_scalar_execute!(execute_elu, maidenx_core::be::ops::unary::elu);
    impl_unary_with_scalar_execute!(execute_eq_scalar, maidenx_core::be::ops::unary::eq_scalar);
    impl_unary_with_scalar_execute!(execute_ne_scalar, maidenx_core::be::ops::unary::ne_scalar);
    impl_unary_with_scalar_execute!(execute_lt_scalar, maidenx_core::be::ops::unary::lt_scalar);
    impl_unary_with_scalar_execute!(execute_le_scalar, maidenx_core::be::ops::unary::le_scalar);
    impl_unary_with_scalar_execute!(execute_gt_scalar, maidenx_core::be::ops::unary::gt_scalar);
    impl_unary_with_scalar_execute!(execute_ge_scalar, maidenx_core::be::ops::unary::ge_scalar);

    fn prepare_metadata(&self) -> Vec<usize> {
        let mut info = Vec::new();
        info.extend_from_slice(&self.shape());
        info.extend_from_slice(&self.strides());
        info.push(self.offset());
        info
    }
}
