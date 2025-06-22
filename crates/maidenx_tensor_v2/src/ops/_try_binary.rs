use crate::{
    get_mode, insert_metadata, is_grad_enabled, lazy, next_tensor_id, no_grad,
    utils::{
        broadcast::broadcast_tensors,
        graph::{accumulate, add_to_graph},
        promotion::get_promoted_dtype,
    },
    Tensor, TensorMetadata, TensorMode, TensorUpdateStatus,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};

macro_rules! impl_binary_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(&self, rhs: &Tensor, target: &Tensor, target_dtype: DType, target_layout: Layout) -> Result<()> {
            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let lhs_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };
                let rhs_tensor = if rhs.dtype() != target_dtype {
                    rhs.try_to_dtype(target_dtype)?
                } else {
                    rhs.clone()
                };

                let metadata = Tensor::prepare_binary_metadata(&lhs_tensor, &rhs_tensor);

                let lhs_storage = lhs_tensor.storage()?;
                let lhs_buffer = lhs_storage.buffer();

                let rhs_storage = rhs_tensor.storage()?;
                let rhs_buffer = rhs_storage.buffer();

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        lhs_buffer,
                        rhs_buffer,
                        lhs_tensor.size(),
                        lhs_tensor.ndim(),
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
    pub fn try_add(&self, rhs: &Self) -> Result<Self> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();

        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_add(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("add", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_add(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && (original_lhs.requires_grad() || original_rhs.requires_grad()) {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            if original_lhs.requires_grad() {
                let g1 = output.grad().try_sum_to_shape(&original_lhs.shape())?;
                accumulate(&g1, &original_lhs.grad())?;
            }

            if original_rhs.requires_grad() {
                let g1 = output.grad().try_sum_to_shape(&original_rhs.shape())?;
                accumulate(&g1, &original_rhs.grad())?;
            }
        }

        Ok(output)
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();

        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_sub(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("sub", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_sub(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && (original_lhs.requires_grad() || original_rhs.requires_grad()) {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            if original_lhs.requires_grad() {
                let g1 = output.grad().try_sum_to_shape(&original_lhs.shape())?;
                accumulate(&g1, &original_lhs.grad())?;
            }

            if original_rhs.requires_grad() {
                let g1 = output.grad().try_neg()?;
                let g2 = g1.try_sum_to_shape(&original_rhs.shape())?;
                accumulate(&g2, &original_rhs.grad())?;
            }
        }

        Ok(output)
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();

        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_mul(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("mul", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_mul(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && (original_lhs.requires_grad() || original_rhs.requires_grad()) {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            if original_lhs.requires_grad() {
                let g1 = output.grad().try_mul(&original_rhs)?;
                let g2 = g1.try_sum_to_shape(&original_lhs.shape())?;
                accumulate(&g2, &original_lhs.grad())?;
            }

            if original_rhs.requires_grad() {
                let g1 = output.grad().try_mul(&original_lhs)?;
                let g2 = g1.try_sum_to_shape(&original_rhs.shape())?;
                accumulate(&g2, &original_rhs.grad())?;
            }
        }

        Ok(output)
    }

    pub fn try_div(&self, rhs: &Self) -> Result<Self> {
        let original_lhs = self.clone();
        let original_rhs = rhs.clone();

        let target_dtype = if self.dtype().is_int() && rhs.dtype().is_int() {
            DType::F32
        } else {
            get_promoted_dtype(self.dtype(), rhs.dtype())
        };

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_div(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("div", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_div(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        if is_grad_enabled() && (original_lhs.requires_grad() || original_rhs.requires_grad()) {
            no_grad!();
            lazy!();
            output.try_enable_grad()?;

            if original_lhs.requires_grad() {
                let g1 = output.grad().try_div(&original_rhs)?;
                let g2 = g1.try_sum_to_shape(&original_lhs.shape())?;
                accumulate(&g2, &original_lhs.grad())?;
            }

            if original_rhs.requires_grad() {
                let g1 = output.grad().try_mul(&original_lhs)?;
                let g2 = g1.try_div(&original_rhs.try_square()?)?;
                let g3 = g2.try_neg()?;
                let g4 = g3.try_sum_to_shape(&original_rhs.shape())?;
                accumulate(&g4, &original_rhs.grad())?;
            }
        }

        Ok(output)
    }

    pub fn try_maximum(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_maximum(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("maximum", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_maximum(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_minimum(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());

        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: target_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_minimum(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("minimum", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_minimum(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_logical_and(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_logical_and(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("logical_and", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_logical_and(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        }

        Ok(output)
    }

    pub fn try_logical_or(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_logical_or(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("logical_or", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_logical_or(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_logical_xor(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_logical_xor(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("logical_xor", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_logical_xor(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_eq(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_eq(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("eq", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_eq(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_ne(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_ne(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("ne", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_ne(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_lt(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_lt(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("lt", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_lt(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_le(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_le(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("le", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_le(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_gt(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_gt(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("gt", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_gt(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_ge(&self, rhs: &Self) -> Result<Self> {
        let target_dtype = get_promoted_dtype(self.dtype(), rhs.dtype());
        let (lhs, rhs) = broadcast_tensors(self, rhs)?;
        let target_layout = lhs.layout();
        let result_dtype = DType::BOOL;

        let output_tid = next_tensor_id();
        let metadata = TensorMetadata {
            device: lhs.device(),
            dtype: result_dtype,
            layout: target_layout.clone(),
            requires_grad: false,
            grad_tensor_id: None,
            mode: get_mode(),
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        let output = Tensor(output_tid);

        match get_mode() {
            TensorMode::Eager => {
                lhs.execute_ge(&rhs, &output, target_dtype, target_layout)?;
            },
            TensorMode::Lazy => {
                add_to_graph("ge", &[&lhs, &rhs], &[&output], move |inputs, outputs| {
                    inputs[0].execute_ge(&inputs[1], &outputs[0], target_dtype, target_layout.clone())?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_add_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::add_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_add(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    pub fn try_sub_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::sub_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_sub(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    pub fn try_mul_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::mul_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_mul(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    pub fn try_div_(&mut self, rhs: &Self) -> Result<()> {
        match get_mode() {
            TensorMode::Eager => {
                let mut rhs = if self.shape() == rhs.shape() {
                    rhs.clone()
                } else {
                    rhs.try_broadcast(&self.shape())?
                };

                if self.dtype() != rhs.dtype() {
                    rhs = rhs.try_to_dtype(self.dtype())?;
                }

                let metadata = Tensor::prepare_binary_metadata(self, &rhs);

                let mut storage = self.storage_mut()?;
                unsafe {
                    use std::sync::Arc;

                    let buffer_mut = Arc::get_mut(&mut storage.buffer).ok_or(Error::BufferShared)?;
                    maidenx_core::be::ops::binary::div_inplace(
                        buffer_mut,
                        rhs.storage()?.buffer(),
                        self.size(),
                        self.ndim(),
                        Some(&metadata),
                    )?;
                }
                Ok(())
            },
            TensorMode::Lazy => {
                let result = self.try_div(rhs)?;
                *self = result;
                Ok(())
            },
        }
    }

    impl_binary_execute!(execute_add, maidenx_core::be::ops::binary::add);
    impl_binary_execute!(execute_sub, maidenx_core::be::ops::binary::sub);
    impl_binary_execute!(execute_mul, maidenx_core::be::ops::binary::mul);
    impl_binary_execute!(execute_div, maidenx_core::be::ops::binary::div);
    impl_binary_execute!(execute_maximum, maidenx_core::be::ops::binary::maximum);
    impl_binary_execute!(execute_minimum, maidenx_core::be::ops::binary::minimum);
    impl_binary_execute!(execute_logical_and, maidenx_core::be::ops::binary::logical_and);
    impl_binary_execute!(execute_logical_or, maidenx_core::be::ops::binary::logical_or);
    impl_binary_execute!(execute_logical_xor, maidenx_core::be::ops::binary::logical_xor);
    impl_binary_execute!(execute_eq, maidenx_core::be::ops::binary::eq);
    impl_binary_execute!(execute_ne, maidenx_core::be::ops::binary::ne);
    impl_binary_execute!(execute_lt, maidenx_core::be::ops::binary::lt);
    impl_binary_execute!(execute_le, maidenx_core::be::ops::binary::le);
    impl_binary_execute!(execute_gt, maidenx_core::be::ops::binary::gt);
    impl_binary_execute!(execute_ge, maidenx_core::be::ops::binary::ge);

    fn prepare_binary_metadata(lhs: &Tensor, rhs: &Tensor) -> Vec<usize> {
        let mut metadata = Vec::new();
        metadata.extend_from_slice(&lhs.shape());
        metadata.extend_from_slice(&lhs.strides());
        metadata.extend_from_slice(&rhs.strides());
        metadata.push(lhs.offset());
        metadata.push(rhs.offset());
        metadata
    }
}
