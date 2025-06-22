use crate::{
    get_mode, insert_metadata, next_tensor_id, utils::graph::add_to_graph, Tensor, TensorMetadata, TensorMode,
    TensorUpdateStatus,
};
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};

macro_rules! impl_padding_execute_with_scalar {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target: &Tensor,
            target_dtype: DType,
            target_layout: Layout,
            paddings: &[(usize, usize)],
            pad_scalar: Scalar,
        ) -> Result<()> {
            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };

                let metadata = input_tensor.prepare_metadata_for_padding(paddings);

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        target_size,
                        input_tensor.ndim(),
                        Some(&metadata),
                        pad_scalar,
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

macro_rules! impl_padding_execute {
    ($fn_name:ident, $backend_fn:path) => {
        fn $fn_name(
            &self,
            target: &Tensor,
            target_dtype: DType,
            target_layout: Layout,
            paddings: &[(usize, usize)],
        ) -> Result<()> {
            use maidenx_core::buffer::BufferManager;
            use std::sync::Arc;

            let target_size = target_layout.size();
            let mut buffer = BufferManager::create(target_size, self.device(), target_dtype)?;

            {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                let input_tensor = if self.dtype() != target_dtype {
                    self.try_to_dtype(target_dtype)?
                } else {
                    self.clone()
                };

                let metadata = input_tensor.prepare_metadata_for_padding(paddings);

                unsafe {
                    $backend_fn(
                        buffer_mut,
                        input_tensor.storage()?.buffer(),
                        input_tensor.size(),
                        target_size,
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
    pub fn try_pad(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor> {
        self.try_pad_with_constant(paddings, pad_value)
    }

    pub fn try_pad_with_constant(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let mut output_shape = Vec::with_capacity(self.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(self.shape()[i] + pad_before + pad_after);
        }

        let target_layout = Layout::from_shape(&output_shape);
        let pad_scalar = pad_value.into();

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

        match get_mode() {
            TensorMode::Eager => {
                self.execute_pad_with_constant(&output, target_dtype, target_layout, paddings, pad_scalar)?;
            },
            TensorMode::Lazy => {
                let paddings_vec = paddings.to_vec();
                add_to_graph("pad_with_constant", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_pad_with_constant(
                        &outputs[0],
                        target_dtype,
                        target_layout.clone(),
                        &paddings_vec,
                        pad_scalar,
                    )?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_pad_with_reflection(&self, paddings: &[(usize, usize)]) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            let dim_size = self.shape()[i];

            if pad_before >= dim_size || pad_after >= dim_size {
                return Err(Error::InvalidArgument(format!(
                    "Reflection padding width ({}, {}) must be less than the dimension size ({})",
                    pad_before, pad_after, dim_size
                )));
            }

            if dim_size <= 1 && (pad_before > 0 || pad_after > 0) {
                return Err(Error::InvalidArgument(format!(
                    "Reflection padding requires input dimension > 1 for non-zero padding, but got dim {} with size {}",
                    i, dim_size
                )));
            }
        }

        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let mut output_shape = Vec::with_capacity(self.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(self.shape()[i] + pad_before + pad_after);
        }

        let target_layout = Layout::from_shape(&output_shape);

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

        match get_mode() {
            TensorMode::Eager => {
                self.execute_pad_with_reflection(&output, target_dtype, target_layout, paddings)?;
            },
            TensorMode::Lazy => {
                let paddings_vec = paddings.to_vec();
                add_to_graph("pad_with_reflection", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_pad_with_reflection(
                        &outputs[0],
                        target_dtype,
                        target_layout.clone(),
                        &paddings_vec,
                    )?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    pub fn try_pad_with_replication(&self, paddings: &[(usize, usize)]) -> Result<Tensor> {
        if paddings.len() != self.ndim() {
            return Err(Error::ShapeMismatch {
                expected: self.ndim(),
                got: paddings.len(),
                msg: "Number of padding pairs should match tensor dimensions".to_string(),
            });
        }

        let target_dtype = if self.dtype().is_bool() {
            DType::U8
        } else {
            self.dtype()
        };

        let mut output_shape = Vec::with_capacity(self.ndim());
        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            output_shape.push(self.shape()[i] + pad_before + pad_after);
        }

        let target_layout = Layout::from_shape(&output_shape);

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

        match get_mode() {
            TensorMode::Eager => {
                self.execute_pad_with_replication(&output, target_dtype, target_layout, paddings)?;
            },
            TensorMode::Lazy => {
                let paddings_vec = paddings.to_vec();
                add_to_graph("pad_with_replication", &[self], &[&output], move |inputs, outputs| {
                    inputs[0].execute_pad_with_replication(
                        &outputs[0],
                        target_dtype,
                        target_layout.clone(),
                        &paddings_vec,
                    )?;
                    Ok(())
                })?;
            },
        };

        Ok(output)
    }

    impl_padding_execute_with_scalar!(
        execute_pad_with_constant,
        maidenx_core::be::ops::padding::pad_with_constant
    );
    impl_padding_execute!(
        execute_pad_with_reflection,
        maidenx_core::be::ops::padding::pad_with_reflection
    );
    impl_padding_execute!(
        execute_pad_with_replication,
        maidenx_core::be::ops::padding::pad_with_replication
    );

    fn prepare_metadata_for_padding(&self, paddings: &[(usize, usize)]) -> Vec<usize> {
        let mut info = Vec::new();

        let input_shape = self.shape();
        let input_strides = self.strides();

        info.extend_from_slice(&input_shape);
        info.extend_from_slice(&input_strides);
        info.push(self.offset());

        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            info.push(input_shape[i] + pad_before + pad_after);
        }

        for &(pad_before, pad_after) in paddings {
            info.push(pad_before);
            info.push(pad_after);
        }

        info
    }

    fn prepare_metadata_for_padding_backward(
        input_shape: &[usize],
        input_strides: &[usize],
        input_offset: usize,
        paddings: &[(usize, usize)],
    ) -> Vec<usize> {
        let mut info = Vec::new();

        info.extend_from_slice(input_shape);
        info.extend_from_slice(input_strides);
        info.push(input_offset);

        for (i, &(pad_before, pad_after)) in paddings.iter().enumerate() {
            info.push(input_shape[i] + pad_before + pad_after);
        }

        for &(pad_before, pad_after) in paddings {
            info.push(pad_before);
            info.push(pad_after);
        }

        info
    }
}
