use crate::Tensor;
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};

impl Tensor {
    pub fn to_flatten_vec<T: Default + Clone + 'static>(&self) -> Result<Vec<T>> {
        let target_dtype = get_dtype_for_type::<T>().ok_or_else(|| Error::InvalidArgument("Unsupported type".into()))?;

        let tensor = if self.dtype() != target_dtype {
            self.to_dtype(target_dtype)?
        } else {
            self.clone()
        };

        let size = tensor.size();
        let shape = tensor.shape();
        let strides = tensor.strides();
        let elem_size = tensor.dtype().size_in_bytes();

        // Get raw data from buffer
        let mut raw_data = vec![0u8; size * elem_size];
        unsafe {
            tensor
                .buffer()
                .copy_to_host(raw_data.as_mut_ptr() as *mut std::ffi::c_void, raw_data.len())?;
        }

        let mut result = vec![T::default(); size];
        let mut indices = vec![0; shape.len()];
        let mut dst_idx = 0;

        // Helper function to calculate source offset using strides
        let calc_src_offset =
            |indices: &[usize], strides: &[usize]| -> usize { indices.iter().zip(strides.iter()).map(|(&idx, &stride)| idx * stride).sum() };

        loop {
            // Calculate source offset using strides
            let src_offset = calc_src_offset(&indices, strides);

            // Copy element from source to destination
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw_data.as_ptr().add(src_offset * elem_size),
                    (result.as_mut_ptr() as *mut u8).add(dst_idx * elem_size),
                    elem_size,
                );
            }

            dst_idx += 1;

            // Update indices
            if dst_idx == size {
                return Ok(result);
            }

            let mut dim = shape.len();
            while dim > 0 {
                dim -= 1;
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    pub fn item_at_flat_index(&self, index: usize) -> Result<Scalar> {
        if index >= self.size() {
            return Err(Error::IndexOutOfBounds { index, size: self.size() });
        }

        let indices = self.flat_index_to_indices(index)?;
        let buffer_index = self.indices_to_buffer_index(&indices)?;

        unsafe {
            let buffer = self.buffer();
            let ptr = (buffer.as_ptr() as *const u8).add(buffer_index * self.dtype().size_in_bytes());
            Ok(self.dtype().read_scalar(ptr))
        }
    }

    pub fn set_flat_index(&mut self, index: usize, value: impl Into<Scalar>) -> Result<()> {
        if index >= self.size() {
            return Err(Error::IndexOutOfBounds { index, size: self.size() });
        }

        let scalar_value = value.into();

        let indices = self.flat_index_to_indices(index)?;
        let buffer_index = self.indices_to_buffer_index(&indices)?;
        let dtype = self.dtype();

        unsafe {
            self.with_buffer_mut(|buf| {
                let ptr = (buf.as_mut_ptr() as *mut u8).add(buffer_index * dtype.size_in_bytes());
                dtype.write_scalar(ptr, scalar_value);
                Ok(())
            })?
        }

        Ok(())
    }

    // helper

    fn indices_to_buffer_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(Error::InvalidShape {
                message: format!("Expected {} indices, got {}", self.ndim(), indices.len()),
            });
        }

        let mut buffer_idx = 0;
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[dim] {
                return Err(Error::IndexOutOfBounds {
                    index: idx,
                    size: self.shape()[dim],
                });
            }
            buffer_idx += idx * self.strides()[dim];
        }

        Ok(buffer_idx)
    }

    fn flat_index_to_indices(&self, flat_index: usize) -> Result<Vec<usize>> {
        if flat_index >= self.size() {
            return Err(Error::IndexOutOfBounds {
                index: flat_index,
                size: self.size(),
            });
        }

        let mut indices = vec![0; self.ndim()];
        let mut remaining = flat_index;

        for i in (0..self.ndim()).rev() {
            let dim_size = self.shape()[i];
            indices[i] = remaining % dim_size;
            remaining /= dim_size;
        }

        Ok(indices)
    }
}

fn get_dtype_for_type<T: 'static>() -> Option<DType> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>() {
        Some(DType::BF16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
        Some(DType::F16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        Some(DType::F32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        Some(DType::F64)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
        Some(DType::BOOL)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
        Some(DType::U8)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        Some(DType::U32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        Some(DType::I8)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        Some(DType::I32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        Some(DType::I64)
    } else {
        None
    }
}
