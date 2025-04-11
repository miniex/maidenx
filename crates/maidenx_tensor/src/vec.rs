use crate::Tensor;
use maidenx_core::{
    dtype::DType,
    error::{Error, Result},
};

impl Tensor {
    pub fn to_flatten_vec<T: Default + Clone + 'static>(&self) -> Result<Vec<T>> {
        let target_dtype =
            get_dtype_for_type::<T>().ok_or_else(|| Error::InvalidArgument("Unsupported type".into()))?;
        let tensor = if self.dtype() != target_dtype {
            self.to_dtype(target_dtype)?
        } else {
            self.clone()
        };

        let size = tensor.size();
        let shape = tensor.shape();
        let strides = tensor.strides();
        let offset = tensor.offset();
        let elem_size = tensor.dtype().size_in_bytes();
        let buffer_size = tensor.buffer().len() * elem_size;

        let mut raw_data = vec![0u8; buffer_size];

        unsafe {
            tensor
                .buffer()
                .copy_to_host(raw_data.as_mut_ptr() as *mut std::ffi::c_void, buffer_size, 0, 0)?;
        }

        let mut result = vec![T::default(); size];
        let mut indices = vec![0; shape.len()];
        let mut dst_idx = 0;

        let calc_src_offset = |indices: &[usize], strides: &[usize], offset: usize| -> usize {
            offset
                + indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>()
        };

        loop {
            let src_offset = calc_src_offset(&indices, strides, offset);

            if src_offset * elem_size < raw_data.len() {
                // Copy element from source to destination
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw_data.as_ptr().add(src_offset * elem_size),
                        (result.as_mut_ptr() as *mut u8).add(dst_idx * elem_size),
                        elem_size,
                    );
                }
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
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>() {
        Some(DType::U16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        Some(DType::U32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        Some(DType::U64)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        Some(DType::I8)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
        Some(DType::I16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        Some(DType::I32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        Some(DType::I64)
    } else {
        None
    }
}
