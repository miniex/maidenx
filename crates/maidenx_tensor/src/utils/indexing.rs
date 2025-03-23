use crate::Tensor;
use maidenx_core::{
    error::{Error, Result},
    scalar::Scalar,
};

pub fn select_dim(src: &Tensor, dim: impl Into<Scalar>, index: impl Into<Scalar>) -> Result<Tensor> {
    let dim_i32 = dim.into().as_i32();
    let index_i32 = index.into().as_i32();

    let dim: usize = if dim_i32 < 0 {
        (src.ndim() as i32 + dim_i32) as usize
    } else {
        dim_i32 as usize
    };

    if dim >= src.ndim() {
        return Err(Error::DimensionOutOfBounds {
            dim: dim as i32,
            ndim: src.ndim(),
        });
    }

    let start = index_i32;
    let end = Some(index_i32 + 1);
    let step = 1;

    let sliced = src.slice(dim, start, end, step)?;
    sliced.squeeze(dim)
}

pub fn get_index(src: &Tensor, indices: &[usize]) -> Result<Scalar> {
    if indices.len() != src.ndim() {
        return Err(Error::InvalidArgument(format!(
            "Indices length ({}) must match tensor dimensions ({})",
            indices.len(),
            src.ndim()
        )));
    }

    let mut usize_indices = Vec::with_capacity(indices.len());
    for (dim, &idx) in indices.iter().enumerate() {
        let idx_i32 = idx as i32;
        let dim_size = src.dim_size(dim).unwrap_or(0) as i32;

        let actual_idx = if idx_i32 < 0 { dim_size + idx_i32 } else { idx_i32 };

        if actual_idx < 0 || actual_idx >= dim_size {
            return Err(Error::IndexOutOfBounds {
                index: actual_idx as usize,
                size: dim_size as usize,
            });
        }

        usize_indices.push(actual_idx as usize);
    }

    let mut flat_idx = 0;
    for (dim, &idx) in usize_indices.iter().enumerate() {
        flat_idx += idx * src.strides()[dim];
    }

    flat_idx += src.offset();
    src.buffer().read_scalar(flat_idx)
}

pub fn set_index(src: &mut Tensor, indices: &[usize], data: impl Into<Scalar>) -> Result<()> {
    if indices.len() != src.ndim() {
        return Err(Error::InvalidArgument(format!(
            "Indices length ({}) must match tensor dimensions ({})",
            indices.len(),
            src.ndim()
        )));
    }

    for (dim, &idx) in indices.iter().enumerate() {
        if idx >= src.shape()[dim] {
            return Err(Error::IndexOutOfBounds {
                index: idx,
                size: src.shape()[dim],
            });
        }
    }

    let mut flat_idx = 0;
    for (dim, &idx) in indices.iter().enumerate() {
        flat_idx += idx * src.strides()[dim];
    }

    flat_idx += src.offset();

    let scalar = data.into();
    src.buffer_mut().write_scalar(flat_idx, scalar)
}

pub fn add_at_index(src: &mut Tensor, indices: &[usize], data: impl Into<Scalar>) -> Result<()> {
    if indices.len() != src.ndim() {
        return Err(Error::InvalidArgument(format!(
            "Indices length ({}) must match tensor dimensions ({})",
            indices.len(),
            src.ndim()
        )));
    }

    for (dim, &idx) in indices.iter().enumerate() {
        if idx >= src.shape()[dim] {
            return Err(Error::IndexOutOfBounds {
                index: idx,
                size: src.shape()[dim],
            });
        }
    }

    let mut flat_idx = 0;
    for (dim, &idx) in indices.iter().enumerate() {
        flat_idx += idx * src.strides()[dim];
    }

    flat_idx += src.offset();

    let add_value = data.into();
    let dtype = src.dtype();

    let current_value = src.buffer().read_scalar(flat_idx)?;

    let new_value = match dtype {
        maidenx_core::dtype::DType::BF16 => Scalar::from(current_value.as_bf16() + add_value.as_bf16()),
        maidenx_core::dtype::DType::F16 => Scalar::from(current_value.as_f16() + add_value.as_f16()),
        maidenx_core::dtype::DType::F32 => Scalar::from(current_value.as_f32() + add_value.as_f32()),
        maidenx_core::dtype::DType::F64 => Scalar::from(current_value.as_f64() + add_value.as_f64()),
        maidenx_core::dtype::DType::BOOL => Scalar::from(current_value.as_bool() || add_value.as_bool()),
        maidenx_core::dtype::DType::U8 => Scalar::from(current_value.as_u8() + add_value.as_u8()),
        maidenx_core::dtype::DType::U16 => Scalar::from(current_value.as_u16() + add_value.as_u16()),
        maidenx_core::dtype::DType::U32 => Scalar::from(current_value.as_u32() + add_value.as_u32()),
        maidenx_core::dtype::DType::U64 => Scalar::from(current_value.as_u64() + add_value.as_u64()),
        maidenx_core::dtype::DType::I8 => Scalar::from(current_value.as_i8() + add_value.as_i8()),
        maidenx_core::dtype::DType::I16 => Scalar::from(current_value.as_i16() + add_value.as_i16()),
        maidenx_core::dtype::DType::I32 => Scalar::from(current_value.as_i32() + add_value.as_i32()),
        maidenx_core::dtype::DType::I64 => Scalar::from(current_value.as_i64() + add_value.as_i64()),
    };

    src.buffer_mut().write_scalar(flat_idx, new_value)
}
