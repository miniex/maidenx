#![allow(clippy::needless_range_loop)]

use crate::Tensor;
use maidenx_core::{
    error::{Error, Result},
    scalar::Scalar,
};

pub fn index_put(dest: &mut Tensor, indices: &[usize], src: &Tensor) -> Result<()> {
    // Validate that the indices specify a valid position in the destination tensor
    if indices.len() > dest.ndim() {
        return Err(Error::InvalidArgument(format!(
            "Indices length ({}) exceeds destination tensor dimensions ({})",
            indices.len(),
            dest.ndim()
        )));
    }

    // Check if we are indexing a subset of dimensions
    let remaining_dims = dest.ndim() - indices.len();

    // If we're not indexing all dimensions, the source tensor should match
    // the remaining dimensions of the destination
    if remaining_dims > 0 {
        // Get the shape of the remaining dimensions in the destination tensor
        let mut dest_remaining_shape = Vec::with_capacity(remaining_dims);
        for i in indices.len()..dest.ndim() {
            dest_remaining_shape.push(dest.dim_size(i).unwrap_or(0));
        }

        // Check if source tensor shape matches the remaining dimensions
        if src.ndim() != remaining_dims {
            return Err(Error::InvalidArgument(format!(
                "Source tensor dimensions ({}) don't match destination's remaining dimensions ({})",
                src.ndim(),
                remaining_dims
            )));
        }

        for i in 0..remaining_dims {
            if src.dim_size(i).unwrap_or(0) != dest_remaining_shape[i] {
                return Err(Error::InvalidArgument(format!(
                    "Dimension mismatch at index {}: source has size {}, destination expects {}",
                    i,
                    src.dim_size(i).unwrap_or(0),
                    dest_remaining_shape[i]
                )));
            }
        }

        // Calculate the base offset in the destination tensor
        let mut base_offset = 0;
        for (dim, &idx) in indices.iter().enumerate() {
            let dim_size = dest.dim_size(dim).unwrap_or(0);
            if idx >= dim_size {
                return Err(Error::IndexOutOfBounds { index: idx, size: dim_size });
            }
            base_offset += idx * dest.strides()[dim];
        }

        fn copy_elements(
            src: &Tensor,
            dest: &mut Tensor,
            base_offset: usize,
            base_dims: usize,
            curr_indices: &mut Vec<usize>,
            curr_dim: usize,
        ) -> Result<()> {
            if curr_dim == src.ndim() {
                let mut dest_flat_idx = base_offset;
                for dim in 0..curr_indices.len() {
                    dest_flat_idx += curr_indices[dim] * dest.strides()[base_dims + dim];
                }

                // Calculate the source flat index
                let mut src_flat_idx = src.offset();
                for dim in 0..curr_indices.len() {
                    src_flat_idx += curr_indices[dim] * src.strides()[dim];
                }

                // Read from source and write to destination
                let value = src.buffer().read_scalar(src_flat_idx)?;
                dest.with_buffer_mut(|buffer| buffer.write_scalar(dest_flat_idx, value))?;

                return Ok(());
            }

            // Recursively iterate through all possible values for the current dimension
            let dim_size = src.dim_size(curr_dim).unwrap_or(0);
            for i in 0..dim_size {
                curr_indices.push(i);
                copy_elements(src, dest, base_offset, base_dims, curr_indices, curr_dim + 1)?;
                curr_indices.pop();
            }

            Ok(())
        }

        let mut curr_indices = Vec::new();
        copy_elements(src, dest, base_offset + dest.offset(), indices.len(), &mut curr_indices, 0)?;
    } else {
        // If we're indexing all dimensions (or the destination is 1D)
        // We can simply set the scalar value from the source
        if src.ndim() == 0 {
            // Source is a scalar
            let scalar = src.buffer().read_scalar(src.offset())?;
            set_index(dest, indices, scalar)?;
        } else if src.size() > 0 {
            // Source is a 1D tensor or has flattened elements that we need to copy over
            let dest_dim = indices[0];
            if dest_dim + src.size() > dest.size() {
                return Err(Error::InvalidArgument(format!(
                    "Source tensor size ({}) exceeds available space in destination starting at index {} (available: {})",
                    src.size(),
                    dest_dim,
                    dest.size() - dest_dim
                )));
            }

            // Copy each element from source to the destination
            for i in 0..src.size() {
                let value = src.buffer().read_scalar(src.offset() + i * src.strides()[0])?;
                let dest_idx = vec![dest_dim + i]; // For a 1D tensor
                set_index(dest, &dest_idx, value)?;
            }
        }
    }

    Ok(())
}

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
    src.with_buffer_mut(|buffer| buffer.write_scalar(flat_idx, scalar))
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
        maidenx_core::dtype::DType::F32 => Scalar::from(current_value.as_f32() + add_value.as_f32()),
        maidenx_core::dtype::DType::F64 => Scalar::from(current_value.as_f64() + add_value.as_f64()),
        maidenx_core::dtype::DType::I32 => Scalar::from(current_value.as_i32() + add_value.as_i32()),
        maidenx_core::dtype::DType::I64 => Scalar::from(current_value.as_i64() + add_value.as_i64()),
        maidenx_core::dtype::DType::U8 => Scalar::from(current_value.as_u8() + add_value.as_u8()),
        maidenx_core::dtype::DType::I8 => Scalar::from(current_value.as_i8() + add_value.as_i8()),
        maidenx_core::dtype::DType::U32 => Scalar::from(current_value.as_u32() + add_value.as_u32()),
        maidenx_core::dtype::DType::BOOL => Scalar::from(current_value.as_bool() || add_value.as_bool()),
        maidenx_core::dtype::DType::BF16 => Scalar::from(current_value.as_bf16() + add_value.as_bf16()),
        maidenx_core::dtype::DType::F16 => Scalar::from(current_value.as_f16() + add_value.as_f16()),
    };

    src.with_buffer_mut(|buffer| buffer.write_scalar(flat_idx, new_value))
}
