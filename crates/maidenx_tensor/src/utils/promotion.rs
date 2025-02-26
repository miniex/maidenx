use maidenx_core::{dtype::DType, error::Result};

use crate::Tensor;

pub fn get_promoted_dtype(dtype1: DType, dtype2: DType) -> DType {
    match (dtype1, dtype2) {
        (dtype1, dtype2) if dtype1 == dtype2 => dtype1,

        (_, DType::F64) | (DType::F64, _) => DType::F64,
        (_, DType::F32) | (DType::F32, _) => DType::F32,
        (DType::BF16, DType::F16) | (DType::F16, DType::BF16) => DType::F32,
        (_, DType::F16) | (DType::F16, _) => DType::F16,
        (_, DType::BF16) | (DType::BF16, _) => DType::BF16,

        (_, DType::I64) | (DType::I64, _) => DType::I64,
        (_, DType::I32) | (DType::I32, _) => DType::I32,
        (_, DType::I8) | (DType::I8, _) => DType::I8,
        (_, DType::U32) | (DType::U32, _) => DType::I64,
        (_, DType::U8) | (DType::U8, _) => DType::I32,
        _ => dtype1,
    }
}

pub fn promote_tensor(src: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let src = if src.dtype() != target_dtype {
        let mut src = src.clone();
        src.with_dtype(target_dtype)?;

        src
    } else {
        src.clone()
    };

    Ok(src)
}
