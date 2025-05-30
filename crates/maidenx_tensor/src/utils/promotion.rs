use crate::Tensor;
#[cfg(feature = "mps")]
use maidenx_core::device::Device;
use maidenx_core::{dtype::DType, error::Result, scalar::Scalar};

pub fn get_signed_dtype(dtype: DType) -> DType {
    match dtype {
        DType::U8 => DType::I32,
        DType::U16 => DType::I32,
        DType::U32 => DType::I64,
        DType::U64 => DType::I64,
        _ => dtype,
    }
}

pub fn get_promoted_dtype(dtype1: DType, dtype2: DType) -> DType {
    let mut dtype1 = dtype1;
    let mut dtype2 = dtype2;

    if dtype1 == DType::BOOL {
        dtype1 = DType::U8;
    }
    if dtype2 == DType::BOOL {
        dtype2 = DType::U8;
    }

    match (dtype1, dtype2) {
        (dtype1, dtype2) if dtype1 == dtype2 => dtype1,

        (_, DType::F64) | (DType::F64, _) => DType::F64,
        (_, DType::F32) | (DType::F32, _) => DType::F32,
        (DType::BF16, DType::F16) | (DType::F16, DType::BF16) => DType::F32,
        (_, DType::F16) | (DType::F16, _) => DType::F16,
        (_, DType::BF16) | (DType::BF16, _) => DType::BF16,

        (_, DType::I64) | (DType::I64, _) => DType::I64,
        (_, DType::I32) | (DType::I32, _) => DType::I32,
        (_, DType::I16) | (DType::I16, _) => DType::I16,
        (_, DType::I8) | (DType::I8, _) => DType::I8,
        (_, DType::U64) | (DType::U64, _) => DType::I64,
        (_, DType::U32) | (DType::U32, _) => DType::I64,
        (_, DType::U16) | (DType::U16, _) => DType::I32,
        (_, DType::U8) | (DType::U8, _) => DType::I32,

        _ => dtype1,
    }
}

// pub fn get_promoted_dtype_with_scalar(tensor_dtype: DType, scalar_dtype: DType) -> DType {
//     let tensor_dtype = if tensor_dtype == DType::BOOL { DType::U8 } else { tensor_dtype };
//
//     let scalar_dtype = if scalar_dtype == DType::BOOL { DType::U8 } else { scalar_dtype };
//
//     match (tensor_dtype, scalar_dtype) {
//         (tensor_dtype, scalar_dtype) if tensor_dtype == scalar_dtype => tensor_dtype,
//
//         (t_dtype, s_dtype) if t_dtype.is_int() && s_dtype.is_float() => s_dtype,
//         _ => tensor_dtype,
//     }
// }

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

pub fn promote_scalar_for_tensor(src: Scalar, target_dtype: DType, with_tensor: &Tensor) -> Result<Scalar> {
    let src = if src.dtype() != target_dtype {
        let src = src.to_dtype(target_dtype);

        src
    } else {
        src.clone()
    };

    #[cfg(feature = "mps")]
    let src = if with_tensor.device() == Device::MPS {
        match src.dtype() {
            DType::U64 => src.to_dtype(DType::U32),
            DType::I64 => src.to_dtype(DType::I32),
            DType::F64 => src.to_dtype(DType::F32),
            _ => src,
        }
    } else {
        src
    };

    #[cfg(not(feature = "mps"))]
    let _ = with_tensor;

    Ok(src)
}
