use maidenx_core::dtype::DType;

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
