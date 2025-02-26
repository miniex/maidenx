#![allow(non_upper_case_globals)]

pub const bfloat16: DType = DType::BF16;
pub const float16: DType = DType::F16;
pub const half: DType = DType::F16;
pub const float32: DType = DType::F32;
pub const float64: DType = DType::F64;
pub const bool: DType = DType::BOOL;
pub const uint8: DType = DType::U8;
pub const uint32: DType = DType::U32;
pub const int8: DType = DType::I8;
pub const int32: DType = DType::I32;
pub const int64: DType = DType::I64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    BOOL,
    U8,
    U32,
    I8,
    I32,
    I64,
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::BOOL => "bool",
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I8 => "i8",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::BOOL => 1,
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I8 => 1,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }

    #[allow(clippy::match_like_matches_macro)]
    pub fn is_uint(&self) -> bool {
        match self {
            Self::U8 | Self::U32 => true,
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::BOOL => false,
            Self::U8 | Self::U32 | Self::I8 | Self::I32 | Self::I64 => true,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 => true,
            Self::BOOL | Self::U8 | Self::U32 | Self::I8 | Self::I32 | Self::I64 => false,
        }
    }
}

thread_local! {
    static DEFAULT_DTYPE: std::cell::Cell<DType> = const { std::cell::Cell::new(DType::F32) };
}

pub fn get_default_dtype() -> DType {
    DEFAULT_DTYPE.with(|d| d.get())
}

pub fn set_default_dtype(dtype: DType) {
    DEFAULT_DTYPE.with(|d| d.set(dtype));
}
