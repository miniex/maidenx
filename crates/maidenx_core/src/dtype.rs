#![allow(non_upper_case_globals)]

use crate::scalar::Scalar;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use half::{bf16, f16};
pub const bfloat16: DType = DType::BF16;
pub const float16: DType = DType::F16;
pub const half: DType = DType::F16;
pub const float32: DType = DType::F32;
pub const float64: DType = DType::F64;
pub const bool: DType = DType::BOOL;
pub const uint8: DType = DType::U8;
pub const uint16: DType = DType::U16;
pub const uint32: DType = DType::U32;
pub const uint64: DType = DType::U64;
pub const int8: DType = DType::I8;
pub const int16: DType = DType::I16;
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
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
}

#[cfg(feature = "serde")]
impl Serialize for DType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for DType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DTypeVisitor;

        impl serde::de::Visitor<'_> for DTypeVisitor {
            type Value = DType;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string representing a DType")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "bf16" => Ok(DType::BF16),
                    "f16" => Ok(DType::F16),
                    "f32" => Ok(DType::F32),
                    "f64" => Ok(DType::F64),
                    "bool" => Ok(DType::BOOL),
                    "u8" => Ok(DType::U8),
                    "u16" => Ok(DType::U16),
                    "u32" => Ok(DType::U32),
                    "u64" => Ok(DType::U64),
                    "i8" => Ok(DType::I8),
                    "i16" => Ok(DType::I16),
                    "i32" => Ok(DType::I32),
                    "i64" => Ok(DType::I64),
                    _ => Err(E::custom(format!("unknown DType: {}", value))),
                }
            }
        }

        deserializer.deserialize_str(DTypeVisitor)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
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
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
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
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }

    #[allow(clippy::match_like_matches_macro)]
    pub fn is_uint(&self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => true,
            _ => false,
        }
    }

    pub fn is_bool(&self) -> bool {
        *self == DType::BOOL
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::BOOL => false,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::I8 | Self::I16 | Self::I32 | Self::I64 => true,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 => true,
            Self::BOOL | Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::I8 | Self::I16 | Self::I32 | Self::I64 => false,
        }
    }

    /// # Safety
    ///
    /// This function performs an unsafe read operation from the provided pointer.
    /// The caller must ensure that:
    /// - The pointer is valid and properly aligned for the target data type
    /// - The pointer points to memory of sufficient size for the data type being read
    /// - The memory location is initialized with valid data for the specified DType
    /// - The lifetime of the pointed memory is valid for the duration of this function call
    pub unsafe fn read_scalar(&self, ptr: *const u8) -> Scalar {
        match self {
            Self::BF16 => {
                let val = *(ptr as *const half::bf16);
                Scalar::from(f32::from(val))
            }
            Self::F16 => {
                let val = *(ptr as *const half::f16);
                Scalar::from(f32::from(val))
            }
            Self::F32 => {
                let val = *(ptr as *const f32);
                Scalar::from(val)
            }
            Self::F64 => {
                let val = *(ptr as *const f64);
                Scalar::from(val)
            }
            Self::BOOL => {
                let val = *(ptr as *const bool);
                Scalar::from(val)
            }
            Self::U8 => {
                let val = *ptr;
                Scalar::from(val)
            }
            Self::U16 => {
                let val = *(ptr as *const u16);
                Scalar::from(val)
            }
            Self::U32 => {
                let val = *(ptr as *const u32);
                Scalar::from(val as i32)
            }
            Self::U64 => {
                let val = *(ptr as *const u64);
                Scalar::from(val)
            }
            Self::I8 => {
                let val = *(ptr as *const i8);
                Scalar::from(val)
            }
            Self::I16 => {
                let val = *(ptr as *const i16);
                Scalar::from(val)
            }
            Self::I32 => {
                let val = *(ptr as *const i32);
                Scalar::from(val)
            }
            Self::I64 => {
                let val = *(ptr as *const i64);
                Scalar::from(val)
            }
        }
    }

    /// # Safety
    ///
    /// This function performs an unsafe write operation to the provided pointer.
    /// The caller must ensure that:
    /// - The pointer is valid and properly aligned for the target data type
    /// - The pointer points to writable memory of sufficient size for the data type
    /// - No other references to this memory location are being accessed concurrently
    /// - The lifetime of the pointed memory is valid for the duration of this function call
    pub unsafe fn write_scalar(&self, ptr: *mut u8, value: Scalar) {
        match self {
            Self::BF16 => {
                let float_val = value.as_f32();
                *(ptr as *mut half::bf16) = half::bf16::from_f32(float_val);
            }
            Self::F16 => {
                let float_val = value.as_f32();
                *(ptr as *mut half::f16) = half::f16::from_f32(float_val);
            }
            Self::F32 => {
                *(ptr as *mut f32) = value.as_f32();
            }
            Self::F64 => {
                *(ptr as *mut f64) = value.as_f64();
            }
            Self::BOOL => {
                *(ptr as *mut bool) = value.as_bool();
            }
            Self::U8 => {
                *ptr = value.as_u8();
            }
            Self::U16 => {
                *(ptr as *mut u16) = value.as_u16();
            }
            Self::U32 => {
                *(ptr as *mut u32) = value.as_u32();
            }
            Self::U64 => {
                *(ptr as *mut u64) = value.as_u64();
            }
            Self::I8 => {
                *(ptr as *mut i8) = value.as_i8();
            }
            Self::I16 => {
                *(ptr as *mut i16) = value.as_i16();
            }
            Self::I32 => {
                *(ptr as *mut i32) = value.as_i32();
            }
            Self::I64 => {
                *(ptr as *mut i64) = value.as_i64();
            }
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

