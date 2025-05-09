use crate::dtype::DType;
use half::{bf16, f16};
#[cfg(feature = "serde")]
use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;
#[cfg(feature = "serde")]
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BOOL(v) => write!(f, "{}", v),
            Self::BF16(v) => write!(f, "{:.8}", f32::from(*v)),
            Self::F16(v) => write!(f, "{:.8}", f32::from(*v)),
            Self::F32(v) => write!(f, "{:.8}", v),
            Self::F64(v) => write!(f, "{:.8}", v),
            Self::U8(v) => write!(f, "{}", v),
            Self::U16(v) => write!(f, "{}", v),
            Self::U32(v) => write!(f, "{}", v),
            Self::U64(v) => write!(f, "{}", v),
            Self::I8(v) => write!(f, "{}", v),
            Self::I16(v) => write!(f, "{}", v),
            Self::I32(v) => write!(f, "{}", v),
            Self::I64(v) => write!(f, "{}", v),
        }
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BOOL(v) => write!(f, "Scalar(dtype=BOOL, value={})", v),
            Self::BF16(v) => write!(f, "Scalar(dtype=BF16, value={:.8})", f32::from(*v)),
            Self::F16(v) => write!(f, "Scalar(dtype=F16, value={:.8})", f32::from(*v)),
            Self::F32(v) => write!(f, "Scalar(dtype=F32, value={:.8})", v),
            Self::F64(v) => write!(f, "Scalar(dtype=F64, value={:.8})", v),
            Self::U8(v) => write!(f, "Scalar(dtype=U8, value={})", v),
            Self::U16(v) => write!(f, "Scalar(dtype=U16, value={})", v),
            Self::U32(v) => write!(f, "Scalar(dtype=U32, value={})", v),
            Self::U64(v) => write!(f, "Scalar(dtype=U64, value={})", v),
            Self::I8(v) => write!(f, "Scalar(dtype=I8, value={})", v),
            Self::I16(v) => write!(f, "Scalar(dtype=I16, value={})", v),
            Self::I32(v) => write!(f, "Scalar(dtype=I32, value={})", v),
            Self::I64(v) => write!(f, "Scalar(dtype=I64, value={})", v),
        }
    }
}

macro_rules! numeric_variants {
    ($($variant:ident => $type:ty),* $(,)?) => {
        #[derive(Clone, Copy, PartialEq)]
        pub enum Scalar {
            BOOL(bool),
            $($variant($type),)*
        }

        impl Scalar {
            #[inline]
            pub fn new<T: Into<Self>>(value: T) -> Self {
                value.into()
            }

            #[inline]
            pub fn dtype(&self) -> DType {
                match self {
                    Self::BOOL(_) => DType::BOOL,
                    $(Self::$variant(_) => DType::$variant,)*
                }
            }

            #[inline]
            pub fn to_dtype(&self, dtype: DType) -> Scalar {
                if self.dtype() == dtype {
                    return *self;
                }

                match dtype {
                    DType::BOOL => Scalar::BOOL(self.as_bool()),
                    DType::BF16 => Scalar::BF16(bf16::from_f32(self.as_f32())),
                    DType::F16 => Scalar::F16(f16::from_f32(self.as_f32())),
                    DType::F32 => Scalar::F32(self.as_f32()),
                    DType::F64 => Scalar::F64(self.as_f64()),
                    DType::U8 => Scalar::U8(self.as_u8()),
                    DType::U16 => Scalar::U16(self.as_u16()),
                    DType::U32 => Scalar::U32(self.as_u32()),
                    DType::U64 => Scalar::U64(self.as_u64()),
                    DType::I8 => Scalar::I8(self.as_i8()),
                    DType::I16 => Scalar::I16(self.as_i16()),
                    DType::I32 => Scalar::I32(self.as_i32()),
                    DType::I64 => Scalar::I64(self.as_i64()),
                }
            }

            #[inline]
            pub fn is_int(&self) -> bool {
                match self {
                    Self::BF16(_) | Self::F16(_) | Self::F32(_) | Self::F64(_) | Self::BOOL(_) => false,
                    Self::U8(_) | Self::U16(_) | Self::U32(_) | Self::U64(_) | Self::I8(_) | Self::I16(_) | Self::I32(_) | Self::I64(_) => true,
                }
            }

            #[inline]
            pub fn is_float(&self) -> bool {
                match self {
                    Self::BF16(_) | Self::F16(_) | Self::F32(_) | Self::F64(_) => true,
                    Self::BOOL(_) | Self::U8(_) | Self::U16(_) | Self::U32(_) | Self::U64(_) | Self::I8(_) | Self::I16(_) | Self::I32(_) | Self::I64(_) => false,
                }
            }

            #[inline]
            pub fn is_integer_value(&self) -> bool {
                if self.is_int() {
                    return true;
                }

                match self {
                    Self::BF16(v) => {
                        let f = f32::from(*v);
                        f == f.trunc()
                    },
                    Self::F16(v) => {
                        let f = f32::from(*v);
                        f == f.trunc()
                    },
                    Self::F32(v) => *v == v.trunc(),
                    Self::F64(v) => *v == v.trunc(),
                    _ => self.is_int(),
                }
            }

            #[inline]
            pub fn as_f64_any(&self) -> f64 {
                match *self {
                    Self::BOOL(x) => if x { 1.0 } else { 0.0 },
                    $(
                        Self::$variant(x) => {
                            numeric_variants!(@as_f64 $variant, x)
                        },
                    )*
                }
            }

            $(
                paste::paste! {
                    #[inline]
                    pub fn [<as_ $variant:lower>](&self) -> $type {
                        match *self {
                            Self::$variant(x) => x,
                            _ => numeric_variants!(@convert $variant => self.as_f64_any()),
                        }
                    }
                }
            )*

            #[inline]
            pub fn sqrt(self) -> Self {
                match self {
                    Self::BOOL(x) => Self::BOOL(x),
                    $(
                        Self::$variant(x) => {
                            numeric_variants!(@sqrt $variant, x)
                        }
                    ),*
                }
            }

            #[inline]
            pub fn as_bool(&self) -> bool {
                match *self {
                    Self::BF16(x) => f32::from(x) != 0.0,
                    Self::F16(x) => f32::from(x) != 0.0,
                    Self::F32(x) => x != 0.0,
                    Self::F64(x) => x != 0.0,
                    Self::BOOL(x) => x,
                    Self::U8(x) => x != 0,
                    Self::U16(x) => x != 0,
                    Self::U32(x) => x != 0,
                    Self::U64(x) => x != 0,
                    Self::I8(x) => x != 0,
                    Self::I16(x) => x != 0,
                    Self::I32(x) => x != 0,
                    Self::I64(x) => x != 0,
                }
            }
        }

        impl From<bool> for Scalar {
            #[inline]
            fn from(x: bool) -> Self {
                Self::BOOL(x)
            }
        }

        $(
            impl From<$type> for Scalar {
                #[inline]
                fn from(x: $type) -> Self {
                    Self::$variant(x)
                }
            }
        )*

        impl Add for Scalar {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a || b),
                    $(
                        (Self::$variant(a), Self::$variant(b)) => Self::$variant(a + b),
                    )*
                    (lhs, rhs) => Self::F64(lhs.as_f64_any() + rhs.as_f64_any()),
                }
            }
        }

        impl Sub for Scalar {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (Self::BOOL(_), Self::BOOL(_)) => Self::BOOL(false),
                    $(
                        (Self::$variant(a), Self::$variant(b)) => Self::$variant(a - b),
                    )*
                    (lhs, rhs) => Self::F64(lhs.as_f64_any() - rhs.as_f64_any()),
                }
            }
        }

        impl Mul for Scalar {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a && b),
                    $(
                        (Self::$variant(a), Self::$variant(b)) => Self::$variant(a * b),
                    )*
                    (lhs, rhs) => Self::F64(lhs.as_f64_any() * rhs.as_f64_any()),
                }
            }
        }

        impl Div for Scalar {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (Self::BOOL(_), Self::BOOL(_)) => Self::BOOL(false),
                    $(
                        (Self::$variant(a), Self::$variant(b)) => Self::$variant(a / b),
                    )*
                    (lhs, rhs) => Self::F64(lhs.as_f64_any() / rhs.as_f64_any()),
                }
            }
        }

        impl Scalar {
            #[inline]
            pub fn powi(self, exp: i32) -> Self {
                match self {
                    Self::BOOL(a) => {
                        if exp == 0 {
                            Self::BOOL(true)
                        } else {
                            Self::BOOL(a)
                        }
                    },
                    Self::BF16(a) => Self::BF16(bf16::from_f32(a.to_f32().powi(exp))),
                    Self::F16(a) => Self::F16(f16::from_f32(a.to_f32().powi(exp))),
                    Self::F32(a) => Self::F32(a.powi(exp)),
                    Self::F64(a) => Self::F64(a.powi(exp)),
                    Self::U8(a) => {
                        if exp == 0 {
                            Self::U8(1)
                        } else if exp < 0 {
                            Self::F32((a as f32).powi(exp))
                        } else {
                            let result = (a as u32).pow(exp as u32);
                            if result <= u8::MAX as u32 {
                                Self::U8(result as u8)
                            } else {
                                Self::F32((a as f32).powi(exp))
                            }
                        }
                    },
                    Self::U16(a) => {
                        if exp == 0 {
                            Self::U16(1)
                        } else if exp < 0 {
                            Self::F32((a as f32).powi(exp))
                        } else {
                            let result = (a as u32).pow(exp as u32);
                            if result <= u16::MAX as u32 {
                                Self::U16(result as u16)
                            } else {
                                Self::F32((a as f32).powi(exp))
                            }
                        }
                    },
                    Self::U32(a) => {
                        if exp == 0 {
                            Self::U32(1)
                        } else if exp < 0 {
                            Self::F32((a as f32).powi(exp))
                        } else {
                            let result = (a as u64).pow(exp as u32);
                            if result <= u32::MAX as u64 {
                                Self::U32(result as u32)
                            } else {
                                Self::F64((a as f64).powi(exp))
                            }
                        }
                    },
                    Self::U64(a) => {
                        if exp == 0 {
                            Self::U64(1)
                        } else if exp < 0 {
                            Self::F64((a as f64).powi(exp))
                        } else {
                            let mut result: u64 = 1;
                            let mut overflow = false;
                            for _ in 0..exp {
                                if let Some(r) = result.checked_mul(a) {
                                    result = r;
                                } else {
                                    overflow = true;
                                    break;
                                }
                            }
                            if overflow {
                                Self::F64((a as f64).powi(exp))
                            } else {
                                Self::U64(result)
                            }
                        }
                    },
                    Self::I8(a) => {
                        if exp == 0 {
                            Self::I8(1)
                        } else if exp < 0 {
                            Self::F32((a as f32).powi(exp))
                        } else {
                            let result = (a as i32).pow(exp as u32);
                            if result >= i8::MIN as i32 && result <= i8::MAX as i32 {
                                Self::I8(result as i8)
                            } else {
                                Self::F32((a as f32).powi(exp))
                            }
                        }
                    },
                    Self::I16(a) => {
                        if exp == 0 {
                            Self::I16(1)
                        } else if exp < 0 {
                            Self::F32((a as f32).powi(exp))
                        } else {
                            let result = (a as i32).pow(exp as u32);
                            if result >= i16::MIN as i32 && result <= i16::MAX as i32 {
                                Self::I16(result as i16)
                            } else {
                                Self::F32((a as f32).powi(exp))
                            }
                        }
                    },
                    Self::I32(a) => {
                        if exp == 0 {
                            Self::I32(1)
                        } else if exp < 0 {
                            Self::F32((a as f32).powi(exp))
                        } else {
                            let result = (a as i64).pow(exp as u32);
                            if result >= i32::MIN as i64 && result <= i32::MAX as i64 {
                                Self::I32(result as i32)
                            } else {
                                Self::F64((a as f64).powi(exp))
                            }
                        }
                    },
                    Self::I64(a) => {
                        if exp == 0 {
                            Self::I64(1)
                        } else if exp < 0 {
                            Self::F64((a as f64).powi(exp))
                        } else {
                            let mut result: i64 = 1;
                            let mut overflow = false;

                            for _ in 0..exp {
                                if let Some(r) = result.checked_mul(a) {
                                    result = r;
                                } else {
                                    overflow = true;
                                    break;
                                }
                            }

                            if overflow {
                                Self::F64((a as f64).powi(exp))
                            } else {
                                Self::I64(result)
                            }
                        }
                    },

                }
            }
        }
    };

    (@as_f64 BF16, $x:ident) => {
        f32::from($x) as f64
    };
    (@as_f64 F16, $x:ident) => {
        f32::from($x) as f64
    };
    (@as_f64 F32, $x:ident) => {
        $x as f64
    };
    (@as_f64 F64, $x:ident) => {
        $x
    };
    (@as_f64 U8,  $x:ident) => {
        $x as f64
    };
    (@as_f64 U16, $x:ident) => {
        $x as f64
    };
    (@as_f64 U32, $x:ident) => {
        $x as f64
    };
    (@as_f64 U64, $x:ident) => {
        $x as f64
    };
    (@as_f64 I8,  $x:ident) => {
        $x as f64
    };
    (@as_f64 I16, $x:ident) => {
        $x as f64
    };
    (@as_f64 I32, $x:ident) => {
        $x as f64
    };
    (@as_f64 I64, $x:ident) => {
        $x as f64
    };

    (@convert BF16 => $val:expr) => {
        bf16::from_f32($val as f32)
    };
    (@convert F16 => $val:expr) => {
        f16::from_f32($val as f32)
    };
    (@convert F32 => $val:expr) => {
        $val as f32
    };
    (@convert F64 => $val:expr) => {
        $val
    };
    (@convert U8  => $val:expr) => {
        $val.clamp(0.0, u8::MAX as f64) as u8
    };
    (@convert U16 => $val:expr) => {
        $val.clamp(0.0, u16::MAX as f64) as u16
    };
    (@convert U64 => $val:expr) => {
        $val.clamp(0.0, u64::MAX as f64) as u64
    };
    (@convert U32 => $val:expr) => {
        $val.clamp(0.0, u32::MAX as f64) as u32
    };
    (@convert I8  => $val:expr) => {
        $val.clamp(i8::MIN as f64, i8::MAX as f64) as i8
    };
    (@convert I16 => $val:expr) => {
        $val.clamp(i16::MIN as f64, i16::MAX as f64) as i16
    };
    (@convert I32 => $val:expr) => {
        $val.clamp(i32::MIN as f64, i32::MAX as f64) as i32
    };
    (@convert I64 => $val:expr) => {
        $val.clamp(i64::MIN as f64, i64::MAX as f64) as i64
    };

    (@sqrt BF16, $x:ident) => {
        Scalar::BF16(bf16::from_f32(f32::from($x).sqrt()))
    };
    (@sqrt F16, $x:ident) => {
        Scalar::F16(f16::from_f32(f32::from($x).sqrt()))
    };
    (@sqrt F32, $x:ident) => {
        Scalar::F32($x.sqrt())
    };
    (@sqrt F64, $x:ident) => {
        Scalar::F64($x.sqrt())
    };
    (@sqrt U8, $x:ident) => {
        Scalar::U8((($x as f64).sqrt()).clamp(0.0, u8::MAX as f64) as u8)
    };
    (@sqrt U16, $x:ident) => {
        Scalar::U16((($x as f64).sqrt()).clamp(0.0, u16::MAX as f64) as u16)
    };
    (@sqrt U64, $x:ident) => {
        Scalar::U64((($x as f64).sqrt()).clamp(0.0, u64::MAX as f64) as u64)
    };
    (@sqrt U32, $x:ident) => {
        Scalar::U32((($x as f64).sqrt()).clamp(0.0, u32::MAX as f64) as u32)
    };
    (@sqrt I8, $x:ident) => {
        Scalar::I8((($x as f64).sqrt()).clamp(i8::MIN as f64, i8::MAX as f64) as i8)
    };
    (@sqrt I16, $x:ident) => {
        Scalar::I16((($x as f64).sqrt()).clamp(i16::MIN as f64, i16::MAX as f64) as i16)
    };
    (@sqrt I32, $x:ident) => {
        Scalar::I32((($x as f64).sqrt()).clamp(i32::MIN as f64, i32::MAX as f64) as i32)
    };
    (@sqrt I64, $x:ident) => {
        Scalar::I64((($x as f64).sqrt()).clamp(i64::MIN as f64, i64::MAX as f64) as i64)
    };
}

numeric_variants! {
    BF16 => bf16,
    F16  => f16,
    F32  => f32,
    F64  => f64,
    U8   => u8,
    U16  => u16,
    U32  => u32,
    U64  => u64,
    I8   => i8,
    I16  => i16,
    I32  => i32,
    I64  => i64,
}

impl From<usize> for Scalar {
    #[inline]
    fn from(x: usize) -> Self {
        if x <= u32::MAX as usize {
            Scalar::U32(x as u32)
        } else {
            Scalar::F64(x as f64)
        }
    }
}

impl From<isize> for Scalar {
    #[inline]
    fn from(x: isize) -> Self {
        if x >= i32::MIN as isize && x <= i32::MAX as isize {
            Scalar::I32(x as i32)
        } else {
            Scalar::I64(x as i64)
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for Scalar {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Scalar", 2)?;

        match self {
            Self::BOOL(v) => {
                state.serialize_field("dtype", "BOOL")?;
                state.serialize_field("value", v)?;
            },
            Self::BF16(v) => {
                state.serialize_field("dtype", "BF16")?;
                state.serialize_field("value", &f32::from(*v))?;
            },
            Self::F16(v) => {
                state.serialize_field("dtype", "F16")?;
                state.serialize_field("value", &f32::from(*v))?;
            },
            Self::F32(v) => {
                state.serialize_field("dtype", "F32")?;
                state.serialize_field("value", v)?;
            },
            Self::F64(v) => {
                state.serialize_field("dtype", "F64")?;
                state.serialize_field("value", v)?;
            },
            Self::U8(v) => {
                state.serialize_field("dtype", "U8")?;
                state.serialize_field("value", v)?;
            },
            Self::U16(v) => {
                state.serialize_field("dtype", "U16")?;
                state.serialize_field("value", v)?;
            },
            Self::U32(v) => {
                state.serialize_field("dtype", "U32")?;
                state.serialize_field("value", v)?;
            },
            Self::U64(v) => {
                state.serialize_field("dtype", "U64")?;
                state.serialize_field("value", v)?;
            },
            Self::I8(v) => {
                state.serialize_field("dtype", "I8")?;
                state.serialize_field("value", v)?;
            },
            Self::I16(v) => {
                state.serialize_field("dtype", "I16")?;
                state.serialize_field("value", v)?;
            },
            Self::I32(v) => {
                state.serialize_field("dtype", "I32")?;
                state.serialize_field("value", v)?;
            },
            Self::I64(v) => {
                state.serialize_field("dtype", "I64")?;
                state.serialize_field("value", v)?;
            },
        }

        state.end()
    }
}

#[cfg(feature = "serde")]
struct ScalarVisitor {
    marker: PhantomData<fn() -> Scalar>,
}

#[cfg(feature = "serde")]
impl ScalarVisitor {
    fn new() -> Self {
        ScalarVisitor { marker: PhantomData }
    }
}

#[cfg(feature = "serde")]
impl<'de> Visitor<'de> for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a struct representing a Scalar")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut dtype: Option<String> = None;
        let mut value_f64: Option<f64> = None;
        let mut value_bool: Option<bool> = None;
        let mut value_u64: Option<u64> = None;
        let mut value_i64: Option<i64> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "dtype" => {
                    dtype = Some(map.next_value()?);
                },
                "value" => {
                    if let Some(dtype_str) = &dtype {
                        match dtype_str.as_str() {
                            "BOOL" => {
                                value_bool = Some(map.next_value()?);
                            },
                            "BF16" | "F16" | "F32" | "F64" => {
                                value_f64 = Some(map.next_value()?);
                            },
                            "U8" | "U16" | "U32" | "U64" => {
                                value_u64 = Some(map.next_value()?);
                            },
                            "I8" | "I16" | "I32" | "I64" => {
                                value_i64 = Some(map.next_value()?);
                            },
                            _ => {
                                return Err(de::Error::unknown_variant(
                                    dtype_str,
                                    &[
                                        "BOOL", "BF16", "F16", "F32", "F64", "U8", "U16", "U32", "U64", "I8", "I16",
                                        "I32", "I64",
                                    ],
                                ))
                            },
                        }
                    } else {
                        return Err(de::Error::missing_field("dtype"));
                    }
                },
                _ => return Err(de::Error::unknown_field(&key, &["dtype", "value"])),
            }
        }

        let dtype = dtype.ok_or_else(|| de::Error::missing_field("dtype"))?;

        match dtype.as_str() {
            "BOOL" => {
                let value = value_bool.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::BOOL(value))
            },
            "BF16" => {
                let value = value_f64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::BF16(bf16::from_f32(value as f32)))
            },
            "F16" => {
                let value = value_f64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::F16(f16::from_f32(value as f32)))
            },
            "F32" => {
                let value = value_f64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::F32(value as f32))
            },
            "F64" => {
                let value = value_f64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::F64(value))
            },
            "U8" => {
                let value = value_u64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::U8(value as u8))
            },
            "U16" => {
                let value = value_u64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::U16(value as u16))
            },
            "U32" => {
                let value = value_u64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::U32(value as u32))
            },
            "U64" => {
                let value = value_u64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::U64(value))
            },
            "I8" => {
                let value = value_i64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::I8(value as i8))
            },
            "I16" => {
                let value = value_i64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::I16(value as i16))
            },
            "I32" => {
                let value = value_i64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::I32(value as i32))
            },
            "I64" => {
                let value = value_i64.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(Scalar::I64(value))
            },
            _ => Err(de::Error::unknown_variant(
                &dtype,
                &[
                    "BOOL", "BF16", "F16", "F32", "F64", "U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64",
                ],
            )),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("Scalar", &["dtype", "value"], ScalarVisitor::new())
    }
}
