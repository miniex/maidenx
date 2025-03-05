use crate::dtype::DType;
use half::{bf16, f16};
use std::ops::{Add, Div, Mul, Sub};

macro_rules! numeric_variants {
    ($($variant:ident => $type:ty),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
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
            pub fn is_int(&self) -> bool {
                match self {
                    Self::BF16(_) | Self::F16(_) | Self::F32(_) | Self::F64(_) | Self::BOOL(_) => false,
                    Self::U8(_) | Self::U32(_) | Self::I8(_) | Self::I32(_) | Self::I64(_) => true,
                }
            }

            #[inline]
            pub fn is_float(&self) -> bool {
                match self {
                    Self::BF16(_) | Self::F16(_) | Self::F32(_) | Self::F64(_) => true,
                    Self::BOOL(_) | Self::U8(_) | Self::U32(_) | Self::I8(_) | Self::I32(_) | Self::I64(_) => false,
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
                    Self::BOOL(x) => x,
                    Self::U8(x) => x != 0,
                    Self::I8(x) => x != 0,
                    Self::U32(x) => x != 0,
                    Self::I32(x) => x != 0,
                    Self::I64(x) => x != 0,
                    Self::F32(x) => x != 0.0,
                    Self::F64(x) => x != 0.0,
                    Self::BF16(x) => f32::from(x) != 0.0,
                    Self::F16(x) => f32::from(x) != 0.0,
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
                    Self::F32(a) => Self::F32(a.powi(exp)),
                    Self::F64(a) => Self::F64(a.powi(exp)),
                    Self::F16(a) => Self::F16(f16::from_f32(a.to_f32().powi(exp))),
                    Self::BF16(a) => Self::BF16(bf16::from_f32(a.to_f32().powi(exp))),
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
    (@as_f64 I8,  $x:ident) => {
        $x as f64
    };
    (@as_f64 U32, $x:ident) => {
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
    (@convert I8  => $val:expr) => {
        $val.clamp(i8::MIN as f64, i8::MAX as f64) as i8
    };
    (@convert U32 => $val:expr) => {
        $val.clamp(0.0, u32::MAX as f64) as u32
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
    (@sqrt I8, $x:ident) => {
        Scalar::I8((($x as f64).sqrt()).clamp(i8::MIN as f64, i8::MAX as f64) as i8)
    };
    (@sqrt U32, $x:ident) => {
        Scalar::U32((($x as f64).sqrt()).clamp(0.0, u32::MAX as f64) as u32)
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
    I8   => i8,
    U32  => u32,
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
