use maidenx_core::device::{set_default_device, Device};

// Helper functions
pub fn setup_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(feature = "mps")]
    set_default_device(Device::MPS);
    #[cfg(not(any(feature = "cuda", feature = "mps")))]
    set_default_device(Device::CPU);
}

#[macro_export]
macro_rules! test_ops {
    ([$($op:ident),*]) => {
        $(
            mod $op {
                use super::*;
                use paste::paste;

                paste! {
                    #[test]
                    fn bf16() -> Result<()> {
                        test_functions::[<$op _test>](DType::BF16)
                    }

                    #[test]
                    fn f16() -> Result<()> {
                        test_functions::[<$op _test>](DType::F16)
                    }

                    #[test]
                    fn f32() -> Result<()> {
                        test_functions::[<$op _test>](DType::F32)
                    }

                    #[test]
                    fn f64() -> Result<()> {
                        test_functions::[<$op _test>](DType::F64)
                    }

                    #[test]
                    fn u8() -> Result<()> {
                        test_functions::[<$op _test>](DType::U8)
                    }

                    #[test]
                    fn u16() -> Result<()> {
                        test_functions::[<$op _test>](DType::U16)
                    }

                    #[test]
                    fn u32() -> Result<()> {
                        test_functions::[<$op _test>](DType::U32)
                    }

                    #[test]
                    fn u64() -> Result<()> {
                        test_functions::[<$op _test>](DType::U64)
                    }

                    #[test]
                    fn i8() -> Result<()> {
                        test_functions::[<$op _test>](DType::I8)
                    }

                    #[test]
                    fn i16() -> Result<()> {
                        test_functions::[<$op _test>](DType::I16)
                    }

                    #[test]
                    fn i32() -> Result<()> {
                        test_functions::[<$op _test>](DType::I32)
                    }

                    #[test]
                    fn i64() -> Result<()> {
                        test_functions::[<$op _test>](DType::I64)
                    }
                }
            }
        )*
    };
}

#[macro_export]
macro_rules! test_ops_with_dtype {
    ([
        $($op:ident: [$($dtype:ident),*$(,)?]),*$(,)?
    ]) => {
        $(
            mod $op {
                use super::*;
                use paste::paste;
                paste! {
                    $(
                        #[test]
                        fn [<$dtype:lower>]() -> Result<()> {
                            test_functions::[<$op _test>](DType::$dtype)
                        }
                    )*
                }
            }
        )*
    };
}

#[macro_export]
macro_rules! test_logical_ops {
    ([$($op:ident),*]) => {
        $(
            mod $op {
                use super::*;
                use paste::paste;

                paste! {
                    #[test]
                    fn bf16() -> Result<()> {
                        test_functions::[<$op _test>](DType::BF16)
                    }

                    #[test]
                    fn f16() -> Result<()> {
                        test_functions::[<$op _test>](DType::F16)
                    }

                    #[test]
                    fn f32() -> Result<()> {
                        test_functions::[<$op _test>](DType::F32)
                    }

                    #[test]
                    fn f64() -> Result<()> {
                        test_functions::[<$op _test>](DType::F64)
                    }

                    #[test]
                    fn bool() -> Result<()> {
                        test_functions::[<$op _test>](DType::BOOL)
                    }

                    #[test]
                    fn u8() -> Result<()> {
                        test_functions::[<$op _test>](DType::U8)
                    }

                    #[test]
                    fn u16() -> Result<()> {
                        test_functions::[<$op _test>](DType::U16)
                    }

                    #[test]
                    fn u32() -> Result<()> {
                        test_functions::[<$op _test>](DType::U32)
                    }

                    #[test]
                    fn u64() -> Result<()> {
                        test_functions::[<$op _test>](DType::U64)
                    }

                    #[test]
                    fn i8() -> Result<()> {
                        test_functions::[<$op _test>](DType::I8)
                    }

                    #[test]
                    fn i16() -> Result<()> {
                        test_functions::[<$op _test>](DType::I16)
                    }

                    #[test]
                    fn i32() -> Result<()> {
                        test_functions::[<$op _test>](DType::I32)
                    }

                    #[test]
                    fn i64() -> Result<()> {
                        test_functions::[<$op _test>](DType::I64)
                    }
                }
            }
        )*
    };
}
