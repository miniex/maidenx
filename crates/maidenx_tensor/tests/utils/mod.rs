#![allow(dead_code)]

use maidenx_core::{
    device::{set_default_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Helper functions
pub fn setup_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(feature = "mps")]
    set_default_device(Device::MPS);
    #[cfg(not(any(feature = "cuda", feature = "mps")))]
    set_default_device(Device::CPU);
}

pub fn setup_tensor_without_dtype<T: Clone + 'static>(data: Vec<T>) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let tensor = Tensor::new(data)?;

    Ok(tensor)
}

pub fn setup_tensor<T: Clone + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let mut tensor = Tensor::new(data)?;
    tensor.with_dtype(dtype)?;

    Ok(tensor)
}

pub fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor(data, dtype)?;
    tensor.with_grad().ok();

    Ok(tensor)
}

pub fn setup_tensor_with_shape<T: Clone + 'static>(data: Vec<T>, dtype: DType, shape: &[usize]) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let mut tensor = Tensor::new(data)?;
    tensor.with_shape(shape)?;
    tensor.with_dtype(dtype)?;

    Ok(tensor)
}

pub fn setup_grad_tensor_with_shape<T: Clone + 'static>(data: Vec<T>, dtype: DType, shape: &[usize]) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor_with_shape(data, dtype, shape)?;
    tensor.with_grad().ok();

    Ok(tensor)
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

                    #[cfg(not(feature = "mps"))]
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

                    #[cfg(not(feature = "mps"))]
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

                    #[cfg(not(feature = "mps"))]
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

                    #[cfg(not(feature = "mps"))]
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

                    #[cfg(not(feature = "mps"))]
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

                    #[cfg(not(feature = "mps"))]
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
macro_rules! test_ops_only_integer {
    ([$($op:ident),*]) => {
        $(
            mod $op {
                use super::*;
                use paste::paste;

                paste! {
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

                    #[cfg(not(feature = "mps"))]
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

                    #[cfg(not(feature = "mps"))]
                    #[test]
                    fn i64() -> Result<()> {
                        test_functions::[<$op _test>](DType::I64)
                    }
                }
            }
        )*
    };
}
