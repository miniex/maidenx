// #![allow(dead_code)]
//
// use maidenx_core::{device::auto_set_device, dtype::DType, error::Result};
// use maidenx_tensor_new::{adapter::TensorAdapter, Tensor};
//
// pub fn setup_tensor_without_dtype<T: Clone + Default + 'static>(data: Vec<T>) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     auto_set_device();
//
//     let mut padded_data = Vec::with_capacity(data.len() + 2);
//     padded_data.push(T::default());
//     padded_data.extend(data);
//     padded_data.push(T::default());
//     let padded_data_len = padded_data.len();
//
//     let mut tensor = Tensor::new(padded_data)?;
//     tensor = tensor.slice(0, 1, Some(padded_data_len - 1), 1)?;
//
//     Ok(tensor)
// }
//
// pub fn setup_tensor<T: Clone + Default + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     auto_set_device();
//
//     let mut padded_data = Vec::with_capacity(data.len() + 2);
//     padded_data.push(T::default());
//     padded_data.extend(data);
//     padded_data.push(T::default());
//     let padded_data_len = padded_data.len();
//
//     let mut tensor = Tensor::new(padded_data)?;
//     tensor = tensor.slice(0, 1, Some(padded_data_len - 1), 1)?;
//     tensor.with_dtype(dtype)?;
//
//     Ok(tensor)
// }
//
// pub fn setup_grad_tensor<T: Clone + Default + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     let mut tensor = setup_tensor(data, dtype)?;
//     tensor.with_grad().ok();
//
//     Ok(tensor)
// }
//
// pub fn setup_tensor_with_shape<T: Clone + Default + 'static>(
//     data: Vec<T>,
//     dtype: DType,
//     shape: &[usize],
// ) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     auto_set_device();
//
//     let mut padded_data = Vec::with_capacity(data.len() + 2);
//     padded_data.push(T::default());
//     padded_data.extend(data);
//     padded_data.push(T::default());
//     let padded_data_len = padded_data.len();
//
//     let mut tensor = Tensor::new(padded_data)?;
//     tensor = tensor.slice(0, 1, Some(padded_data_len - 1), 1)?;
//     tensor.with_dtype(dtype)?;
//     tensor.with_shape(shape)?;
//
//     Ok(tensor)
// }
//
// pub fn setup_grad_tensor_with_shape<T: Clone + Default + 'static>(
//     data: Vec<T>,
//     dtype: DType,
//     shape: &[usize],
// ) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     let mut tensor = setup_tensor_with_shape(data, dtype, shape)?;
//     tensor.with_grad().ok();
//
//     Ok(tensor)
// }
//
// pub fn setup_contiguous_tensor_with_shape<T: Clone + Default + 'static>(
//     data: Vec<T>,
//     dtype: DType,
//     shape: &[usize],
// ) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     auto_set_device();
//
//     let mut tensor = Tensor::new(data)?;
//     tensor.with_dtype(dtype)?;
//     tensor.with_shape(shape)?;
//
//     Ok(tensor)
// }
//
// pub fn setup_grad_contiguous_tensor_with_shape<T: Clone + Default + 'static>(
//     data: Vec<T>,
//     dtype: DType,
//     shape: &[usize],
// ) -> Result<Tensor>
// where
//     Vec<T>: TensorAdapter,
// {
//     let mut tensor = setup_contiguous_tensor_with_shape(data, dtype, shape)?;
//     tensor.with_grad().ok();
//
//     Ok(tensor)
// }
//
// #[macro_export]
// macro_rules! test_ops {
//     ([$($op:ident),*]) => {
//         $(
//             mod $op {
//                 use super::*;
//                 use paste::paste;
//
//                 paste! {
//                     #[test]
//                     fn bf16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::BF16)
//                     }
//
//                     #[test]
//                     fn f16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::F16)
//                     }
//
//                     #[test]
//                     fn f32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::F32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn f64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::F64)
//                     }
//
//                     #[test]
//                     fn u8() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U8)
//                     }
//
//                     #[test]
//                     fn u16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U16)
//                     }
//
//                     #[test]
//                     fn u32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn u64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U64)
//                     }
//
//                     #[test]
//                     fn i8() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I8)
//                     }
//
//                     #[test]
//                     fn i16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I16)
//                     }
//
//                     #[test]
//                     fn i32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn i64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I64)
//                     }
//                 }
//             }
//         )*
//     };
// }
//
// #[macro_export]
// macro_rules! test_logical_ops {
//     ([$($op:ident),*]) => {
//         $(
//             mod $op {
//                 use super::*;
//                 use paste::paste;
//
//                 paste! {
//                     #[test]
//                     fn bf16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::BF16)
//                     }
//
//                     #[test]
//                     fn f16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::F16)
//                     }
//
//                     #[test]
//                     fn f32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::F32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn f64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::F64)
//                     }
//
//                     #[test]
//                     fn bool() -> Result<()> {
//                         test_functions::[<$op _test>](DType::BOOL)
//                     }
//
//                     #[test]
//                     fn u8() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U8)
//                     }
//
//                     #[test]
//                     fn u16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U16)
//                     }
//
//                     #[test]
//                     fn u32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn u64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U64)
//                     }
//
//                     #[test]
//                     fn i8() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I8)
//                     }
//
//                     #[test]
//                     fn i16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I16)
//                     }
//
//                     #[test]
//                     fn i32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn i64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I64)
//                     }
//                 }
//             }
//         )*
//     };
// }
//
// #[macro_export]
// macro_rules! test_ops_only_integer {
//     ([$($op:ident),*]) => {
//         $(
//             mod $op {
//                 use super::*;
//                 use paste::paste;
//
//                 paste! {
//                     #[test]
//                     fn u8() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U8)
//                     }
//
//                     #[test]
//                     fn u16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U16)
//                     }
//
//                     #[test]
//                     fn u32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn u64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::U64)
//                     }
//
//                     #[test]
//                     fn i8() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I8)
//                     }
//
//                     #[test]
//                     fn i16() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I16)
//                     }
//
//                     #[test]
//                     fn i32() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I32)
//                     }
//
//                     #[cfg(not(feature = "mps"))]
//                     #[test]
//                     fn i64() -> Result<()> {
//                         test_functions::[<$op _test>](DType::I64)
//                     }
//                 }
//             }
//         )*
//     };
// }

use maidenx_core::error::Result;
use maidenx_tensor_v2::{eager_mode, lazy_mode, TensorMode};

pub fn test_both_modes<F>(test_fn: F) -> Result<()>
where
    F: Fn(TensorMode) -> Result<()>,
{
    {
        let _guard = eager_mode();
        test_fn(TensorMode::Eager)?;
    }

    {
        let _guard = lazy_mode();
        test_fn(TensorMode::Lazy)?;
    }

    Ok(())
}
