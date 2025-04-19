mod utils;

use maidenx_core::{
    device::{auto_set_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor_v2::Tensor;
// use utils::setup_tensor_without_dtype;

#[test]
fn new() -> Result<()> {
    auto_set_device();

    let x = Tensor::new(vec![1, 2, 3])?;
    assert_eq!(x.to_flatten_vec::<i32>()?, [1, 2, 3]);

    Ok(())
}

#[test]
fn new_with_spec() -> Result<()> {
    let x = Tensor::new_with_spec(vec![1, 2, 3], Device::CPU, DType::I32)?;

    assert_eq!(x.device(), Device::CPU);
    assert_eq!(x.dtype(), DType::I32);
    assert_eq!(x.to_flatten_vec::<i32>()?, [1, 2, 3]);

    // // Test type conversion
    // let y = Tensor::new_with_spec(vec![1, 2, 3], Device::CPU, DType::F32)?;
    // assert_eq!(y.dtype(), DType::F32);
    // assert_eq!(y.to_flatten_vec::<f32>()?, [1.0, 2.0, 3.0]);

    Ok(())
}

// #[test]
// fn share_buffer() -> Result<()> {
//     auto_set_device();
//
//     let x = setup_tensor_without_dtype(vec![1, 2, 3, 4])?;
//     let y = Tensor::share_buffer(&x)?;
//
//     // Should have same values
//     assert_eq!(y.to_flatten_vec::<i32>()?, [1, 2, 3, 4]);
//
//     // Since they share the same buffer, we can't directly test this,
//     // but we can check that the operation completes successfully
//     assert_eq!(y.shape(), x.shape());
//     assert_eq!(y.device(), x.device());
//     assert_eq!(y.dtype(), x.dtype());
//
//     Ok(())
// }
//
// #[test]
// fn empty() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::empty(&[2, 3])?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.size(), 6);
//
//     Ok(())
// }
//
// #[test]
// fn empty_like() -> Result<()> {
//     auto_set_device();
//
//     let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;
//     x.with_shape(&[2, 3])?;
//
//     let y = Tensor::empty_like(&x)?;
//
//     assert_eq!(y.shape(), &[2, 3]);
//     assert_eq!(y.size(), 6);
//     assert_eq!(y.device(), x.device());
//     assert_eq!(y.dtype(), x.dtype());
//
//     Ok(())
// }
//
// #[test]
// fn empty_with_spec() -> Result<()> {
//     let x = Tensor::empty_with_spec(&[2, 3], Device::CPU, DType::F32)?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.size(), 6);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::F32);
//
//     Ok(())
// }
//
// #[test]
// fn zeros() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::zeros(&[2, 3])?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![0.0; 6]);
//
//     Ok(())
// }
//
// #[test]
// fn zeros_like() -> Result<()> {
//     auto_set_device();
//
//     let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;
//     x.with_shape(&[2, 3])?;
//
//     let y = Tensor::zeros_like(&x)?;
//
//     assert_eq!(y.shape(), &[2, 3]);
//     assert_eq!(y.size(), 6);
//     assert_eq!(y.device(), x.device());
//     assert_eq!(y.dtype(), x.dtype());
//     assert!(y.to_flatten_vec::<i32>()?.iter().all(|&v| v == 0));
//
//     Ok(())
// }
//
// #[test]
// fn zeros_with_spec() -> Result<()> {
//     let x = Tensor::zeros_with_spec(&[2, 3], Device::CPU, DType::F32)?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::F32);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![0.0; 6]);
//
//     Ok(())
// }
//
// #[test]
// fn ones() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::ones(&[2, 3])?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![1.0; 6]);
//
//     Ok(())
// }
//
// #[test]
// fn ones_like() -> Result<()> {
//     auto_set_device();
//
//     let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;
//     x.with_shape(&[2, 3])?;
//
//     let y = Tensor::ones_like(&x)?;
//
//     assert_eq!(y.shape(), &[2, 3]);
//     assert_eq!(y.size(), 6);
//     assert_eq!(y.device(), x.device());
//     assert_eq!(y.dtype(), x.dtype());
//     assert!(y.to_flatten_vec::<i32>()?.iter().all(|&v| v == 1));
//
//     Ok(())
// }
//
// #[test]
// fn ones_with_spec() -> Result<()> {
//     let x = Tensor::ones_with_spec(&[2, 3], Device::CPU, DType::F32)?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::F32);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![1.0; 6]);
//
//     Ok(())
// }
//
// #[test]
// fn fill() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::fill(&[2, 3], 5.0)?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![5.0; 6]);
//
//     Ok(())
// }
//
// #[test]
// fn fill_like() -> Result<()> {
//     auto_set_device();
//
//     let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;
//     x.with_shape(&[2, 3])?;
//
//     let y = Tensor::fill_like(&x, 7)?;
//
//     assert_eq!(y.shape(), &[2, 3]);
//     assert_eq!(y.size(), 6);
//     assert_eq!(y.device(), x.device());
//     assert_eq!(y.dtype(), x.dtype());
//     assert!(y.to_flatten_vec::<i32>()?.iter().all(|&v| v == 7));
//
//     Ok(())
// }
//
// #[test]
// fn fill_with_spec() -> Result<()> {
//     let x = Tensor::fill_with_spec(&[2, 3], 5.0, Device::CPU, DType::F32)?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::F32);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![5.0; 6]);
//
//     Ok(())
// }
//
// #[test]
// fn randn() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::randn(&[2, 3])?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.size(), 6);
//
//     Ok(())
// }
//
// #[test]
// fn randn_like() -> Result<()> {
//     auto_set_device();
//
//     let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;
//     x.with_shape(&[2, 3])?;
//
//     let y = Tensor::randn_like(&x)?;
//
//     assert_eq!(y.shape(), &[2, 3]);
//     assert_eq!(y.size(), 6);
//     assert_eq!(y.device(), x.device());
//     assert_eq!(y.dtype(), x.dtype());
//
//     Ok(())
// }
//
// #[test]
// fn randn_with_spec() -> Result<()> {
//     let x = Tensor::randn_with_spec(&[2, 3], Device::CPU, DType::F32)?;
//
//     assert_eq!(x.shape(), &[2, 3]);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::F32);
//     assert_eq!(x.size(), 6);
//
//     Ok(())
// }
//
// #[test]
// fn range() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::range(5)?;
//
//     assert_eq!(x.shape(), &[5]);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
//
//     Ok(())
// }
//
// #[test]
// fn range_with_spec() -> Result<()> {
//     let x = Tensor::range_with_spec(5, Device::CPU, DType::I32)?;
//
//     assert_eq!(x.shape(), &[5]);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::I32);
//     assert_eq!(x.to_flatten_vec::<i32>()?, vec![0, 1, 2, 3, 4]);
//
//     Ok(())
// }
//
// #[test]
// fn arange() -> Result<()> {
//     auto_set_device();
//
//     let x = Tensor::arange(1.0, 5.0, 1.0)?;
//
//     assert_eq!(x.shape(), &[4]);
//     assert_eq!(x.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
//
//     // Test with different step
//     let y = Tensor::arange(0, 10, 2)?;
//     assert_eq!(y.to_flatten_vec::<f32>()?, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
//
//     // Test with negative step
//     let z = Tensor::arange(5, 0, -1)?;
//     assert_eq!(z.to_flatten_vec::<f32>()?, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
//
//     Ok(())
// }
//
// #[test]
// fn arange_with_spec() -> Result<()> {
//     let x = Tensor::arange_with_spec(1, 5, 1, Device::CPU, DType::I32)?;
//
//     assert_eq!(x.shape(), &[4]);
//     assert_eq!(x.device(), Device::CPU);
//     assert_eq!(x.dtype(), DType::I32);
//     assert_eq!(x.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4]);
//
//     // Test with float dtype
//     let y = Tensor::arange_with_spec(0.5, 4.0, 0.5, Device::CPU, DType::F32)?;
//     assert_eq!(y.dtype(), DType::F32);
//     assert_eq!(y.to_flatten_vec::<f32>()?, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
//
//     // Test error case: zero step
//     let result = Tensor::arange_with_spec(0, 5, 0, Device::CPU, DType::I32);
//     assert!(result.is_err());
//
//     Ok(())
// }
