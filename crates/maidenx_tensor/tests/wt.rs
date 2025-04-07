mod utils;

use maidenx_core::{
    device::{auto_set_device, Device},
    dtype::DType,
    error::Result,
};
use utils::{setup_tensor, setup_tensor_without_dtype};

#[test]
fn with_shape() -> Result<()> {
    auto_set_device();

    let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;

    // Test reshaping to 2x3
    x.with_shape(&[2, 3])?;
    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4, 5, 6]);

    // Test reshaping to 3x2
    x.with_shape(&[3, 2])?;
    assert_eq!(x.shape(), &[3, 2]);
    assert_eq!(x.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4, 5, 6]);

    // Test with invalid shape (should fail)
    let result = x.with_shape(&[2, 2]);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn to_shape() -> Result<()> {
    auto_set_device();

    let x = setup_tensor_without_dtype(vec![1, 2, 3, 4, 5, 6])?;

    // Test to_shape to 2x3
    let y = x.to_shape(&[2, 3])?;
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4, 5, 6]);

    // Original tensor should remain unchanged
    assert_eq!(x.shape(), &[6]);

    // Test with invalid shape (should fail)
    let result = x.to_shape(&[2, 2]);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn with_device() -> Result<()> {
    auto_set_device();

    let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4])?;
    let original_device = x.device();

    // Test with same device (no-op)
    x.with_device(original_device)?;
    assert_eq!(x.device(), original_device);
    assert_eq!(x.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4]);

    // Test with CPU device (always available)
    x.with_device(Device::CPU)?;
    assert_eq!(x.device(), Device::CPU);
    assert_eq!(x.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4]);

    Ok(())
}

#[test]
fn to_device() -> Result<()> {
    auto_set_device();

    let x = setup_tensor_without_dtype(vec![1, 2, 3, 4])?;
    let original_device = x.device();

    // Test to_device to CPU
    let y = x.to_device(Device::CPU)?;
    assert_eq!(y.device(), Device::CPU);
    assert_eq!(y.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4]);

    // Original tensor should remain unchanged
    assert_eq!(x.device(), original_device);

    Ok(())
}

#[test]
fn with_dtype() -> Result<()> {
    auto_set_device();

    let mut x = setup_tensor_without_dtype(vec![1, 2, 3, 4])?;
    let original_dtype = x.dtype();

    // Test conversion to F32 dtype
    x.with_dtype(DType::F32)?;
    assert_eq!(x.dtype(), DType::F32);
    assert_eq!(x.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    // Test conversion back to original dtype
    x.with_dtype(original_dtype)?;
    assert_eq!(x.dtype(), original_dtype);

    // Test invalid dtype conversion for MPS (conditionally compiled)
    #[cfg(feature = "mps")]
    if x.device() == Device::MPS {
        let result = x.with_dtype(DType::F64);
        assert!(result.is_err());
    }

    Ok(())
}

#[test]
fn to_dtype() -> Result<()> {
    auto_set_device();

    let x = setup_tensor_without_dtype(vec![1, 2, 3, 4])?;
    let original_dtype = x.dtype();

    // Test to_dtype to F32
    let y = x.to_dtype(DType::F32)?;
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    // Original tensor should remain unchanged
    assert_eq!(x.dtype(), original_dtype);

    // Test invalid dtype conversion for MPS (conditionally compiled)
    #[cfg(feature = "mps")]
    if x.device() == Device::MPS {
        let result = x.to_dtype(DType::F64);
        assert!(result.is_err());
    }

    Ok(())
}

#[test]
fn with_grad() -> Result<()> {
    auto_set_device();

    // Float tensor (should work)
    let mut x = setup_tensor(vec![1.0, 2.0, 3.0, 4.0], DType::F32)?;

    // Should not have grad initially
    assert!(!x.requires_grad());
    assert!(x.grad()?.is_none());

    // Enable grad
    x.with_grad()?;
    assert!(x.requires_grad());
    assert!(x.grad()?.is_some());

    // Check grad tensor has same shape and initialized to zeros
    let grad = x.grad()?.unwrap();
    assert_eq!(grad.shape(), x.shape());
    assert_eq!(grad.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 0.0, 0.0]);

    // Non-float tensor (should fail)
    let mut y = setup_tensor(vec![1, 2, 3, 4], DType::I32)?;
    let result = y.with_grad();
    assert!(result.is_err());

    Ok(())
}
