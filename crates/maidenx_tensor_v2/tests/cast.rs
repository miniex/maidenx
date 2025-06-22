mod utils;

use maidenx_core::{
    device::{auto_set_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor_v2::{Tensor, TensorMode};
use utils::test_both_modes;

#[test]
fn with_device() -> Result<()> {
    auto_set_device();
    let mut tensor = Tensor::ones(&[2, 3]);
    let original_id = tensor.id();
    let original_data = tensor.to_flatten_vec::<f32>();

    tensor.with_device(Device::CPU);

    assert_eq!(tensor.device(), Device::CPU);
    assert_eq!(tensor.mode(), TensorMode::Eager);
    assert_eq!(tensor.id(), original_id); // Same tensor (in-place modification)
    assert!(tensor.is_const());
    assert!(tensor.is_storaged());

    let new_data = tensor.to_flatten_vec::<f32>();
    assert_eq!(original_data, new_data);
    assert_eq!(new_data, vec![1.0; 6]);
    Ok(())
}

#[test]
fn with_dtype() -> Result<()> {
    auto_set_device();
    let mut tensor = Tensor::ones(&[2, 3]);
    let original_tid = tensor.id();

    tensor.with_dtype(DType::F16);

    assert_eq!(tensor.dtype(), DType::F16);
    assert_eq!(tensor.mode(), TensorMode::Eager);
    assert_eq!(tensor.id(), original_tid); // Same tensor (in-place modification)
    assert!(tensor.is_const());
    assert!(tensor.is_storaged());

    let new_data = tensor.to_flatten_vec::<f32>();
    assert_eq!(new_data, vec![1.0; 6]);
    Ok(())
}

#[test]
fn to_device() -> Result<()> {
    test_both_modes(|mode| {
        auto_set_device();
        let tensor = Tensor::ones(&[2, 3]);
        let original_data = tensor.to_flatten_vec::<f32>();
        let new_tensor = tensor.to_device(Device::CPU);

        assert_eq!(new_tensor.device(), Device::CPU);
        assert_eq!(new_tensor.mode(), mode);
        assert_ne!(tensor.id(), new_tensor.id());

        match mode {
            TensorMode::Eager => {
                assert!(new_tensor.is_const());
                assert!(new_tensor.is_storaged());

                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(original_data, new_data);
                assert_eq!(new_data, vec![1.0; 6]);
            },
            TensorMode::Lazy => {
                assert!(!new_tensor.is_const());
                assert!(!new_tensor.is_storaged());

                new_tensor.forward();
                assert!(new_tensor.is_storaged());

                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(original_data, new_data);
                assert_eq!(new_data, vec![1.0; 6]);
            },
        }
        Ok(())
    })
}

#[test]
fn to_dtype() -> Result<()> {
    test_both_modes(|mode| {
        auto_set_device();
        let tensor = Tensor::ones(&[2, 3]);
        let new_tensor = tensor.to_dtype(DType::F16);

        assert_eq!(new_tensor.dtype(), DType::F16);
        assert_eq!(new_tensor.mode(), mode);
        assert_ne!(tensor.id(), new_tensor.id());

        match mode {
            TensorMode::Eager => {
                assert!(new_tensor.is_const());
                assert!(new_tensor.is_storaged());

                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(new_data, vec![1.0; 6]);
            },
            TensorMode::Lazy => {
                assert!(!new_tensor.is_const());
                assert!(!new_tensor.is_storaged());

                new_tensor.forward();
                assert!(new_tensor.is_storaged());

                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(new_data, vec![1.0; 6]);
            },
        }
        Ok(())
    })
}
