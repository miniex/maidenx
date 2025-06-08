mod utils;

use maidenx_core::{
    device::{auto_set_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor_v2::{Tensor, TensorMode};
use utils::test_both_modes;

#[test]
fn test_with_device() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let mut tensor = Tensor::ones(&[2, 3]);
        let original_tid = tensor.tid();

        tensor.with_device(Device::CPU);

        assert_eq!(tensor.device(), Device::CPU);
        assert_eq!(tensor.tid(), original_tid); // Same tensor (modified in place)
        assert!(tensor.is_storaged());
        assert_eq!(tensor.to_flatten_vec::<i32>(), [1, 1, 1, 1, 1, 1]);
        Ok(())
    })
}

#[test]
fn test_with_dtype() -> Result<()> {
    test_both_modes(|_mode| {
        auto_set_device();
        let mut tensor = Tensor::ones(&[2, 3]);
        let original_tid = tensor.tid();

        tensor.with_dtype(DType::F16);

        assert_eq!(tensor.dtype(), DType::F16);
        assert_eq!(tensor.tid(), original_tid); // Same tensor (modified in place)
        assert!(tensor.is_storaged());
        assert_eq!(tensor.to_flatten_vec::<i32>(), [1, 1, 1, 1, 1, 1]);
        Ok(())
    })
}

#[test]
fn test_to_device() -> Result<()> {
    test_both_modes(|mode| {
        auto_set_device();
        let tensor = Tensor::ones(&[2, 3]);
        let original_data = tensor.to_flatten_vec::<f32>();
        let new_tensor = tensor.to_device(Device::CPU);

        assert_eq!(new_tensor.device(), Device::CPU);
        assert_eq!(new_tensor.mode(), mode);
        assert_ne!(tensor.tid(), new_tensor.tid()); // Different tensor

        match mode {
            TensorMode::Eager => {
                assert!(new_tensor.is_const());
                assert!(new_tensor.is_storaged());

                // Check data integrity immediately in eager mode
                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(original_data, new_data);
                assert_eq!(new_data, vec![1.0; 6]); // 2x3 tensor of ones
            },
            TensorMode::Lazy => {
                assert!(!new_tensor.is_const()); // Part of graph
                assert!(!new_tensor.is_storaged()); // Pending

                // Execute computation
                new_tensor.forward();
                assert!(new_tensor.is_storaged()); // Now materialized

                // Check data integrity after computation
                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(original_data, new_data);
                assert_eq!(new_data, vec![1.0; 6]); // 2x3 tensor of ones
            },
        }
        Ok(())
    })
}

#[test]
fn test_to_dtype() -> Result<()> {
    test_both_modes(|mode| {
        auto_set_device();
        let tensor = Tensor::ones(&[2, 3]);
        let new_tensor = tensor.to_dtype(DType::F16);

        assert_eq!(new_tensor.dtype(), DType::F16);
        assert_eq!(new_tensor.mode(), mode);
        assert_ne!(tensor.tid(), new_tensor.tid()); // Different tensor

        match mode {
            TensorMode::Eager => {
                assert!(new_tensor.is_const());
                assert!(new_tensor.is_storaged());

                // Check data integrity immediately in eager mode
                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(new_data, vec![1.0; 6]); // 2x3 tensor of ones
            },
            TensorMode::Lazy => {
                assert!(!new_tensor.is_const()); // Part of graph
                assert!(!new_tensor.is_storaged()); // Pending

                // Execute computation
                new_tensor.forward();
                assert!(new_tensor.is_storaged()); // Now materialized

                // Check data integrity after computation
                let new_data = new_tensor.to_flatten_vec::<f32>();
                assert_eq!(new_data, vec![1.0; 6]); // 2x3 tensor of ones
            },
        }
        Ok(())
    })
}
