mod utils;

use maidenx_core::{device::auto_set_device, error::Result};
use maidenx_tensor_v2::{Tensor, TensorMode};
use utils::test_both_modes;

#[test]
fn contiguous() -> Result<()> {
    test_both_modes(|mode| {
        auto_set_device();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::from_flatten_vec(data, &[3, 4]);
        assert!(tensor.is_contiguous());

        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());

        let contiguous_tensor = transposed.contiguous();

        assert!(contiguous_tensor.is_contiguous());
        assert_eq!(contiguous_tensor.mode(), mode);
        assert_ne!(transposed.id(), contiguous_tensor.id());

        match mode {
            TensorMode::Eager => {
                assert!(contiguous_tensor.is_const());
                assert!(contiguous_tensor.is_storaged());

                let contiguous_data = contiguous_tensor.to_flatten_vec::<f32>();
                assert_eq!(contiguous_data.len(), 12);
            },
            TensorMode::Lazy => {
                assert!(!contiguous_tensor.is_const());
                assert!(!contiguous_tensor.is_storaged());

                contiguous_tensor.forward();
                assert!(contiguous_tensor.is_storaged());

                let contiguous_data = contiguous_tensor.to_flatten_vec::<f32>();
                assert_eq!(contiguous_data.len(), 12);
            },
        }
        Ok(())
    })
}
