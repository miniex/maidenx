#![allow(clippy::useless_vec)]

mod utils;

use maidenx_core::{dtype::DType, error::Result};
use utils::{setup_grad_tensor_with_shape, setup_tensor_with_shape};

mod test_functions {
    use super::*;

    pub fn broadcast_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                assert_eq!(broadcasted.to_flatten_vec::<f32>()?, vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0]);
            }
            DType::BOOL => {
                let x = setup_tensor_with_shape(vec![true, false], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                assert_eq!(broadcasted.to_flatten_vec::<bool>()?, vec![true, false, true, false, true, false]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 2]);

                let actual = broadcasted.to_flatten_vec::<f32>()?;
                let expected = vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0];
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![3.0f32, 3.0];
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                assert_eq!(broadcasted.to_flatten_vec::<f32>()?, vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0]);

                if let Some(g) = x.grad()? {
                    if dtype == DType::BOOL {
                        assert_eq!(g.to_flatten_vec::<bool>()?, vec![true; 2]);
                    } else {
                        assert_eq!(g.to_flatten_vec::<f32>()?, vec![3.0f32, 3.0]);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn broadcast_left_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0f32, 2.0].repeat(12);
                assert_eq!(broadcasted.to_flatten_vec::<f32>()?, expected);
            }
            DType::BOOL => {
                let x = setup_tensor_with_shape(vec![true, false], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [true, false].repeat(12);
                assert_eq!(broadcasted.to_flatten_vec::<bool>()?, expected);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0f32, 2.0].repeat(12);

                let actual = broadcasted.to_flatten_vec::<f32>()?;
                for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert!((a - e).abs() < 0.1, "Value at index {} expected to be close to {}, got {}", i, e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![12.0f32, 12.0];
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0f32, 2.0].repeat(12);
                assert_eq!(broadcasted.to_flatten_vec::<f32>()?, expected);

                if let Some(g) = x.grad()? {
                    if dtype == DType::BOOL {
                        assert_eq!(g.to_flatten_vec::<bool>()?, vec![true; 2]);
                    } else {
                        assert_eq!(g.to_flatten_vec::<f32>()?, vec![12.0f32, 12.0]);
                    }
                }
            }
        }
        Ok(())
    }
}

test_ops!([broadcast, broadcast_left]);
