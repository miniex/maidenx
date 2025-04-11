#![allow(clippy::useless_vec)]

mod utils;

use maidenx_core::{dtype::DType, error::Result};
use utils::{setup_contiguous_tensor_with_shape, setup_grad_contiguous_tensor_with_shape};

mod test_functions {
    use super::*;

    const TEST_DATA_F32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    const TEST_DATA_BOOL: [bool; 4] = [true, false, true, false];

    pub fn view_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?;
                let viewed = x.view(&[1, 2, 1, 2])?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);
                assert_eq!(viewed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());
            },
            DType::BOOL => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_BOOL.to_vec(), dtype, &[4])?;
                let viewed = x.view(&[1, 2, 1, 2])?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);
                assert_eq!(viewed.to_flatten_vec::<bool>()?, TEST_DATA_BOOL.to_vec());
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?;
                let viewed = x.view(&[1, 2, 1, 2])?;
                let y = x.view(&[2, 2])?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);

                let actual = viewed.to_flatten_vec::<f32>()?;
                let expected = TEST_DATA_F32.to_vec();
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let ones = vec![1.0f32; 4];
                    for (a, e) in actual_grad.iter().zip(ones.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?;
                let viewed = x.view(&[1, 2, 1, 2])?;
                let y = x.view(&[2, 2])?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);
                assert_eq!(viewed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32; 4]);
                }
            },
        }
        Ok(())
    }

    pub fn squeeze_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);
                assert_eq!(squeezed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());
            },
            DType::BOOL => {
                let x =
                    setup_contiguous_tensor_with_shape(TEST_DATA_BOOL.to_vec(), dtype, &[4])?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);
                assert_eq!(squeezed.to_flatten_vec::<bool>()?, TEST_DATA_BOOL.to_vec());
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?
                    .view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;
                let y = x.squeeze(1)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);

                let actual = squeezed.to_flatten_vec::<f32>()?;
                let expected = TEST_DATA_F32.to_vec();
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let ones = vec![1.0f32; 4];
                    for (a, e) in actual_grad.iter().zip(ones.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?
                    .view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;
                let y = x.squeeze(1)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);
                assert_eq!(squeezed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32; 4]);
                }
            },
        }
        Ok(())
    }

    pub fn squeeze_all_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;

                assert_eq!(squeezed.shape(), &[2, 2]);
                assert_eq!(squeezed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());
            },
            DType::BOOL => {
                let x =
                    setup_contiguous_tensor_with_shape(TEST_DATA_BOOL.to_vec(), dtype, &[4])?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;

                assert_eq!(squeezed.shape(), &[2, 2]);
                assert_eq!(squeezed.to_flatten_vec::<bool>()?, TEST_DATA_BOOL.to_vec());
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?
                    .view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;
                let y = x.squeeze_all()?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(squeezed.shape(), &[2, 2]);

                let actual = squeezed.to_flatten_vec::<f32>()?;
                let expected = TEST_DATA_F32.to_vec();
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let ones = vec![1.0f32; 4];
                    for (a, e) in actual_grad.iter().zip(ones.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?
                    .view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;
                let y = x.squeeze_all()?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(squeezed.shape(), &[2, 2]);
                assert_eq!(squeezed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32; 4]);
                }
            },
        }
        Ok(())
    }

    pub fn unsqueeze_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
                assert_eq!(unsqueezed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());
            },
            DType::BOOL => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_BOOL.to_vec(), dtype, &[4])?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
                assert_eq!(unsqueezed.to_flatten_vec::<bool>()?, TEST_DATA_BOOL.to_vec());
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;
                let y = x.unsqueeze(0)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);

                let actual = unsqueezed.to_flatten_vec::<f32>()?;
                let expected = TEST_DATA_F32.to_vec();
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let ones = vec![1.0f32; 4];
                    for (a, e) in actual_grad.iter().zip(ones.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;
                let y = x.unsqueeze(0)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
                assert_eq!(unsqueezed.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32; 4]);
                }
            },
        }
        Ok(())
    }

    pub fn transpose_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let matrix_data = vec![1, 2, 3, 4];
                let x = setup_contiguous_tensor_with_shape(matrix_data, dtype, &[2, 2])?;
                let transposed = x.transpose(0, 1)?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);
                assert_eq!(transposed.to_flatten_vec::<f32>()?, vec![1.0f32, 3.0, 2.0, 4.0]);
            },
            DType::BOOL => {
                let matrix_data = vec![true, false, false, true];
                let x = setup_contiguous_tensor_with_shape(matrix_data, dtype, &[2, 2])?;
                let transposed = x.transpose(0, 1)?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);
                assert_eq!(transposed.to_flatten_vec::<bool>()?, vec![true, false, false, true]);
            },
            DType::BF16 | DType::F16 => {
                let matrix_data = vec![1.0, 2.0, 3.0, 4.0];
                let x = setup_grad_contiguous_tensor_with_shape(matrix_data, dtype, &[2, 2])?;
                let transposed = x.transpose(0, 1)?;
                let y = x.transpose(0, 1)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);

                let actual = transposed.to_flatten_vec::<f32>()?;
                let expected = vec![1.0f32, 3.0, 2.0, 4.0];
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let ones = vec![1.0f32; 4];
                    for (a, e) in actual_grad.iter().zip(ones.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let matrix_data = vec![1.0, 2.0, 3.0, 4.0];
                let x = setup_grad_contiguous_tensor_with_shape(matrix_data, dtype, &[2, 2])?;
                let transposed = x.transpose(0, 1)?;
                let y = x.transpose(0, 1)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);
                assert_eq!(transposed.to_flatten_vec::<f32>()?, vec![1.0f32, 3.0, 2.0, 4.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32, 1.0, 1.0, 1.0]);
                }
            },
        }
        Ok(())
    }

    pub fn slice_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let sliced1 = x.slice(0, 0, Some(1), 1)?;
                let sliced2 = x.slice(1, 1, Some(3), 1)?;
                let sliced3 = x.slice(1, 0, Some(3), 2)?;

                assert_eq!(sliced1.shape(), &[1, 3]);
                assert_eq!(sliced2.shape(), &[2, 2]);
                assert_eq!(sliced3.shape(), &[2, 2]);

                assert_eq!(sliced1.to_flatten_vec::<f32>()?, vec![1.0f32, 2.0, 3.0]);
                assert_eq!(sliced2.to_flatten_vec::<f32>()?, vec![2.0f32, 3.0, 5.0, 6.0]);
                assert_eq!(sliced3.to_flatten_vec::<f32>()?, vec![1.0f32, 3.0, 4.0, 6.0]);
            },
            DType::BOOL => {
                let data = vec![true, false, true, false, true, false];
                let x = setup_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let sliced1 = x.slice(0, 0, Some(1), 1)?;
                let sliced2 = x.slice(1, 1, Some(3), 1)?;
                let sliced3 = x.slice(1, 0, Some(3), 2)?;

                assert_eq!(sliced1.shape(), &[1, 3]);
                assert_eq!(sliced2.shape(), &[2, 2]);
                assert_eq!(sliced3.shape(), &[2, 2]);

                assert_eq!(sliced1.to_flatten_vec::<bool>()?, vec![true, false, true]);
                assert_eq!(sliced2.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
                assert_eq!(sliced3.to_flatten_vec::<bool>()?, vec![true, true, false, false]);
            },
            DType::BF16 | DType::F16 => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_grad_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let sliced = x.slice(0, 1, Some(2), 1)?;
                let z = sliced.sum_all()?;
                z.backward()?;

                assert_eq!(sliced.shape(), &[1, 3]);

                let actual = sliced.to_flatten_vec::<f32>()?;
                let expected = vec![4.0f32, 5.0, 6.0];
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
                    for (i, (a, e)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
                        assert!(
                            (a - e).abs() < 0.1,
                            "Gradient at index {} expected to be close to {}, got {}",
                            i,
                            e,
                            a
                        );
                    }
                }

                let neg_sliced = x.slice(0, -1, Some(0), -1)?;
                assert_eq!(neg_sliced.shape(), &[1, 3]);

                let actual = neg_sliced.to_flatten_vec::<f32>()?;
                let expected = vec![4.0f32, 5.0, 6.0];
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }
            },
            _ => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_grad_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let sliced = x.slice(0, 1, Some(2), 1)?;
                let z = sliced.sum_all()?;
                z.backward()?;

                assert_eq!(sliced.shape(), &[1, 3]);
                assert_eq!(sliced.to_flatten_vec::<f32>()?, vec![4.0f32, 5.0, 6.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
                }

                let neg_sliced = x.slice(0, -1, Some(0), -1)?;
                assert_eq!(neg_sliced.shape(), &[1, 3]);
                assert_eq!(neg_sliced.to_flatten_vec::<f32>()?, vec![4.0f32, 5.0, 6.0]);
            },
        }
        Ok(())
    }

    pub fn unfold_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let unfolded1 = x.unfold(1, 2, 1)?;
                let unfolded2 = x.unfold(0, 1, 1)?;

                assert_eq!(unfolded1.shape(), &[2, 2, 2]);
                assert_eq!(unfolded2.shape(), &[2, 1, 3]);

                assert_eq!(
                    unfolded1.to_flatten_vec::<f32>()?,
                    vec![1.0f32, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0]
                );
                assert_eq!(
                    unfolded2.to_flatten_vec::<f32>()?,
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                );
            },
            DType::BOOL => {
                let data = vec![true, false, true, false, true, false];
                let x = setup_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let unfolded1 = x.unfold(1, 2, 1)?;
                let unfolded2 = x.unfold(0, 1, 1)?;

                assert_eq!(unfolded1.shape(), &[2, 2, 2]);
                assert_eq!(unfolded2.shape(), &[2, 1, 3]);

                assert_eq!(
                    unfolded1.to_flatten_vec::<bool>()?,
                    vec![true, false, false, true, false, true, true, false]
                );
                assert_eq!(
                    unfolded2.to_flatten_vec::<bool>()?,
                    vec![true, false, true, false, true, false]
                );
            },
            DType::BF16 | DType::F16 => {
                let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_grad_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let unfolded = x.unfold(1, 2, 1)?;
                let z = unfolded.sum_all()?;
                z.backward()?;

                assert_eq!(unfolded.shape(), &[2, 2, 2]);

                let actual = unfolded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0f32, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0];
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![1.0f32, 2.0, 1.0, 1.0, 2.0, 1.0];
                    for (i, (a, e)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
                        assert!(
                            (a - e).abs() < 0.1,
                            "Gradient at index {} expected to be close to {}, got {}",
                            i,
                            e,
                            a
                        );
                    }
                }

                let large_step_unfolded = x.unfold(1, 2, 2)?;
                assert_eq!(large_step_unfolded.shape(), &[2, 1, 2]);

                let actual = large_step_unfolded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0f32, 2.0, 4.0, 5.0];
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }
            },
            _ => {
                let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_grad_contiguous_tensor_with_shape(data.clone(), dtype, &[6])?.view(&[2, 3])?;

                let unfolded = x.unfold(1, 2, 1)?;
                let z = unfolded.sum_all()?;
                z.backward()?;

                assert_eq!(unfolded.shape(), &[2, 2, 2]);
                assert_eq!(
                    unfolded.to_flatten_vec::<f32>()?,
                    vec![1.0f32, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0]
                );

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32, 2.0, 1.0, 1.0, 2.0, 1.0]);
                }

                let large_step_unfolded = x.unfold(1, 2, 2)?;
                assert_eq!(large_step_unfolded.shape(), &[2, 1, 2]);
                assert_eq!(
                    large_step_unfolded.to_flatten_vec::<f32>()?,
                    vec![1.0f32, 2.0, 4.0, 5.0]
                );
            },
        }
        Ok(())
    }

    pub fn broadcast_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                assert_eq!(
                    broadcasted.to_flatten_vec::<f32>()?,
                    vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0]
                );
            },
            DType::BOOL => {
                let x = setup_contiguous_tensor_with_shape(vec![true, false], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                assert_eq!(
                    broadcasted.to_flatten_vec::<bool>()?,
                    vec![true, false, true, false, true, false]
                );
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
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
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast(&[3, 2])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                assert_eq!(
                    broadcasted.to_flatten_vec::<f32>()?,
                    vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0]
                );

                if let Some(g) = x.grad()? {
                    if dtype == DType::BOOL {
                        assert_eq!(g.to_flatten_vec::<bool>()?, vec![true; 2]);
                    } else {
                        assert_eq!(g.to_flatten_vec::<f32>()?, vec![3.0f32, 3.0]);
                    }
                }
            },
        }
        Ok(())
    }

    pub fn broadcast_left_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0f32, 2.0].repeat(12);
                assert_eq!(broadcasted.to_flatten_vec::<f32>()?, expected);
            },
            DType::BOOL => {
                let x = setup_contiguous_tensor_with_shape(vec![true, false], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [true, false].repeat(12);
                assert_eq!(broadcasted.to_flatten_vec::<bool>()?, expected);
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let broadcasted = x.broadcast_left(&[3, 4])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0f32, 2.0].repeat(12);

                let actual = broadcasted.to_flatten_vec::<f32>()?;
                for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (a - e).abs() < 0.1,
                        "Value at index {} expected to be close to {}, got {}",
                        i,
                        e,
                        a
                    );
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![12.0f32, 12.0];
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
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
            },
        }
        Ok(())
    }

    pub fn reshape_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?;
                let reshaped = x.reshape(&[2, 2])?;

                assert_eq!(reshaped.shape(), &[2, 2]);
                assert_eq!(reshaped.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());
            },
            DType::BOOL => {
                let x = setup_contiguous_tensor_with_shape(TEST_DATA_BOOL.to_vec(), dtype, &[4])?;
                let reshaped = x.reshape(&[2, 2])?;

                assert_eq!(reshaped.shape(), &[2, 2]);
                assert_eq!(reshaped.to_flatten_vec::<bool>()?, TEST_DATA_BOOL.to_vec());
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?;
                let reshaped = x.reshape(&[2, 2])?;
                reshaped.backward()?;

                assert_eq!(reshaped.shape(), &[2, 2]);

                let actual = reshaped.to_flatten_vec::<f32>()?;
                let expected = TEST_DATA_F32.to_vec();
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    let ones = vec![1.0f32; 4];
                    for (a, e) in actual_grad.iter().zip(ones.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected gradient close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let x = setup_grad_contiguous_tensor_with_shape(TEST_DATA_F32.to_vec(), dtype, &[4])?;
                let reshaped = x.reshape(&[2, 2])?;
                reshaped.backward()?;

                assert_eq!(reshaped.shape(), &[2, 2]);
                assert_eq!(reshaped.to_flatten_vec::<f32>()?, TEST_DATA_F32.to_vec());

                if let Some(g) = x.grad()? {
                    if dtype == DType::BOOL {
                        assert_eq!(g.to_flatten_vec::<bool>()?, vec![true; 4]);
                    } else {
                        assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0f32; 4]);
                    }
                }
            },
        }
        Ok(())
    }
}

test_ops!([
    // view
    view,
    squeeze,
    squeeze_all,
    unsqueeze,
    transpose,
    slice,
    unfold,
    broadcast,
    broadcast_left,
    // reshape
    reshape
]);
