mod utils;

use maidenx_core::{dtype::DType, error::Result};
use utils::{setup_grad_tensor_with_shape, setup_tensor_with_shape};

mod test_functions {
    use super::*;

    const TEST_DATA_F32_1D: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    const TEST_DATA_F32_2D: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    fn assert_close_vectors(a: &[f32], b: &[f32], tolerance: f32, message: &str) {
        assert_eq!(a.len(), b.len(), "{} - vector lengths differ", message);
        for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (a_val - b_val).abs() < tolerance,
                "{} - element {} differs: {} vs {}",
                message,
                i,
                a_val,
                b_val
            );
        }
    }

    pub fn pad_with_constant_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad(&[(1, 2)], 0.0)?;

                assert_eq!(padded.shape(), &[7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
                assert_eq!(padded_data, expected, "1D constant padding results don't match");

                let x = setup_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad(&[(1, 1), (2, 1)], 3.0)?;

                assert_eq!(padded.shape(), &[4, 6]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 3.0, 3.0, 3.0,
                    3.0, 3.0, 3.0, 3.0,
                ];
                assert_eq!(padded_data, expected, "2D constant padding results don't match");

                let x = setup_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad(&[(0, 2)], 9.9)?;

                assert_eq!(padded.shape(), &[6]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let int_expected = vec![1.0, 2.0, 3.0, 4.0, 9.0, 9.0];
                assert_close_vectors(
                    &padded_data,
                    &int_expected,
                    0.01,
                    "Non-zero pad values (int types) don't match",
                );
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad(&[(1, 2)], 0.0)?;

                assert_eq!(padded.shape(), &[7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
                assert_close_vectors(&padded_data, &expected, 0.01, "1D constant padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![1.0, 1.0, 1.0, 1.0];
                    assert_close_vectors(
                        &grad_data,
                        &expected_grad,
                        0.01,
                        "1D constant padding gradients don't match",
                    );
                }

                let x =
                    setup_grad_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad(&[(1, 1), (2, 1)], 3.0)?;

                assert_eq!(padded.shape(), &[4, 6]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 3.0, 3.0, 3.0,
                    3.0, 3.0, 3.0, 3.0,
                ];
                assert_close_vectors(&padded_data, &expected, 0.01, "2D constant padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
                    assert_close_vectors(
                        &grad_data,
                        &expected_grad,
                        0.01,
                        "2D constant padding gradients don't match",
                    );
                }

                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad(&[(0, 2)], 9.9)?;

                assert_eq!(padded.shape(), &[6]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0, 2.0, 3.0, 4.0, 9.9, 9.9];
                assert_close_vectors(&padded_data, &expected, 0.1, "Non-zero pad values don't match");
            },
            _ => {
                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad(&[(1, 2)], 0.0)?;

                assert_eq!(padded.shape(), &[7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
                assert_eq!(padded_data, expected, "1D constant padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![1.0, 1.0, 1.0, 1.0];
                    assert_eq!(grad_data, expected_grad, "1D constant padding gradients don't match");
                }

                let x =
                    setup_grad_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad(&[(1, 1), (2, 1)], 3.0)?;

                assert_eq!(padded.shape(), &[4, 6]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 3.0, 3.0, 3.0,
                    3.0, 3.0, 3.0, 3.0,
                ];
                assert_eq!(padded_data, expected, "2D constant padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
                    assert_eq!(grad_data, expected_grad, "2D constant padding gradients don't match");
                }

                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad(&[(0, 2)], 9.9)?;

                assert_eq!(padded.shape(), &[6]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0, 2.0, 3.0, 4.0, 9.9, 9.9];
                assert_eq!(padded_data, expected, "Non-zero pad values don't match");
            },
        }

        Ok(())
    }

    pub fn pad_with_reflection_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad_with_reflection(&[(2, 2)])?;

                assert_eq!(padded.shape(), &[8]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
                assert_eq!(padded_data, expected, "1D reflection padding results don't match");

                let x = setup_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad_with_reflection(&[(1, 1), (2, 2)])?;

                assert_eq!(padded.shape(), &[4, 7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 5.0, 6.0, 5.0,
                    4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0,
                ];
                assert_eq!(padded_data, expected, "2D reflection padding results don't match");

                let x = setup_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let result = x.pad_with_reflection(&[(2, 1)]);
                assert!(result.is_err(), "Should error when pad width >= dimension size");
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad_with_reflection(&[(2, 2)])?;

                assert_eq!(padded.shape(), &[8]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
                assert_close_vectors(
                    &padded_data,
                    &expected,
                    0.01,
                    "1D reflection padding results don't match",
                );

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_data.len(), 4);
                    let expected_grad = vec![1.0, 3.0, 3.0, 1.0];
                    assert_close_vectors(
                        &grad_data,
                        &expected_grad,
                        0.01,
                        "1D reflection padding gradients don't match",
                    );
                }

                let x =
                    setup_grad_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad_with_reflection(&[(1, 1), (2, 2)])?;

                assert_eq!(padded.shape(), &[4, 7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 5.0, 6.0, 5.0,
                    4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0,
                ];
                assert_close_vectors(
                    &padded_data,
                    &expected,
                    0.01,
                    "2D reflection padding results don't match",
                );
            },
            _ => {
                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad_with_reflection(&[(2, 2)])?;

                assert_eq!(padded.shape(), &[8]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
                assert_eq!(padded_data, expected, "1D reflection padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_data.len(), 4);
                    let expected_grad = vec![1.0, 3.0, 3.0, 1.0];
                    assert_eq!(grad_data, expected_grad, "1D reflection padding gradients don't match");
                }

                let x =
                    setup_grad_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad_with_reflection(&[(1, 1), (2, 2)])?;

                assert_eq!(padded.shape(), &[4, 7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 5.0, 6.0, 5.0,
                    4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0,
                ];
                assert_eq!(padded_data, expected, "2D reflection padding results don't match");

                let x = setup_tensor_with_shape(vec![1.0, 2.0], dtype, &[2])?;
                let result = x.pad_with_reflection(&[(2, 1)]);
                assert!(result.is_err(), "Should error when pad width >= dimension size");
            },
        }
        Ok(())
    }

    pub fn pad_with_replication_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad_with_replication(&[(2, 2)])?;

                assert_eq!(padded.shape(), &[8]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0];
                assert_eq!(padded_data, expected, "1D replication padding results don't match");

                let x = setup_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad_with_replication(&[(1, 1), (2, 2)])?;

                assert_eq!(padded.shape(), &[4, 7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0,
                    6.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0,
                ];
                assert_eq!(padded_data, expected, "2D replication padding results don't match");

                let x = setup_tensor_with_shape(vec![42.0], dtype, &[])?;
                let padded = x.pad_with_replication(&[])?;

                assert!(padded.shape().is_empty());
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![42.0];
                assert_eq!(padded_data, expected, "0D replication padding results don't match");
            },
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad_with_replication(&[(2, 2)])?;

                assert_eq!(padded.shape(), &[8]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0];
                assert_close_vectors(
                    &padded_data,
                    &expected,
                    0.01,
                    "1D replication padding results don't match",
                );

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_data.len(), 4);
                    let expected_grad = vec![3.0, 1.0, 1.0, 3.0];
                    assert_close_vectors(
                        &grad_data,
                        &expected_grad,
                        0.01,
                        "1D replication padding gradients don't match",
                    );
                }

                let x =
                    setup_grad_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad_with_replication(&[(1, 1), (2, 2)])?;

                assert_eq!(padded.shape(), &[4, 7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0,
                    6.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0,
                ];
                assert_close_vectors(
                    &padded_data,
                    &expected,
                    0.01,
                    "2D replication padding results don't match",
                );

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_data.len(), 6);
                }

                let x = setup_grad_tensor_with_shape(vec![42.0], dtype, &[])?;
                let padded = x.pad_with_replication(&[])?;

                assert!(padded.shape().is_empty());
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![42.0];
                assert_close_vectors(
                    &padded_data,
                    &expected,
                    0.01,
                    "0D replication padding results don't match",
                );
            },
            _ => {
                let x = setup_grad_tensor_with_shape(TEST_DATA_F32_1D.to_vec(), dtype, &[4])?;
                let padded = x.pad_with_replication(&[(2, 2)])?;

                assert_eq!(padded.shape(), &[8]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0];
                assert_eq!(padded_data, expected, "1D replication padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_data.len(), 4);
                    let expected_grad = vec![3.0, 1.0, 1.0, 3.0];
                    assert_eq!(grad_data, expected_grad, "1D replication padding gradients don't match");
                }

                let x =
                    setup_grad_tensor_with_shape(TEST_DATA_F32_2D.iter().flatten().copied().collect(), dtype, &[2, 3])?;
                let padded = x.pad_with_replication(&[(1, 1), (2, 2)])?;

                assert_eq!(padded.shape(), &[4, 7]);
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![
                    1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0,
                    6.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0,
                ];
                assert_eq!(padded_data, expected, "2D replication padding results don't match");

                padded.backward()?;

                if let Some(g) = x.grad()? {
                    let grad_data = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_data.len(), 6);

                    assert!(grad_data.iter().all(|&x| x > 0.0), "All gradients should be positive");
                    assert!(
                        grad_data[0] > grad_data[1],
                        "Corner elements should have higher gradients"
                    );
                    assert!(
                        grad_data[5] > grad_data[4],
                        "Corner elements should have higher gradients"
                    );
                }

                let x = setup_grad_tensor_with_shape(vec![42.0], dtype, &[])?;
                let padded = x.pad_with_replication(&[])?;

                assert!(padded.shape().is_empty());
                let padded_data = padded.to_flatten_vec::<f32>()?;
                let expected = vec![42.0];
                assert_eq!(padded_data, expected, "0D replication padding results don't match");
            },
        }
        Ok(())
    }
}

test_ops!([pad_with_constant, pad_with_reflection, pad_with_replication]);
