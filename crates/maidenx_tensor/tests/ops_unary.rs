#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::useless_vec)]

mod utils;

use maidenx_core::{dtype::DType, error::Result};
use utils::{setup_grad_tensor, setup_tensor};

mod test_functions {
    use super::*;

    const TEST_DATA_F32: [f32; 4] = [-1.0, 0.0, 2.0, -3.0];
    const TEST_DATA_U32: [u32; 4] = [1, 0, 2, 3];
    const TEST_DATA_BOOL: [bool; 4] = [true, false, false, true];

    pub fn neg_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.neg()?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-1.0, 0.0, -2.0, -3.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.neg()?;
                result.backward()?;

                let expected = vec![1.0, 0.0, -2.0, 3.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![-1.0, -1.0, -1.0, -1.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.neg()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, -2.0, 3.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.0, -1.0, -1.0, -1.0]);
                }
            }
        }
        Ok(())
    }

    pub fn abs_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.abs()?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 2.0, 3.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.abs()?;
                result.backward()?;

                let expected = vec![1.0, 0.0, 2.0, 3.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![-1.0, 0.0, 1.0, -1.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.abs()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 2.0, 3.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.0, 0.0, 1.0, -1.0]);
                }
            }
        }
        Ok(())
    }

    pub fn sign_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.sign()?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 1.0, 1.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sign()?;
                result.backward()?;

                let expected = vec![-1.0, 0.0, 1.0, -1.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sign()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-1.0, 0.0, 1.0, -1.0]);
            }
        }
        Ok(())
    }

    pub fn square_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.square()?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 4.0, 9.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.square()?;
                result.backward()?;

                let expected = vec![1.0, 0.0, 4.0, 9.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![-2.0, 0.0, 4.0, -6.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.square()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 4.0, 9.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![-2.0, 0.0, 4.0, -6.0]);
                }
            }
        }
        Ok(())
    }

    pub fn sqrt_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.sqrt()?;

                let expected = vec![1.0, 0.0, 1.4142135381698608, 1.7320507764816284];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 1e-3, "Expected value close to {}, got {}", e, a);
                }
            }
            DType::BF16 | DType::F16 => {
                // Skip negative values since sqrt of negative is undefined
                let test_data = vec![1.0f32, 0.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.sqrt()?;
                result.backward()?;

                let expected = vec![1.0, 0.0, 1.4142135381698608, 1.7320507764816284];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.5, f32::INFINITY, 0.3535533845424652, 0.28867512941360474];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (i, (a, e)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
                        if e.is_infinite() && a.is_infinite() {
                            assert_eq!(
                                e.is_sign_positive(),
                                a.is_sign_positive(),
                                "Expected infinite value with same sign at index {}",
                                i
                            );
                            continue;
                        }
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                // Skip negative values since sqrt of negative is undefined
                let test_data = vec![1.0f32, 0.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.sqrt()?;
                result.backward()?;

                let expected = vec![1.0, 0.0, 1.4142135381698608, 1.7320507764816284];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 1e-3, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.5, f32::INFINITY, 0.3535533845424652, 0.28867512941360474];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (i, (a, e)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
                        if e.is_infinite() && a.is_infinite() {
                            assert_eq!(
                                e.is_sign_positive(),
                                a.is_sign_positive(),
                                "Expected infinite value with same sign at index {}",
                                i
                            );
                            continue;
                        }
                        assert!((a - e).abs() < 1e-3, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn relu_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.relu()?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 2.0, 3.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.relu()?;
                result.backward()?;

                let expected = vec![0.0, 0.0, 2.0, 0.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.0, 0.0, 1.0, 0.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.relu()?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 2.0, 0.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 1.0, 0.0]);
                }
            }
        }
        Ok(())
    }

    pub fn sigmoid_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.sigmoid()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.7310586).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.5).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.8807971).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.9525741).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sigmoid()?;
                result.backward()?;

                let expected = vec![0.2689414, 0.5, 0.8807971, 0.04742587];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.19661193, 0.25, 0.10499358, 0.04517666];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sigmoid()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.2689414).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.5).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.8807971).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.04742587).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.19661193).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 0.25).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 0.10499358).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.04517666).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn tanh_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.tanh()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.7615942).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.9640276).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.9950547).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.tanh()?;
                result.backward()?;

                let expected = vec![-0.7615942, 0.0, 0.9640276, -0.9950547];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.41997434, 1.0, 0.07065082, 0.009866037];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.tanh()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - (-0.7615942)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.9640276).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - (-0.9950547)).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.41997434).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 0.07065082).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.009866037).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn gelu_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.gelu()?;

                let expected = vec![0.8413447141647339, 0.0, 1.9544997215270996, 2.9959497451782227];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 1e-3, "Expected value close to {}, got {}", e, a);
                }
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.gelu()?;
                result.backward()?;

                let expected = vec![-0.15865525603294373, 0.0, 1.9544997215270996, -0.004050225019454956];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.15, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![-0.08331543207168579, 0.5, 1.085231900215149, -0.011945605278015137];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.15, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.gelu()?;
                result.backward()?;

                let expected = vec![-0.15865525603294373, 0.0, 1.9544997215270996, -0.004050225019454956];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 1e-3, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![-0.08331543207168579, 0.5, 1.085231900215149, -0.011945605278015137];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 1e-3, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn sin_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.sin()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.8414710).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.9092974).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.1411200).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sin()?;
                result.backward()?;

                let expected = vec![-0.8414710, 0.0, 0.9092974, -0.1411200];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.5403023, 1.0, -0.4161468, -0.9899925];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sin()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - (-0.8414710)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.9092974).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - (-0.1411200)).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.5403023).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - (-0.4161468)).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - (-0.9899925)).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn cos_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.cos()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.5403023).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - (-0.4161468)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - (-0.9899925)).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.cos()?;
                result.backward()?;

                let expected = vec![0.5403023, 1.0, -0.4161468, -0.9899925];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.8414710, 0.0, -0.9092974, 0.1411200];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.cos()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.5403023).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - (-0.4161468)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - (-0.9899925)).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.8414710).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - (-0.9092974)).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.1411200).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn tan_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.tan()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 1.5574077).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - (-2.1850399)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - (-0.1425465)).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.tan()?;
                result.backward()?;

                let expected = vec![-1.5574077, 0.0, -2.1850399, 0.1425465];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.2, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![3.4255188, 1.0, 5.7743992, 1.0203195];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.2, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.tan()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - (-1.5574077)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - (-2.1850399)).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.1425465).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 3.4255188).abs() < 1e-3);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-3);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 5.7743992).abs() < 1e-3);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 1.0203195).abs() < 1e-3);
                }
            }
        }
        Ok(())
    }

    pub fn ln_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.ln()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.0).abs() < 1e-6);
                // Skip index 1 which is 0 (ln(0) is undefined)
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.6931472).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 1.0986123).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                // Use positive data for ln
                let test_data = vec![1.0f32, 1.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.ln()?;
                result.backward()?;

                let expected = vec![0.0, 0.0, 0.6931472, 1.0986123];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![1.0, 1.0, 0.5, 0.33333334];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                // Use positive data for ln
                let test_data = vec![1.0f32, 1.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.ln()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.6931472).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 1.0986123).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 1.0).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 0.5).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.33333334).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn log10_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.log10()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.0).abs() < 1e-6);
                // Skip index 1 which is 0 (log10(0) is undefined)
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.30103).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.47712126).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                // Use positive data for log10
                let test_data = vec![1.0f32, 1.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.log10()?;
                result.backward()?;

                let expected = vec![0.0, 0.0, 0.30103, 0.47712126];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.43429448, 0.43429448, 0.21714724, 0.14476483];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                // Use positive data for log10
                let test_data = vec![1.0f32, 1.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.log10()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.30103).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.47712126).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.43429448).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 0.43429448).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 0.21714724).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.14476483).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn log2_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.log2()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.0).abs() < 1e-6);
                // Skip index 1 which is 0 (log2(0) is undefined)
                assert!((result.to_flatten_vec::<f32>()?[2] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 1.5849625).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                // Use positive data for log2
                let test_data = vec![1.0f32, 1.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.log2()?;
                result.backward()?;

                let expected = vec![0.0, 0.0, 1.0, 1.5849625];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![1.4426950, 1.4426950, 0.7213475, 0.4808983];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                // Use positive data for log2
                let test_data = vec![1.0f32, 1.0, 2.0, 3.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.log2()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 1.5849625).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 1.4426950).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 1.4426950).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 0.7213475).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.4808983).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn exp_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.exp()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 2.7182817).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[2] - 7.389056).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[3] - 20.085537).abs() < 1e-3);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.exp()?;
                result.backward()?;

                let expected = vec![0.36787945, 1.0, 7.389056, 0.0497871];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.36787945, 1.0, 7.389056, 0.0497871];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.exp()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.36787945).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 7.389056).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.0497871).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.36787945).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 7.389056).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.0497871).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn exp10_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.exp10()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 10.0).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[2] - 100.0).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[3] - 1000.0).abs() < 1e-3);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.exp10()?;
                result.backward()?;

                let expected = vec![0.1, 1.0, 100.0, 0.001];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.2 * e.abs(), "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.23025851, 2.3025851, 230.25851, 0.0023025851];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.2 * e.abs(), "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.exp10()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.1).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[2] - 100.0).abs() < 1e-3);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.001).abs() < 1e-3);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.23025851).abs() < 1e-3);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 2.3025851).abs() < 1e-3);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 230.25851).abs() < 1e-3);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.0023025851).abs() < 1e-3);
                }
            }
        }
        Ok(())
    }

    pub fn exp2_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.exp2()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 2.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 4.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 8.0).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.exp2()?;
                result.backward()?;

                let expected = vec![0.5, 1.0, 4.0, 0.125];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.34657359, 0.6931472, 2.7725887, 0.08664339];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.exp2()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.5).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 4.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.125).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.34657359).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 0.6931472).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 2.7725887).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.08664339).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn softplus_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.softplus()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 1.3132617).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.6931472).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 2.1269281).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 3.0485873).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.softplus()?;
                result.backward()?;

                let expected = vec![0.31326169, 0.6931472, 2.1269281, 0.04858732];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.26894143, 0.5, 0.8807971, 0.04742587];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.softplus()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 0.31326169).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.6931472).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 2.1269281).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.04858732).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - 0.26894143).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - 0.5).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - 0.8807971).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - 0.04742587).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn recip_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.recip()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 1.0).abs() < 1e-6);
                // Skip index 1 which is 0 (1/0 is undefined)
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.5).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.33333334).abs() < 1e-6);
            }
            DType::BF16 | DType::F16 => {
                // Skip zero and use positive values for recip
                let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.recip()?;
                result.backward()?;

                let expected = vec![1.0, 0.5, 0.33333334, 0.25];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![-1.0, -0.25, -0.11111111, -0.0625];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                // Skip negative values and zero for recip
                let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.recip()?;
                result.backward()?;

                assert!((result.to_flatten_vec::<f32>()?[0] - 1.0).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - 0.5).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - 0.33333334).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - 0.25).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    assert!((g.to_flatten_vec::<f32>()?[0] - (-1.0)).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - (-0.25)).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - (-0.11111111)).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - (-0.0625)).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    pub fn logical_not_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.logical_not()?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.logical_not()?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, false, false]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.logical_not()?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, false, false]);
            }
        }
        Ok(())
    }

    // With scalar operations
    pub fn add_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.add_scalar(2.0)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.0, 2.0, 4.0, 5.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.add_scalar(2.0)?;
                result.backward()?;

                let expected = vec![1.0, 2.0, 4.0, -1.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![1.0, 1.0, 1.0, 1.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.add_scalar(2.0)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 4.0, -1.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);
                }
            }
        }
        Ok(())
    }

    pub fn sub_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.sub_scalar(0.5)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![0.5, -0.5, 1.5, 2.5]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sub_scalar(0.5)?;
                result.backward()?;

                let expected = vec![-1.5, -0.5, 1.5, -3.5];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![1.0, 1.0, 1.0, 1.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.sub_scalar(0.5)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-1.5, -0.5, 1.5, -3.5]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);
                }
            }
        }
        Ok(())
    }

    pub fn mul_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.mul_scalar(2.0)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.0, 0.0, 4.0, 6.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.mul_scalar(2.0)?;
                result.backward()?;

                let expected = vec![-2.0, 0.0, 4.0, -6.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![2.0, 2.0, 2.0, 2.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.mul_scalar(2.0)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-2.0, 0.0, 4.0, -6.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![2.0, 2.0, 2.0, 2.0]);
                }
            }
        }
        Ok(())
    }

    pub fn div_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.div_scalar(2.0)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![0.5, 0.0, 1.0, 1.5]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.div_scalar(2.0)?;
                result.backward()?;

                let expected = vec![-0.5, 0.0, 1.0, -1.5];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.5, 0.5, 0.5, 0.5];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.div_scalar(2.0)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-0.5, 0.0, 1.0, -1.5]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.5, 0.5, 0.5]);
                }
            }
        }
        Ok(())
    }

    pub fn maximum_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.maximum_scalar(0.5)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.5, 2.0, 3.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.maximum_scalar(0.5)?;
                result.backward()?;

                let expected = vec![0.5, 0.5, 2.0, 0.5];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.0, 0.0, 1.0, 0.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.maximum_scalar(0.5)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![0.5, 0.5, 2.0, 0.5]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 1.0, 0.0]);
                }
            }
        }
        Ok(())
    }

    pub fn minimum_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.minimum_scalar(0.5)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![0.5, 0.0, 0.5, 0.5]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.minimum_scalar(0.5)?;
                result.backward()?;

                let expected = vec![-1.0, 0.0, 0.5, -3.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![1.0, 1.0, 0.0, 1.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.minimum_scalar(0.5)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-1.0, 0.0, 0.5, -3.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 0.0, 1.0]);
                }
            }
        }
        Ok(())
    }

    pub fn pow_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.pow(3.0)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 8.0, 27.0]);
            }
            DType::BF16 | DType::F16 => {
                // Use positive values for pow to avoid complex results
                let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.pow(3.0)?;
                result.backward()?;

                let expected = vec![1.0, 8.0, 27.0, 64.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.2 * e.abs(), "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![3.0, 12.0, 27.0, 48.0];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.2 * e.abs(), "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                // Use positive values for pow to avoid complex results
                let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
                let x = setup_grad_tensor(test_data, dtype)?;

                let result = x.pow(3.0)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 8.0, 27.0, 64.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![3.0, 12.0, 27.0, 48.0]);
                }
            }
        }
        Ok(())
    }

    pub fn leaky_relu_test(dtype: DType) -> Result<()> {
        let alpha = 0.1; // Common value for leaky ReLU

        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.leaky_relu(alpha)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 2.0, 3.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.leaky_relu(alpha)?;
                result.backward()?;

                let expected = vec![-0.10000000149011612, 0.0, 2.0, -0.30000001192092896];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![0.10000000149011612, 0.10000000149011612, 1.0, 0.10000000149011612];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.leaky_relu(alpha)?;
                result.backward()?;

                assert_eq!(
                    result.to_flatten_vec::<f32>()?,
                    vec![-0.10000000149011612, 0.0, 2.0, -0.30000001192092896]
                );

                if let Some(g) = x.grad()? {
                    assert_eq!(
                        g.to_flatten_vec::<f32>()?,
                        vec![0.10000000149011612, 0.10000000149011612, 1.0, 0.10000000149011612]
                    );
                }
            }
        }
        Ok(())
    }

    pub fn elu_test(dtype: DType) -> Result<()> {
        let alpha = 1.0; // Common value for ELU

        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.elu(alpha)?;
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 2.0, 3.0]);
            }
            DType::BF16 | DType::F16 => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.elu(alpha)?;
                result.backward()?;

                let expected = vec![alpha * ((-1.0f32).exp() - 1.0), 0.0, 2.0, alpha * ((-3.0f32).exp() - 1.0)];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![alpha * (-1.0f32).exp(), 1.0, 1.0, alpha * (-3.0f32).exp()];
                    let actual_grad = g.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.elu(alpha)?;
                result.backward()?;

                let expected = vec![alpha * ((-1.0f32).exp() - 1.0), 0.0, 2.0, alpha * ((-3.0f32).exp() - 1.0)];
                assert!((result.to_flatten_vec::<f32>()?[0] - expected[0]).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[1] - expected[1]).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[2] - expected[2]).abs() < 1e-6);
                assert!((result.to_flatten_vec::<f32>()?[3] - expected[3]).abs() < 1e-6);

                if let Some(g) = x.grad()? {
                    let expected_grad = vec![alpha * (-1.0f32).exp(), 1.0, 1.0, alpha * (-3.0f32).exp()];
                    assert!((g.to_flatten_vec::<f32>()?[0] - expected_grad[0]).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[1] - expected_grad[1]).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[2] - expected_grad[2]).abs() < 1e-6);
                    assert!((g.to_flatten_vec::<f32>()?[3] - expected_grad[3]).abs() < 1e-6);
                }
            }
        }
        Ok(())
    }

    // Logical operations with scalar
    pub fn eq_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.eq_scalar(true)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, false, true]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.eq_scalar(2)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, false, true, false]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.eq_scalar(0.0)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, false, false]);
            }
        }
        Ok(())
    }

    pub fn ne_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.ne_scalar(true)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.ne_scalar(2)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, false, true]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.ne_scalar(0.0)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, true, true]);
            }
        }
        Ok(())
    }

    pub fn lt_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.lt_scalar(true)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.lt_scalar(2)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, false, false]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.lt_scalar(0.0)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, false, true]);
            }
        }
        Ok(())
    }

    pub fn le_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.le_scalar(true)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, true, true]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.le_scalar(2)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, true, false]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.le_scalar(0.0)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, false, true]);
            }
        }
        Ok(())
    }

    pub fn gt_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.gt_scalar(false)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, false, true]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.gt_scalar(1)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, false, true, true]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.gt_scalar(0.0)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, false, true, false]);
            }
        }
        Ok(())
    }

    pub fn ge_scalar_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let result = x.ge_scalar(false)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, true, true]);
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let result = x.ge_scalar(1)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, true, true]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let result = x.ge_scalar(0.0)?;
                assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
            }
        }
        Ok(())
    }
}

test_ops!([
    neg,
    abs,
    sign,
    square,
    sqrt,
    relu,
    sigmoid,
    tanh,
    gelu,
    sin,
    cos,
    tan,
    ln,
    log10,
    log2,
    exp,
    exp10,
    exp2,
    softplus,
    recip,
    // with constant
    add_scalar,
    sub_scalar,
    mul_scalar,
    div_scalar,
    maximum_scalar,
    minimum_scalar,
    pow,
    leaky_relu,
    elu
]);

test_logical_ops!([
    logical_not,
    // with constant
    eq_scalar,
    ne_scalar,
    lt_scalar,
    le_scalar,
    gt_scalar,
    ge_scalar
]);

