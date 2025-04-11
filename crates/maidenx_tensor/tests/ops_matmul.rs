#![allow(clippy::useless_vec)]

mod utils;

use maidenx_core::{dtype::DType, error::Result};
use utils::{setup_grad_tensor_with_shape, setup_tensor_with_shape};

mod test_functions {
    use super::*;

    pub fn vector_vector_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let a = setup_tensor_with_shape(vec![1u32, 2, 3], dtype, &[3])?;
                let b = setup_tensor_with_shape(vec![4u32, 5, 6], dtype, &[3])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.ndim(), 0);
                assert_eq!(c.to_flatten_vec::<u32>()?, vec![32u32]);
            },
            DType::BF16 | DType::F16 => {
                let a = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0], dtype, &[3])?;
                let b = setup_grad_tensor_with_shape(vec![4.0f32, 5.0, 6.0], dtype, &[3])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.ndim(), 0);

                let expected = vec![32.0f32];
                let actual = c.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                c.backward()?;
                if let Some(a_grad) = a.grad()? {
                    let expected_grad = vec![4.0f32, 5.0, 6.0];
                    let actual_grad = a_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }

                if let Some(b_grad) = b.grad()? {
                    let expected_grad = vec![1.0f32, 2.0, 3.0];
                    let actual_grad = b_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let a = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0], dtype, &[3])?;
                let b = setup_grad_tensor_with_shape(vec![4.0f32, 5.0, 6.0], dtype, &[3])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.ndim(), 0);
                assert_eq!(c.to_flatten_vec::<f32>()?, vec![32.0f32]);

                c.backward()?;
                if let Some(a_grad) = a.grad()? {
                    assert_eq!(a_grad.to_flatten_vec::<f32>()?, vec![4.0f32, 5.0, 6.0]);
                }

                if let Some(b_grad) = b.grad()? {
                    assert_eq!(b_grad.to_flatten_vec::<f32>()?, vec![1.0f32, 2.0, 3.0]);
                }
            },
        }
        Ok(())
    }

    pub fn matrix_vector_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let matrix = setup_tensor_with_shape(vec![1u32, 2, 3, 4, 5, 6], dtype, &[2, 3])?;
                let vector = setup_tensor_with_shape(vec![2u32, 1, 3], dtype, &[3])?;
                let result = matrix.matmul(&vector)?;
                assert_eq!(result.shape(), &[2]);
                assert_eq!(result.to_flatten_vec::<u32>()?, vec![13u32, 31]);
            },
            DType::BF16 | DType::F16 => {
                let matrix = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype, &[2, 3])?;
                let vector = setup_grad_tensor_with_shape(vec![2.0f32, 1.0, 3.0], dtype, &[3])?;
                let result = matrix.matmul(&vector)?;
                assert_eq!(result.shape(), &[2]);

                let expected = vec![13.0f32, 31.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                result.backward()?;
                if let Some(matrix_grad) = matrix.grad()? {
                    let expected_grad = vec![2.0f32, 1.0, 3.0, 2.0, 1.0, 3.0];
                    let actual_grad = matrix_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }

                if let Some(vector_grad) = vector.grad()? {
                    let expected_grad = vec![5.0f32, 7.0, 9.0];
                    let actual_grad = vector_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let matrix = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype, &[2, 3])?;
                let vector = setup_grad_tensor_with_shape(vec![2.0f32, 1.0, 3.0], dtype, &[3])?;
                let result = matrix.matmul(&vector)?;
                assert_eq!(result.shape(), &[2]);
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![13.0f32, 31.0]);

                result.backward()?;
                if let Some(matrix_grad) = matrix.grad()? {
                    assert_eq!(
                        matrix_grad.to_flatten_vec::<f32>()?,
                        vec![2.0f32, 1.0, 3.0, 2.0, 1.0, 3.0]
                    );
                }

                if let Some(vector_grad) = vector.grad()? {
                    assert_eq!(vector_grad.to_flatten_vec::<f32>()?, vec![5.0f32, 7.0, 9.0]);
                }
            },
        }
        Ok(())
    }

    pub fn vector_matrix_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let vector = setup_tensor_with_shape(vec![2u32, 3], dtype, &[2])?;
                let matrix = setup_tensor_with_shape(vec![1u32, 2, 3, 4, 5, 6], dtype, &[2, 3])?;
                let result = vector.matmul(&matrix)?;
                assert_eq!(result.shape(), &[3]);
                assert_eq!(result.to_flatten_vec::<u32>()?, vec![14u32, 19, 24]);
            },
            DType::BF16 | DType::F16 => {
                let vector = setup_grad_tensor_with_shape(vec![2.0f32, 3.0], dtype, &[2])?;
                let matrix = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype, &[2, 3])?;
                let result = vector.matmul(&matrix)?;
                assert_eq!(result.shape(), &[3]);

                let expected = vec![14.0f32, 19.0, 24.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                result.backward()?;
                if let Some(vector_grad) = vector.grad()? {
                    let expected_grad = vec![6.0f32, 15.0];
                    let actual_grad = vector_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }

                if let Some(matrix_grad) = matrix.grad()? {
                    let expected_grad = vec![2.0f32, 2.0, 2.0, 3.0, 3.0, 3.0];
                    let actual_grad = matrix_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let vector = setup_grad_tensor_with_shape(vec![2.0f32, 3.0], dtype, &[2])?;
                let matrix = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype, &[2, 3])?;
                let result = vector.matmul(&matrix)?;
                assert_eq!(result.shape(), &[3]);
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![14.0f32, 19.0, 24.0]);

                result.backward()?;
                if let Some(vector_grad) = vector.grad()? {
                    assert_eq!(vector_grad.to_flatten_vec::<f32>()?, vec![6.0f32, 15.0]);
                }

                if let Some(matrix_grad) = matrix.grad()? {
                    assert_eq!(
                        matrix_grad.to_flatten_vec::<f32>()?,
                        vec![2.0f32, 2.0, 2.0, 3.0, 3.0, 3.0]
                    );
                }
            },
        }
        Ok(())
    }

    pub fn matrix_matrix_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let a = setup_tensor_with_shape(vec![1u32, 2, 3, 4], dtype, &[2, 2])?;
                let b = setup_tensor_with_shape(vec![5u32, 6, 7, 8], dtype, &[2, 2])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.shape(), &[2, 2]);
                assert_eq!(c.to_flatten_vec::<u32>()?, vec![19u32, 22, 43, 50]);
            },
            DType::BF16 | DType::F16 => {
                let a = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0], dtype, &[2, 2])?;
                let b = setup_grad_tensor_with_shape(vec![5.0f32, 6.0, 7.0, 8.0], dtype, &[2, 2])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.shape(), &[2, 2]);

                let expected = vec![19.0f32, 22.0, 43.0, 50.0];
                let actual = c.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                c.backward()?;
                if let Some(a_grad) = a.grad()? {
                    let expected_grad = vec![11.0f32, 15.0, 11.0, 15.0];
                    let actual_grad = a_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }

                if let Some(b_grad) = b.grad()? {
                    let expected_grad = vec![4.0f32, 4.0, 6.0, 6.0];
                    let actual_grad = b_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let a = setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0], dtype, &[2, 2])?;
                let b = setup_grad_tensor_with_shape(vec![5.0f32, 6.0, 7.0, 8.0], dtype, &[2, 2])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.shape(), &[2, 2]);
                assert_eq!(c.to_flatten_vec::<f32>()?, vec![19.0f32, 22.0, 43.0, 50.0]);

                c.backward()?;
                if let Some(a_grad) = a.grad()? {
                    assert_eq!(a_grad.to_flatten_vec::<f32>()?, vec![11.0f32, 15.0, 11.0, 15.0]);
                }

                if let Some(b_grad) = b.grad()? {
                    assert_eq!(b_grad.to_flatten_vec::<f32>()?, vec![4.0f32, 4.0, 6.0, 6.0]);
                }
            },
        }
        Ok(())
    }

    pub fn batched_matrix_matrix_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let a = setup_tensor_with_shape(vec![1u32, 2, 3, 4, 5, 6, 7, 8], dtype, &[2, 2, 2])?;
                let b = setup_tensor_with_shape(vec![9u32, 10, 11, 12, 13, 14, 15, 16], dtype, &[2, 2, 2])?;
                let c = a.matmul(&b)?;
                assert_eq!(c.shape(), &[2, 2, 2]);
                assert_eq!(c.to_flatten_vec::<u32>()?, vec![31u32, 34, 71, 78, 155, 166, 211, 226]);
            },
            DType::BF16 | DType::F16 => {
                let a =
                    setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype, &[2, 2, 2])?;
                let b = setup_grad_tensor_with_shape(
                    vec![9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                    dtype,
                    &[2, 2, 2],
                )?;
                let c = a.matmul(&b)?;
                assert_eq!(c.shape(), &[2, 2, 2]);

                let expected = vec![31.0f32, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0];
                let actual = c.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                c.backward()?;
                if let Some(a_grad) = a.grad()? {
                    let expected_grad = vec![19.0f32, 23.0, 19.0, 23.0, 27.0, 31.0, 27.0, 31.0];
                    let actual_grad = a_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }

                if let Some(b_grad) = b.grad()? {
                    let expected_grad = vec![4.0f32, 4.0, 6.0, 6.0, 12.0, 12.0, 14.0, 14.0];
                    let actual_grad = b_grad.to_flatten_vec::<f32>()?;
                    for (a, e) in actual_grad.iter().zip(expected_grad.iter()) {
                        assert!((a - e).abs() < 0.1, "Expected grad close to {}, got {}", e, a);
                    }
                }
            },
            _ => {
                let a =
                    setup_grad_tensor_with_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype, &[2, 2, 2])?;
                let b = setup_grad_tensor_with_shape(
                    vec![9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                    dtype,
                    &[2, 2, 2],
                )?;
                let c = a.matmul(&b)?;
                assert_eq!(c.shape(), &[2, 2, 2]);
                assert_eq!(
                    c.to_flatten_vec::<f32>()?,
                    vec![31.0f32, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0]
                );

                c.backward()?;
                if let Some(a_grad) = a.grad()? {
                    assert_eq!(
                        a_grad.to_flatten_vec::<f32>()?,
                        vec![19.0f32, 23.0, 19.0, 23.0, 27.0, 31.0, 27.0, 31.0]
                    );
                }

                if let Some(b_grad) = b.grad()? {
                    assert_eq!(
                        b_grad.to_flatten_vec::<f32>()?,
                        vec![4.0f32, 4.0, 6.0, 6.0, 12.0, 12.0, 14.0, 14.0]
                    );
                }
            },
        }
        Ok(())
    }
}

test_ops!([
    vector_vector,
    matrix_vector,
    vector_matrix,
    matrix_matrix,
    batched_matrix_matrix
]);
