#![allow(clippy::useless_vec)]

mod utils;

use maidenx_core::{dtype::DType, error::Result};
use utils::{setup_grad_tensor, setup_tensor};

mod test_functions {
    use super::*;

    pub fn index_add_inplace_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let mut input = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices = setup_tensor(vec![0i32, 1], DType::I32)?;
                let src = setup_tensor(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], dtype)?.reshape(&[2, 3])?;

                input.index_add_(0, &indices, &src)?;

                assert_eq!(input.shape(), &[2, 3]);
                assert_eq!(input.to_flatten_vec::<f32>()?, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

                let mut input2 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices2 = setup_tensor(vec![0i32, 2], DType::I32)?;
                let src2 = setup_tensor(vec![10.0f32, 30.0, 40.0, 60.0], dtype)?.reshape(&[2, 2])?;

                input2.index_add_(1, &indices2, &src2)?;

                assert_eq!(input2.shape(), &[2, 3]);
                assert_eq!(input2.to_flatten_vec::<f32>()?, vec![11.0, 2.0, 33.0, 44.0, 5.0, 66.0]);
            }
            DType::BF16 | DType::F16 => {
                let mut input = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices = setup_tensor(vec![0i32, 1], DType::I32)?;
                let src = setup_tensor(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], dtype)?.reshape(&[2, 3])?;

                input.index_add_(0, &indices, &src)?;

                assert_eq!(input.shape(), &[2, 3]);

                let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0];
                let actual = input.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }
            }
            _ => {
                let mut input = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices = setup_tensor(vec![0i32, 1], DType::I32)?;
                let src = setup_tensor(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], dtype)?.reshape(&[2, 3])?;

                input.index_add_(0, &indices, &src)?;

                assert_eq!(input.shape(), &[2, 3]);
                assert_eq!(input.to_flatten_vec::<f32>()?, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

                let mut input2 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices2 = setup_tensor(vec![0i32, 2], DType::I32)?;
                let src2 = setup_tensor(vec![10.0f32, 30.0, 40.0, 60.0], dtype)?.reshape(&[2, 2])?;

                input2.index_add_(1, &indices2, &src2)?;

                assert_eq!(input2.shape(), &[2, 3]);
                assert_eq!(input2.to_flatten_vec::<f32>()?, vec![11.0, 2.0, 33.0, 44.0, 5.0, 66.0]);

                let mut input3 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices3 = setup_tensor(vec![1i32, 2], DType::I32)?;
                let src3 = setup_tensor(vec![20.0f32, 30.0, 50.0, 60.0], dtype)?.reshape(&[2, 2])?;

                input3.index_add_(-1, &indices3, &src3)?;

                assert_eq!(input3.shape(), &[2, 3]);
                assert_eq!(input3.to_flatten_vec::<f32>()?, vec![1.0, 22.0, 33.0, 4.0, 55.0, 66.0]);
            }
        }
        Ok(())
    }

    pub fn index_select_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let input = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices = setup_tensor(vec![0i32, 0, 1], DType::I32)?;
                let result = input.index_select(0, &indices)?;

                assert_eq!(result.shape(), &[3, 3]);
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

                let input2 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices2 = setup_tensor(vec![0i32, 2], DType::I32)?;
                let result2 = input2.index_select(1, &indices2)?;

                assert_eq!(result2.shape(), &[2, 2]);
                assert_eq!(result2.to_flatten_vec::<f32>()?, vec![1.0, 3.0, 4.0, 6.0]);
            }
            DType::BF16 | DType::F16 => {
                let input = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices = setup_tensor(vec![0i32, 0, 1], DType::I32)?;
                let result = input.index_select(0, &indices)?;

                assert_eq!(result.shape(), &[3, 3]);

                let expected = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                result.backward()?;
                if let Some(g) = input.grad()? {
                    assert_eq!(g.shape(), input.shape());

                    let grad_vec = g.to_flatten_vec::<f32>()?;
                    let expected_grad = vec![2.0, 2.0, 2.0, 1.0, 1.0, 1.0];
                    for (i, (a, e)) in grad_vec.iter().zip(expected_grad.iter()).enumerate() {
                        assert!((a - e).abs() < 0.1, "Index {}: expected grad close to {}, got {}", i, e, a);
                    }
                }
            }
            _ => {
                let input = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices = setup_tensor(vec![0i32, 0, 1], DType::I32)?;
                let result = input.index_select(0, &indices)?;

                assert_eq!(result.shape(), &[3, 3]);
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

                result.backward()?;
                if let Some(g) = input.grad()? {
                    assert_eq!(g.shape(), input.shape());

                    let grad_vec = g.to_flatten_vec::<f32>()?;
                    assert_eq!(grad_vec[0], 2.0);
                    assert_eq!(grad_vec[1], 2.0);
                    assert_eq!(grad_vec[2], 2.0);
                    assert_eq!(grad_vec[3], 1.0);
                    assert_eq!(grad_vec[4], 1.0);
                    assert_eq!(grad_vec[5], 1.0);
                }

                let input2 = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices2 = setup_tensor(vec![0i32, 2], DType::I32)?;
                let result2 = input2.index_select(1, &indices2)?;

                assert_eq!(result2.shape(), &[2, 2]);
                assert_eq!(result2.to_flatten_vec::<f32>()?, vec![1.0, 3.0, 4.0, 6.0]);

                let input3 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let indices3 = setup_tensor(vec![1i32, 2], DType::I32)?;
                let result3 = input3.index_select(-1, &indices3)?;

                assert_eq!(result3.shape(), &[2, 2]);
                assert_eq!(result3.to_flatten_vec::<f32>()?, vec![2.0, 3.0, 5.0, 6.0]);

                let embeddings = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype)?.reshape(&[3, 3])?;
                let indices4 = setup_tensor(vec![2i32, 0, 1, 2], DType::I32)?;
                let result4 = embeddings.index(&indices4)?;

                assert_eq!(result4.shape(), &[4, 3]);
                assert_eq!(
                    result4.to_flatten_vec::<f32>()?,
                    vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                );
            }
        }
        Ok(())
    }

    pub fn index_put_inplace_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let mut x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0], dtype)?;
                let y = setup_tensor(vec![1.0f32, 2.0], dtype)?;
                x.index_put_(&[3], &y)?;
                assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0f32, 4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            DType::BF16 | DType::F16 => {
                let mut x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0], dtype)?;
                let y = setup_tensor(vec![1.0f32, 2.0], dtype)?;
                x.index_put_(&[3], &y)?;

                let expected = vec![3.0f32, 4.0, 5.0, 1.0, 2.0, 3.0];
                let actual = x.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }
            }
            _ => {
                let mut x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0], dtype)?;
                let y = setup_tensor(vec![1.0f32, 2.0], dtype)?;
                x.index_put_(&[3], &y)?;
                assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0f32, 4.0, 5.0, 1.0, 2.0, 3.0]);
            }
        }
        Ok(())
    }

    pub fn gather_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let input = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let index = setup_tensor(vec![0i32, 2, 1, 0, 2, 1], DType::I32)?.reshape(&[2, 3])?;

                let result = input.gather(1, &index)?;

                assert_eq!(result.shape(), &[2, 3]);
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
            }
            DType::BF16 | DType::F16 => {
                let input = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let index = setup_tensor(vec![0i32, 2, 1, 0, 2, 1], DType::I32)?.reshape(&[2, 3])?;

                let result = input.gather(1, &index)?;
                result.backward()?;

                assert_eq!(result.shape(), &[2, 3]);

                let expected = vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0];
                let actual = result.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }

                if let Some(g) = input.grad()? {
                    assert_eq!(g.shape(), input.shape());

                    let grad_sum: f32 = g.to_flatten_vec::<f32>()?.iter().sum();
                    assert!(
                        (grad_sum - (result.size() as f32)).abs() < 0.1,
                        "Expected grad sum close to {}, got {}",
                        result.size(),
                        grad_sum
                    );
                }
            }
            _ => {
                let input = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let index = setup_tensor(vec![0i32, 2, 1, 0, 2, 1], DType::I32)?.reshape(&[2, 3])?;

                let result = input.gather(1, &index)?;
                result.backward()?;

                assert_eq!(result.shape(), &[2, 3]);
                assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0]);

                if let Some(g) = input.grad()? {
                    assert_eq!(g.shape(), input.shape());

                    let grad_sum: f32 = g.to_flatten_vec::<f32>()?.iter().sum();
                    assert_eq!(grad_sum, result.size() as f32);
                }
            }
        }
        Ok(())
    }

    pub fn scatter_add_inplace_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                let mut input = setup_tensor(vec![0.0f32; 6], dtype)?.reshape(&[2, 3])?;
                let src = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let index = setup_tensor(vec![0i32, 2, 1, 2, 0, 1], DType::I32)?.reshape(&[2, 3])?;

                input.scatter_add_(1, &index, &src)?;

                assert_eq!(input.shape(), &[2, 3]);
                let expected = vec![1.0, 3.0, 2.0, 5.0, 6.0, 4.0];
                assert_eq!(input.to_flatten_vec::<f32>()?, expected);
            }
            DType::BF16 | DType::F16 => {
                let mut input = setup_tensor(vec![0.0f32; 6], dtype)?.reshape(&[2, 3])?;
                let src = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let index = setup_tensor(vec![0i32, 2, 1, 2, 0, 1], DType::I32)?.reshape(&[2, 3])?;

                input.scatter_add_(1, &index, &src)?;

                assert_eq!(input.shape(), &[2, 3]);

                let expected = vec![1.0, 3.0, 2.0, 5.0, 6.0, 4.0];
                let actual = input.to_flatten_vec::<f32>()?;
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!((a - e).abs() < 0.1, "Expected value close to {}, got {}", e, a);
                }
            }
            _ => {
                let mut input = setup_tensor(vec![0.0f32; 6], dtype)?.reshape(&[2, 3])?;
                let src = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
                let index = setup_tensor(vec![0i32, 2, 1, 2, 0, 1], DType::I32)?.reshape(&[2, 3])?;

                input.scatter_add_(1, &index, &src)?;

                assert_eq!(input.shape(), &[2, 3]);

                let expected = vec![1.0, 3.0, 2.0, 5.0, 6.0, 4.0];
                assert_eq!(input.to_flatten_vec::<f32>()?, expected);

                let mut input2 = setup_tensor(vec![0.0f32; 4], dtype)?.reshape(&[2, 2])?;
                let src2 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0], dtype)?.reshape(&[2, 2])?;
                let index2 = setup_tensor(vec![0i32, 0, 1, 1], DType::I32)?.reshape(&[2, 2])?;

                input2.scatter_add_(1, &index2, &src2)?;

                let expected2 = vec![3.0, 0.0, 0.0, 7.0];
                assert_eq!(input2.to_flatten_vec::<f32>()?, expected2);
            }
        }
        Ok(())
    }

    pub fn bincount_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                // Basic bincount with no weights
                let input = setup_tensor(vec![1i32, 2, 1, 3, 0, 1, 4, 2], dtype)?;
                let result = input.bincount(None, None)?;
                assert_eq!(result.shape(), &[5]);
                assert_eq!(result.to_flatten_vec::<i32>()?, vec![1, 3, 2, 1, 1]);

                // Bincount with weights
                let input2 = setup_tensor(vec![0i32, 1, 1, 2], dtype)?;
                let weights = setup_tensor(vec![0.5f32, 1.0, 1.5, 2.0], DType::F32)?;
                let result2 = input2.bincount(Some(&weights), None)?;
                assert_eq!(result2.to_flatten_vec::<f32>()?, vec![0.5, 2.5, 2.0]);

                // Bincount with minlength
                let input3 = setup_tensor(vec![0i32, 1], dtype)?;
                let result3 = input3.bincount(None, Some(5))?;
                assert_eq!(result3.shape(), &[5]);
                assert_eq!(result3.to_flatten_vec::<i32>()?, vec![1, 1, 0, 0, 0]);
            }
            _ => {} // Skip for floating point types
        }
        Ok(())
    }
}

test_ops!([index_add_inplace, index_select, index_put_inplace, gather, scatter_add_inplace]);

test_ops_only_integer!([bincount]);
