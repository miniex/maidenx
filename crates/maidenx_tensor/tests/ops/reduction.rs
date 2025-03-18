use maidenx_core::{
    device::{set_default_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for test data
const TEST_DATA_F32_1D: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const TEST_DATA_F32_2D: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
const TEST_DATA_U32_2D: [[u32; 2]; 2] = [[1, 2], [3, 4]];

// Helper functions
fn setup_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(not(any(feature = "cuda")))]
    set_default_device(Device::CPU);
}

fn setup_tensor<T: Clone + 'static>(data: Vec<T>, shape: &[usize], dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let mut tensor = Tensor::new(data)?;
    tensor.with_shape(shape)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, shape: &[usize], dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor(data, shape, dtype)?;
    tensor.with_grad().ok();
    Ok(tensor)
}

// Core test functions
mod test_functions {
    use super::*;

    pub fn sum_test(dtype: DType) -> Result<()> {
        // Test 1D sum (keep_dim=false)
        let x = setup_grad_tensor(TEST_DATA_F32_1D.to_vec(), &[4], dtype)?;
        let result = x.sum(0, false)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![10.0]);
        assert_eq!(result.shape(), &[]); // Scalar result
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);
        }

        // Test 1D sum (keep_dim=true)
        let x = setup_grad_tensor(TEST_DATA_F32_1D.to_vec(), &[4], dtype)?;
        let result = x.sum(0, true)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![10.0]);
        assert_eq!(result.shape(), &[1]); // Dimension is kept
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);
        }

        // Test 2D sum along dim 0 (keep_dim=false)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        println!("{:?}", x);
        let result = x.sum(0, false)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
        assert_eq!(result.shape(), &[3]); // First dimension removed
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        // Test 2D sum along dim 0 (keep_dim=true)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        println!("{:?}", x);
        let result = x.sum(0, true)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
        assert_eq!(result.shape(), &[1, 3]); // First dimension kept as 1
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        // Test 2D sum along dim 1 (keep_dim=false)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.sum(1, false)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![6.0, 15.0]);
        assert_eq!(result.shape(), &[2]); // Second dimension removed
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        // Test 2D sum along dim 1 (keep_dim=true)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.sum(1, true)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![6.0, 15.0]);
        assert_eq!(result.shape(), &[2, 1]); // Second dimension kept as 1
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        Ok(())
    }

    pub fn sum_all_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.sum_all()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![21.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        Ok(())
    }

    pub fn sum_to_shape_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.sum_to_shape(&[1, 3])?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        Ok(())
    }

    pub fn mean_test(dtype: DType) -> Result<()> {
        // Test with floating point data (keepdims=false)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.mean(0, false)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.5, 3.5, 4.5]);
        assert_eq!(result.shape(), &[3]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        }

        // Test with floating point data (keepdims=true)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.mean(0, true)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.5, 3.5, 4.5]);
        assert_eq!(result.shape(), &[1, 3]); // Dimension is kept
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        }

        // Test with integer data (keepdims=false)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], DType::U32)?;
        let result = x.mean(0, false)?;
        // Should be automatically converted to f32
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.0, 3.0]);
        assert_eq!(result.shape(), &[2]);

        // Test with integer data (keepdims=true)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], DType::U32)?;
        let result = x.mean(0, true)?;
        // Should be automatically converted to f32
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.0, 3.0]);
        assert_eq!(result.shape(), &[1, 2]); // Dimension is kept

        Ok(())
    }

    pub fn mean_all_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.mean_all()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.5]);
        if let Some(g) = x.grad()? {
            match dtype {
                DType::BF16 | DType::F16 => {
                    let grad = g.to_flatten_vec::<f32>()?;

                    assert!(
                        grad.iter().all(|&x| (x - 1.0 / 6.0).abs() < 0.001),
                        "Expected values close to 1/6 (0.16666667), got {:?}",
                        grad
                    );
                }

                _ => {
                    assert_eq!(
                        g.to_flatten_vec::<f32>()?,
                        vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
                    );
                }
            }
        }

        Ok(())
    }

    pub fn fold_test(dtype: DType) -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = setup_grad_tensor(data, &[5], dtype)?;
        let unfolded = x.unfold(0, 2, 1)?;

        assert_eq!(unfolded.shape(), &[4, 2]);

        let folded = unfolded.fold(0, 2, 1)?;

        assert_eq!(folded.shape(), &[5]);
        assert_eq!(folded.to_flatten_vec::<f32>()?, vec![1.0, 4.0, 6.0, 8.0, 5.0]);

        let data_2d = TEST_DATA_F32_2D.iter().flatten().copied().collect();
        let x_2d = setup_grad_tensor(data_2d, &[2, 3], dtype)?;

        let unfolded_2d = x_2d.unfold(1, 2, 1)?;

        assert_eq!(unfolded_2d.shape(), &[2, 2, 2]);

        let folded_2d = unfolded_2d.fold(1, 2, 1)?;

        assert_eq!(folded_2d.shape(), &[2, 3]);
        assert_eq!(folded_2d.to_flatten_vec::<f32>()?, vec![1.0, 4.0, 3.0, 4.0, 10.0, 6.0]);

        folded_2d.backward()?;

        if let Some(g) = unfolded_2d.grad()? {
            let grad_data = g.to_flatten_vec::<f32>()?;
            assert!(
                grad_data.iter().all(|&x| (x - 1.0).abs() < 1e-5),
                "Expected all gradients to be 1.0, got {:?}",
                grad_data
            );
        }

        if let Some(g) = x_2d.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 1.0, 1.0, 2.0, 1.0]);
        }

        Ok(())
    }

    pub fn max_test(dtype: DType) -> Result<()> {
        // Test with floating point data (keepdims=false)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.max(0, false)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![4.0, 5.0, 6.0]);
        assert_eq!(result.shape(), &[3]);
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for max values (second row) and 0.0 for others (first row)
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        }

        // Test with floating point data (keepdims=true)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.max(0, true)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![4.0, 5.0, 6.0]);
        assert_eq!(result.shape(), &[1, 3]); // Dimension is kept
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for max values (second row) and 0.0 for others (first row)
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        }

        // Test with integer data (keepdims=false)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], DType::U32)?;
        let result = x.max(0, false)?;
        assert_eq!(result.to_flatten_vec::<u32>()?, vec![3, 4]);
        assert_eq!(result.shape(), &[2]);

        // Test with integer data (keepdims=true)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], DType::U32)?;
        let result = x.max(0, true)?;
        assert_eq!(result.to_flatten_vec::<u32>()?, vec![3, 4]);
        assert_eq!(result.shape(), &[1, 2]); // Dimension is kept

        Ok(())
    }

    pub fn max_all_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.max_all()?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![6.0]);
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for the maximum value (6.0) and 0.0 for all others
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        }

        // Test with 1D tensor
        let x = setup_grad_tensor(TEST_DATA_F32_1D.to_vec(), &[4], dtype)?;
        let result = x.max_all()?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![4.0]);
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for the maximum value (4.0) and 0.0 for all others
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 0.0, 0.0, 1.0]);
        }

        Ok(())
    }

    pub fn min_test(dtype: DType) -> Result<()> {
        // Test with floating point data (keepdims=false)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.min(0, false)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.shape(), &[3]);
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for min values (first row) and 0.0 for others (second row)
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        }

        // Test with floating point data (keepdims=true)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.min(0, true)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.shape(), &[1, 3]); // Dimension is kept
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for min values (first row) and 0.0 for others (second row)
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        }

        // Test with integer data (keepdims=false)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], DType::U32)?;
        let result = x.min(0, false)?;
        assert_eq!(result.to_flatten_vec::<u32>()?, vec![1, 2]);
        assert_eq!(result.shape(), &[2]);

        // Test with integer data (keepdims=true)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], DType::U32)?;
        let result = x.min(0, true)?;
        assert_eq!(result.to_flatten_vec::<u32>()?, vec![1, 2]);
        assert_eq!(result.shape(), &[1, 2]); // Dimension is kept

        Ok(())
    }

    pub fn min_all_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], dtype)?;
        let result = x.min_all()?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0]);
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for the minimum value (1.0) and 0.0 for all others
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        // Test with 1D tensor
        let x = setup_grad_tensor(TEST_DATA_F32_1D.to_vec(), &[4], dtype)?;
        let result = x.min_all()?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0]);
        if let Some(g) = x.grad()? {
            // Gradient should be 1.0 for the minimum value (1.0) and 0.0 for all others
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 0.0, 0.0]);
        }

        Ok(())
    }
}

// sum operation tests
mod sum {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::sum_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::sum_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::sum_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::sum_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::sum_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::sum_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::sum_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::sum_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::sum_test(DType::I64)
    }
}

// sum_all operation tests
mod sum_all {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::sum_all_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::sum_all_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::sum_all_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::sum_all_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::sum_all_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::sum_all_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::sum_all_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::sum_all_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::sum_all_test(DType::I64)
    }
}

// sum_to_shape operation tests
mod sum_to_shape {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::sum_to_shape_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::sum_to_shape_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::sum_to_shape_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::sum_to_shape_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::sum_to_shape_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::sum_to_shape_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::sum_to_shape_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::sum_to_shape_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::sum_to_shape_test(DType::I64)
    }
}

// mean operation tests
mod mean {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::mean_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::mean_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::mean_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::mean_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::mean_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::mean_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::mean_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::mean_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::mean_test(DType::I64)
    }
}

// mean_all operation tests
mod mean_all {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::mean_all_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::mean_all_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::mean_all_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::mean_all_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::mean_all_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::mean_all_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::mean_all_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::mean_all_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::mean_all_test(DType::I64)
    }
}

// fold operation tests
mod fold {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::fold_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::fold_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::fold_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::fold_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::fold_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::fold_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::fold_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::fold_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::fold_test(DType::I64)
    }
}

// max operation tests
mod max {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::max_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::max_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::max_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::max_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::max_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::max_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::max_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::max_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::max_test(DType::I64)
    }
}

// max_all operation tests
mod max_all {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::max_all_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::max_all_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::max_all_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::max_all_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::max_all_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::max_all_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::max_all_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::max_all_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::max_all_test(DType::I64)
    }
}

// min operation tests
mod min {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::min_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::min_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::min_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::min_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::min_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::min_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::min_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::min_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::min_test(DType::I64)
    }
}

// min_all operation tests
mod min_all {
    use super::*;

    #[test]
    fn bf16() -> Result<()> {
        test_functions::min_all_test(DType::BF16)
    }
    #[test]
    fn f16() -> Result<()> {
        test_functions::min_all_test(DType::F16)
    }
    #[test]
    fn f32() -> Result<()> {
        test_functions::min_all_test(DType::F32)
    }
    #[test]
    fn f64() -> Result<()> {
        test_functions::min_all_test(DType::F64)
    }
    #[test]
    fn u8() -> Result<()> {
        test_functions::min_all_test(DType::U8)
    }
    #[test]
    fn u32() -> Result<()> {
        test_functions::min_all_test(DType::U32)
    }
    #[test]
    fn i8() -> Result<()> {
        test_functions::min_all_test(DType::I8)
    }
    #[test]
    fn i32() -> Result<()> {
        test_functions::min_all_test(DType::I32)
    }
    #[test]
    fn i64() -> Result<()> {
        test_functions::min_all_test(DType::I64)
    }
}
