use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for test data
const TEST_DATA_F32_1D: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const TEST_DATA_F32_2D: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
const TEST_DATA_U32_2D: [[u32; 2]; 2] = [[1, 2], [3, 4]];

// Helper functions
fn setup_tensor<T: Clone + 'static>(data: Vec<T>, shape: &[usize], device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = Tensor::new(data)?;
    tensor.with_shape(shape)?;
    tensor.with_device(device)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, shape: &[usize], device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor(data, shape, device, dtype)?;
    tensor.with_grad().ok();
    Ok(tensor)
}

// Core test functions
mod test_functions {
    use super::*;

    pub fn sum_test(device: Device, dtype: DType) -> Result<()> {
        // Test 1D sum
        let x = setup_grad_tensor(TEST_DATA_F32_1D.to_vec(), &[4], device, dtype)?;
        let result = x.sum(0)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![10.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);
        }

        // Test 2D sum along dim 0
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        println!("{:?}", x);
        let result = x.sum(0)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        Ok(())
    }

    pub fn sum_all_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        let result = x.sum_all()?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![21.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        Ok(())
    }

    pub fn sum_to_shape_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        let result = x.sum_to_shape(&[1, 3])?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        }

        Ok(())
    }

    pub fn mean_test(device: Device, dtype: DType) -> Result<()> {
        // Test with floating point data (keepdims=false)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        let result = x.mean(0, false)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.5, 3.5, 4.5]);
        assert_eq!(result.shape(), &[3]);
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        }

        // Test with floating point data (keepdims=true)
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        let result = x.mean(0, true)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.5, 3.5, 4.5]);
        assert_eq!(result.shape(), &[1, 3]); // Dimension is kept
        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        }

        // Test with integer data (keepdims=false)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], device, DType::U32)?;
        let result = x.mean(0, false)?;
        // Should be automatically converted to f32
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.0, 3.0]);
        assert_eq!(result.shape(), &[2]);

        // Test with integer data (keepdims=true)
        let x = setup_tensor(TEST_DATA_U32_2D.iter().flatten().copied().collect(), &[2, 2], device, DType::U32)?;
        let result = x.mean(0, true)?;
        // Should be automatically converted to f32
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![2.0, 3.0]);
        assert_eq!(result.shape(), &[1, 2]); // Dimension is kept

        Ok(())
    }

    pub fn mean_all_test(device: Device, dtype: DType) -> Result<()> {
        // Test mean_all with keepdims=false
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        let result = x.mean_all(false)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.5]);
        assert_eq!(result.shape(), &[]); // All dimensions are reduced
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

        // Test mean_all with keepdims=true
        let x = setup_grad_tensor(TEST_DATA_F32_2D.iter().flatten().copied().collect(), &[2, 3], device, dtype)?;
        let result = x.mean_all(true)?;
        result.backward()?;
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.5]);
        assert_eq!(result.shape(), &[1, 1]); // All dimensions are kept but with size 1
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

        // Test 3D tensor with keepdims
        let x = setup_grad_tensor((0..24).map(|x| x as f32).collect(), &[2, 3, 4], device, dtype)?;

        // Test reducing middle dimension with keepdims=true
        let result = x.mean(1, true)?;
        assert_eq!(result.shape(), &[2, 1, 4]); // Middle dimension is kept but with size 1

        // Test reducing all dimensions with keepdims=true
        let result = x.mean_all(true)?;
        assert_eq!(result.shape(), &[1, 1, 1]); // All dimensions are kept but with size 1

        Ok(())
    }

    pub fn fold_test(device: Device, dtype: DType) -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = setup_grad_tensor(data, &[5], device, dtype)?;
        let unfolded = x.unfold(0, 2, 1)?;

        assert_eq!(unfolded.shape(), &[4, 2]);

        let folded = unfolded.fold(0, 2, 1)?;

        assert_eq!(folded.shape(), &[5]);
        assert_eq!(folded.to_flatten_vec::<f32>()?, vec![1.0, 4.0, 6.0, 8.0, 5.0]);

        let data_2d = TEST_DATA_F32_2D.iter().flatten().copied().collect();
        let x_2d = setup_grad_tensor(data_2d, &[2, 3], device, dtype)?;

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
}

// sum operation tests
mod sum {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sum_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sum_test(Device::CUDA(0), DType::I64)
        }
    }
}

// sum_all operation tests
mod sum_all {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sum_all_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sum_all_test(Device::CUDA(0), DType::I64)
        }
    }
}

// sum_to_shape operation tests
mod sum_to_shape {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sum_to_shape_test(Device::CUDA(0), DType::I64)
        }
    }
}

// mean operation tests
mod mean {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mean_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mean_test(Device::CUDA(0), DType::I64)
        }
    }
}

// mean_all operation tests
mod mean_all {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mean_all_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mean_all_test(Device::CUDA(0), DType::I64)
        }
    }
}

// fold operation tests
mod fold {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::fold_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::fold_test(Device::CUDA(0), DType::I64)
        }
    }
}
