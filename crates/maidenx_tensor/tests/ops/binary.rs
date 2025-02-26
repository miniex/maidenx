use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for test data
const TEST_DATA_F32: [f32; 2] = [1.0, 2.0];
const TEST_DATA_U32: [u32; 3] = [1, 2, 0];

// Helper functions
fn setup_tensor<T: Clone + 'static>(data: Vec<T>, device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = Tensor::new(data)?;
    tensor.with_device(device)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor(data, device, dtype)?;
    tensor.with_grad().ok();

    Ok(tensor)
}

// Core test functions
mod test_functions {
    use super::*;

    pub fn add_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_grad_tensor(vec![3.0f32, 4.0], device, dtype)?;

        let result = x.add(&y)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![4.0, 6.0]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0]);
        }
        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0]);
        }

        Ok(())
    }

    pub fn sub_test(device: Device, dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U32 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), device, dtype)?;
                let y = setup_tensor(vec![3u32, 1, 5], device, dtype)?;

                let result = x.sub(&y)?;

                assert_eq!(result.to_flatten_vec::<u32>()?, vec![0, 1, 0]);
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
                let y = setup_grad_tensor(vec![3.0f32, 4.0], device, dtype)?;

                let result = x.sub(&y)?;
                result.backward()?;

                assert_eq!(result.to_flatten_vec::<f32>()?, vec![-2.0, -2.0]);

                if let Some(g) = x.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 1.0]);
                }
                if let Some(g) = y.grad()? {
                    assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.0, -1.0]);
                }
            }
        }
        Ok(())
    }

    pub fn mul_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_grad_tensor(vec![3.0f32, 4.0], device, dtype)?;

        let result = x.mul(&y)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.0, 8.0]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![3.0, 4.0]);
        }
        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 2.0]);
        }

        Ok(())
    }

    pub fn div_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(vec![6.0f32, 10.0], device, dtype)?;
        let y = setup_grad_tensor(vec![2.0f32, 4.0], device, dtype)?;

        let result = x.div(&y)?;
        println!("Before:\nx: {:?}\ny: {:?}\nresult: {:?}", x, y, result);
        result.backward()?;
        println!("After:\nx: {:?}\ny: {:?}\nresult: {:?}", x, y, result);

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.0, 2.5]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.25]);
        }
        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.5, -0.625]);
        }

        Ok(())
    }

    pub fn logical_and_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![true, false, true, false], device, dtype)?;
        let y = setup_tensor(vec![true, true, false, false], device, dtype)?;

        let result = x.logical_and(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, false, false]);
        Ok(())
    }

    pub fn logical_or_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![true, false, true, false], device, dtype)?;
        let y = setup_tensor(vec![true, true, false, false], device, dtype)?;

        let result = x.logical_or(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, true, false]);
        Ok(())
    }

    pub fn logical_xor_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![true, false, true, false], device, dtype)?;
        let y = setup_tensor(vec![true, true, false, false], device, dtype)?;

        let result = x.logical_xor(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
        Ok(())
    }

    pub fn eq_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![1.0f32, 3.0], device, dtype)?;

        let result = x.eq(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn ne_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![1.0f32, 3.0], device, dtype)?;

        let result = x.ne(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true]);
        Ok(())
    }

    pub fn lt_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![1.5f32, 1.5], device, dtype)?;

        let result = x.lt(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn le_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![1.0f32, 1.5], device, dtype)?;

        let result = x.le(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn gt_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![0.5f32, 2.0], device, dtype)?;

        let result = x.gt(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn ge_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![1.0f32, 2.5], device, dtype)?;

        let result = x.ge(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn add_inplace_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![3.0f32, 4.0], device, dtype)?;

        x.add_(&y)?;

        assert_eq!(x.to_flatten_vec::<f32>()?, vec![4.0, 6.0]);
        Ok(())
    }

    pub fn sub_inplace_test(device: Device, dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U32 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), device, dtype)?;
                let y = setup_tensor(vec![1u32, 1, 0], device, dtype)?;

                x.sub_(&y)?;

                assert_eq!(x.to_flatten_vec::<u32>()?, vec![0, 1, 0]);
            }
            _ => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
                let y = setup_tensor(vec![3.0f32, 4.0], device, dtype)?;

                x.sub_(&y)?;

                assert_eq!(x.to_flatten_vec::<f32>()?, vec![-2.0, -2.0]);
            }
        }
        Ok(())
    }

    pub fn mul_inplace_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), device, dtype)?;
        let y = setup_tensor(vec![3.0f32, 4.0], device, dtype)?;

        x.mul_(&y)?;

        assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0, 8.0]);
        Ok(())
    }

    pub fn div_inplace_test(device: Device, dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![6.0f32, 8.0], device, dtype)?;
        let y = setup_tensor(vec![2.0f32, 4.0], device, dtype)?;

        x.div_(&y)?;

        assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0, 2.0]);
        Ok(())
    }
}

// Add operation tests
mod add {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::add_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::add_test(Device::CUDA(0), DType::I64)
        }
    }
}

// Sub operation tests
mod sub {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sub_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sub_test(Device::CUDA(0), DType::I64)
        }
    }
}

// Mul operation tests
mod mul {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mul_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mul_test(Device::CUDA(0), DType::I64)
        }
    }
}

// Div operation tests
mod div {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::div_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::div_test(Device::CUDA(0), DType::I64)
        }
    }
}

// logical_and operation tests
mod logical_and {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_and_test(Device::CPU, DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_and_test(Device::CPU, DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_and_test(Device::CPU, DType::I32)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_and_test(Device::CUDA(0), DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_and_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_and_test(Device::CUDA(0), DType::I32)
        }
    }
}

// logical_or operation tests
mod logical_or {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_or_test(Device::CPU, DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_or_test(Device::CPU, DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_or_test(Device::CPU, DType::I32)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_or_test(Device::CUDA(0), DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_or_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_or_test(Device::CUDA(0), DType::I32)
        }
    }
}

// logical_xor operation tests
mod logical_xor {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_xor_test(Device::CPU, DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_xor_test(Device::CPU, DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_xor_test(Device::CPU, DType::I32)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bool() -> Result<()> {
            test_functions::logical_xor_test(Device::CUDA(0), DType::BOOL)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::logical_xor_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::logical_xor_test(Device::CUDA(0), DType::I32)
        }
    }
}

// eq operation tests
mod eq {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::eq_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::eq_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::eq_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::eq_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::eq_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::eq_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::eq_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::eq_test(Device::CUDA(0), DType::F64)
        }
    }
}

// ne operation tests
mod ne {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ne_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ne_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ne_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ne_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ne_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ne_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ne_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ne_test(Device::CUDA(0), DType::F64)
        }
    }
}

// lt operation tests
mod lt {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::lt_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::lt_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::lt_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::lt_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::lt_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::lt_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::lt_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::lt_test(Device::CUDA(0), DType::F64)
        }
    }
}

// le operation tests
mod le {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::le_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::le_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::le_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::le_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::le_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::le_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::le_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::le_test(Device::CUDA(0), DType::F64)
        }
    }
}

// gt operation tests
mod gt {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::gt_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::gt_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::gt_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::gt_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::gt_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::gt_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::gt_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::gt_test(Device::CUDA(0), DType::F64)
        }
    }
}

// ge operation tests
mod ge {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ge_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ge_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ge_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ge_test(Device::CPU, DType::F64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::ge_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::ge_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::ge_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::ge_test(Device::CUDA(0), DType::F64)
        }
    }
}

// add_ operation tests
mod add_ {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::add_inplace_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::add_inplace_test(Device::CUDA(0), DType::I64)
        }
    }
}

// sub_ operation tests
mod sub_ {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sub_inplace_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::sub_inplace_test(Device::CUDA(0), DType::I64)
        }
    }
}

// mul_ operation tests
mod mul_ {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mul_inplace_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::mul_inplace_test(Device::CUDA(0), DType::I64)
        }
    }
}

// div_ operation tests
mod div_ {
    use super::*;

    mod cpu {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::div_inplace_test(Device::CPU, DType::I64)
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;

        #[test]
        fn bf16() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::BF16)
        }
        #[test]
        fn f16() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::F16)
        }
        #[test]
        fn f32() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::F32)
        }
        #[test]
        fn f64() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::F64)
        }
        #[test]
        fn u8() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::U8)
        }
        #[test]
        fn u32() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::U32)
        }
        #[test]
        fn i8() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::I8)
        }
        #[test]
        fn i32() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::I32)
        }
        #[test]
        fn i64() -> Result<()> {
            test_functions::div_inplace_test(Device::CUDA(0), DType::I64)
        }
    }
}
