use crate::test_ops;
use maidenx_core::{
    device::{set_default_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for test data
const TEST_DATA_F32: [f32; 2] = [1.0, 2.0];
const TEST_DATA_U32: [u32; 3] = [1, 2, 0];

// Helper functions
fn setup_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(not(any(feature = "cuda")))]
    set_default_device(Device::CPU);
}

fn setup_tensor<T: Clone + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let mut tensor = Tensor::new(data)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = setup_tensor(data, dtype)?;
    tensor.with_grad().ok();

    Ok(tensor)
}

// Core test functions
mod test_functions {
    use super::*;

    pub fn add_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_grad_tensor(vec![3.0f32, 4.0], dtype)?;

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

    pub fn sub_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U32 => {
                let x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let y = setup_tensor(vec![3u32, 1, 5], dtype)?;

                let result = x.sub(&y)?;

                assert_eq!(result.to_flatten_vec::<u32>()?, vec![0, 1, 0]);
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let y = setup_grad_tensor(vec![3.0f32, 4.0], dtype)?;

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

    pub fn mul_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_grad_tensor(vec![3.0f32, 4.0], dtype)?;

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

    pub fn div_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(vec![6.0f32, 10.0], dtype)?;
        let y = setup_grad_tensor(vec![2.0f32, 4.0], dtype)?;

        let result = x.div(&y)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.0, 2.5]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.5, 0.25]);
        }
        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![-1.5, -0.625]);
        }

        Ok(())
    }

    pub fn maximum_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(vec![1.0f32, 5.0, 3.0], dtype)?;
        let y = setup_grad_tensor(vec![3.0f32, 2.0, 3.0], dtype)?;
        let result = x.maximum(&y)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![3.0, 5.0, 3.0]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 1.0, 0.5]);
        }

        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 0.5]);
        }

        Ok(())
    }

    pub fn minimum_test(dtype: DType) -> Result<()> {
        let x = setup_grad_tensor(vec![1.0f32, 5.0, 3.0], dtype)?;
        let y = setup_grad_tensor(vec![3.0f32, 2.0, 3.0], dtype)?;
        let result = x.minimum(&y)?;
        result.backward()?;

        assert_eq!(result.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0]);

        if let Some(g) = x.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![1.0, 0.0, 0.5]);
        }

        if let Some(g) = y.grad()? {
            assert_eq!(g.to_flatten_vec::<f32>()?, vec![0.0, 1.0, 0.5]);
        }

        Ok(())
    }

    pub fn logical_and_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![true, false, true, false], dtype)?;
        let y = setup_tensor(vec![true, true, false, false], dtype)?;

        let result = x.logical_and(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false, false, false]);
        Ok(())
    }

    pub fn logical_or_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![true, false, true, false], dtype)?;
        let y = setup_tensor(vec![true, true, false, false], dtype)?;

        let result = x.logical_or(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, true, true, false]);
        Ok(())
    }

    pub fn logical_xor_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(vec![true, false, true, false], dtype)?;
        let y = setup_tensor(vec![true, true, false, false], dtype)?;

        let result = x.logical_xor(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true, true, false]);
        Ok(())
    }

    pub fn eq_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![1.0f32, 3.0], dtype)?;

        let result = x.eq(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn ne_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![1.0f32, 3.0], dtype)?;

        let result = x.ne(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![false, true]);
        Ok(())
    }

    pub fn lt_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![1.5f32, 1.5], dtype)?;

        let result = x.lt(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn le_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![1.0f32, 1.5], dtype)?;

        let result = x.le(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn gt_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![0.5f32, 2.0], dtype)?;

        let result = x.gt(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn ge_test(dtype: DType) -> Result<()> {
        let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![1.0f32, 2.5], dtype)?;

        let result = x.ge(&y)?;

        assert_eq!(result.to_flatten_vec::<bool>()?, vec![true, false]);
        Ok(())
    }

    pub fn add_inplace_test(dtype: DType) -> Result<()> {
        let mut x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![3.0f32, 4.0], dtype)?;

        x.add_(&y)?;

        assert_eq!(x.to_flatten_vec::<f32>()?, vec![4.0, 6.0]);
        Ok(())
    }

    pub fn sub_inplace_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U32 => {
                let mut x = setup_tensor(TEST_DATA_U32.to_vec(), dtype)?;
                let y = setup_tensor(vec![1u32, 1, 0], dtype)?;

                x.sub_(&y)?;

                assert_eq!(x.to_flatten_vec::<u32>()?, vec![0, 1, 0]);
            }
            _ => {
                let mut x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let y = setup_tensor(vec![3.0f32, 4.0], dtype)?;

                x.sub_(&y)?;

                assert_eq!(x.to_flatten_vec::<f32>()?, vec![-2.0, -2.0]);
            }
        }
        Ok(())
    }

    pub fn mul_inplace_test(dtype: DType) -> Result<()> {
        let mut x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
        let y = setup_tensor(vec![3.0f32, 4.0], dtype)?;

        x.mul_(&y)?;

        assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0, 8.0]);
        Ok(())
    }

    pub fn div_inplace_test(dtype: DType) -> Result<()> {
        let mut x = setup_tensor(vec![6.0f32, 8.0], dtype)?;
        let y = setup_tensor(vec![2.0f32, 4.0], dtype)?;

        x.div_(&y)?;

        assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0, 2.0]);
        Ok(())
    }
}

test_ops!([
    add,
    sub,
    mul,
    div,
    maximum,
    minimum,
    // logical
    logical_and,
    logical_or,
    logical_xor,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    // inplace
    add_inplace,
    sub_inplace,
    mul_inplace,
    div_inplace
]);
