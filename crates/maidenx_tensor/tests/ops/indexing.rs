use crate::test_ops_without_8byte;
use maidenx_core::{
    device::{set_default_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

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

    pub fn gather_test(dtype: DType) -> Result<()> {
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

        Ok(())
    }

    pub fn scatter_add_inplace_test(dtype: DType) -> Result<()> {
        let mut input = setup_tensor(vec![0.0f32; 6], dtype)?.reshape(&[2, 3])?;
        let src = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let index = setup_tensor(vec![0i32, 2, 1, 2, 0, 1], DType::I32)?.reshape(&[2, 3])?;

        input.scatter_add_(1, &index, &src)?;

        assert_eq!(input.shape(), input.shape());

        let expected = vec![1.0, 3.0, 2.0, 5.0, 6.0, 4.0];
        assert_eq!(input.to_flatten_vec::<f32>()?, expected);

        let mut input2 = setup_tensor(vec![0.0f32; 4], dtype)?.reshape(&[2, 2])?;
        let src2 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0], dtype)?.reshape(&[2, 2])?;
        let index2 = setup_tensor(vec![0i32, 0, 1, 1], DType::I32)?.reshape(&[2, 2])?;

        input2.scatter_add_(1, &index2, &src2)?;

        let expected2 = vec![3.0, 0.0, 0.0, 7.0];
        assert_eq!(input2.to_flatten_vec::<f32>()?, expected2);

        Ok(())
    }
}

test_ops_without_8byte!([
    gather,
    // inplace
    scatter_add_inplace
]);
