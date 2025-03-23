use crate::{test_ops, test_ops_with_dtype};
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

    pub fn index_add_inplace_test(dtype: DType) -> Result<()> {
        let mut input = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let indices = setup_tensor(vec![0i64, 1], DType::I64)?;
        let src = setup_tensor(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], dtype)?.reshape(&[2, 3])?;

        input.index_add_(0, &indices, &src)?;

        assert_eq!(input.shape(), &[2, 3]);
        assert_eq!(input.to_flatten_vec::<f32>()?, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

        let mut input2 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let indices2 = setup_tensor(vec![0i64, 2], DType::I64)?;
        let src2 = setup_tensor(vec![10.0f32, 30.0, 40.0, 60.0], dtype)?.reshape(&[2, 2])?;

        input2.index_add_(1, &indices2, &src2)?;

        assert_eq!(input2.shape(), &[2, 3]);
        assert_eq!(input2.to_flatten_vec::<f32>()?, vec![11.0, 2.0, 33.0, 44.0, 5.0, 66.0]);

        let mut input3 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let indices3 = setup_tensor(vec![1i64, 2], DType::I64)?;
        let src3 = setup_tensor(vec![20.0f32, 30.0, 50.0, 60.0], dtype)?.reshape(&[2, 2])?;

        input3.index_add_(-1, &indices3, &src3)?;

        assert_eq!(input3.shape(), &[2, 3]);
        assert_eq!(input3.to_flatten_vec::<f32>()?, vec![1.0, 22.0, 33.0, 4.0, 55.0, 66.0]);

        Ok(())
    }

    pub fn index_select_test(dtype: DType) -> Result<()> {
        // Test 1: Basic index_select along dimension 0
        let input = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let indices = setup_tensor(vec![0i64, 0, 1], DType::I64)?;
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

        // Test 2: index_select along dimension 1
        let input2 = setup_grad_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let indices2 = setup_tensor(vec![0i64, 2], DType::I64)?;
        let result2 = input2.index_select(1, &indices2)?;

        // Check shape and values
        assert_eq!(result2.shape(), &[2, 2]);
        assert_eq!(result2.to_flatten_vec::<f32>()?, vec![1.0, 3.0, 4.0, 6.0]);

        // Test 3: index_select with negative dimension
        let input3 = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)?.reshape(&[2, 3])?;
        let indices3 = setup_tensor(vec![1i64, 2], DType::I64)?;
        let result3 = input3.index_select(-1, &indices3)?; // -1 refers to last dimension (1)

        assert_eq!(result3.shape(), &[2, 2]);
        assert_eq!(result3.to_flatten_vec::<f32>()?, vec![2.0, 3.0, 5.0, 6.0]);

        // Test 4: Using index method (embedding-like functionality)
        let embeddings = setup_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype)?.reshape(&[3, 3])?;
        let indices4 = setup_tensor(vec![2i64, 0, 1, 2], DType::I64)?;
        let result4 = embeddings.index(&indices4)?;

        assert_eq!(result4.shape(), &[4, 3]);
        assert_eq!(
            result4.to_flatten_vec::<f32>()?,
            vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );

        Ok(())
    }

    pub fn index_put_inplace_test(dtype: DType) -> Result<()> {
        let mut x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0], dtype)?;
        let y = setup_tensor(vec![1.0f32, 2.0], dtype)?;
        x.index_put_(&[3], &y)?;
        assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0f32, 4.0, 5.0, 1.0, 2.0, 3.0]);
        Ok(())
    }

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

    pub fn bincount_test(dtype: DType) -> Result<()> {
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

        Ok(())
    }
}

test_ops!([
    index_add_inplace,
    index_select,
    index_put_inplace,
    gather,
    // inplace
    scatter_add_inplace
]);

test_ops_with_dtype!([
    bincount: [U8, U32, I8, I32, I64]
]);
