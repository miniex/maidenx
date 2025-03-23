mod utils;

use maidenx_core::{dtype::DType, error::Result};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};
use utils::setup_device;

// Helper functions
pub fn setup_tensor<T: Clone + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let mut tensor = Tensor::new(data)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

pub fn setup_grad_tensor<T: Clone + 'static>(data: Vec<T>, dtype: DType) -> Result<Tensor>
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

    pub fn vector_vector_test(dtype: DType) -> Result<()> {
        let a = setup_grad_tensor(vec![1.0f32, 2.0, 3.0], dtype)?;
        let b = setup_grad_tensor(vec![4.0f32, 5.0, 6.0], dtype)?;
        let c = a.matmul(&b)?;

        assert_eq!(c.ndim(), 0);
        assert_eq!(c.to_flatten_vec::<f32>()?, vec![32.0]);

        c.backward()?;
        if let Some(a_grad) = a.grad()? {
            assert_eq!(a_grad.to_flatten_vec::<f32>()?, vec![4.0, 5.0, 6.0]);
        }
        if let Some(b_grad) = b.grad()? {
            assert_eq!(b_grad.to_flatten_vec::<f32>()?, vec![1.0, 2.0, 3.0]);
        }
        Ok(())
    }

    pub fn matrix_vector_test(dtype: DType) -> Result<()> {
        let matrix = setup_grad_tensor(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], dtype)?;
        let vector = setup_grad_tensor(vec![2.0f32, 1.0, 3.0], dtype)?;
        let result = matrix.matmul(&vector)?;

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![13.0, 31.0]);

        result.backward()?;
        if let Some(matrix_grad) = matrix.grad()? {
            assert_eq!(matrix_grad.to_flatten_vec::<f32>()?, vec![2.0, 1.0, 3.0, 2.0, 1.0, 3.0]);
        }
        if let Some(vector_grad) = vector.grad()? {
            assert_eq!(vector_grad.to_flatten_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
        }
        Ok(())
    }

    pub fn vector_matrix_test(dtype: DType) -> Result<()> {
        let vector = setup_grad_tensor(vec![2.0f32, 3.0], dtype)?;
        let matrix = setup_grad_tensor(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], dtype)?;
        let result = vector.matmul(&matrix)?;

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_flatten_vec::<f32>()?, vec![14.0, 19.0, 24.0]);

        result.backward()?;
        if let Some(vector_grad) = vector.grad()? {
            assert_eq!(vector_grad.to_flatten_vec::<f32>()?, vec![6.0, 15.0]);
        }
        if let Some(matrix_grad) = matrix.grad()? {
            assert_eq!(matrix_grad.to_flatten_vec::<f32>()?, vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
        }
        Ok(())
    }

    pub fn matrix_matrix_test(dtype: DType) -> Result<()> {
        let a = setup_grad_tensor(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]], dtype)?;
        let b = setup_grad_tensor(vec![vec![5.0f32, 6.0], vec![7.0, 8.0]], dtype)?;

        let c = a.matmul(&b)?;

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_flatten_vec::<f32>()?, vec![19.0, 22.0, 43.0, 50.0]);

        c.backward()?;
        if let Some(a_grad) = a.grad()? {
            assert_eq!(a_grad.to_flatten_vec::<f32>()?, vec![11.0, 15.0, 11.0, 15.0]);
        }
        if let Some(b_grad) = b.grad()? {
            assert_eq!(b_grad.to_flatten_vec::<f32>()?, vec![4.0, 4.0, 6.0, 6.0]);
        }
        Ok(())
    }

    pub fn batched_matrix_matrix_test(dtype: DType) -> Result<()> {
        let a = setup_grad_tensor(vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]], vec![vec![5.0, 6.0], vec![7.0, 8.0]]], dtype)?;
        let b = setup_grad_tensor(
            vec![vec![vec![9.0f32, 10.0], vec![11.0, 12.0]], vec![vec![13.0, 14.0], vec![15.0, 16.0]]],
            dtype,
        )?;

        let c = a.matmul(&b)?;

        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(c.to_flatten_vec::<f32>()?, vec![31.0, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0]);

        c.backward()?;
        if let Some(a_grad) = a.grad()? {
            assert_eq!(a_grad.to_flatten_vec::<f32>()?, vec![19.0, 23.0, 19.0, 23.0, 27.0, 31.0, 27.0, 31.0]);
        }
        if let Some(b_grad) = b.grad()? {
            assert_eq!(b_grad.to_flatten_vec::<f32>()?, vec![4.0, 4.0, 6.0, 6.0, 12.0, 12.0, 14.0, 14.0]);
        }
        Ok(())
    }
}

test_ops!([vector_vector, matrix_vector, vector_matrix, matrix_matrix, batched_matrix_matrix]);
