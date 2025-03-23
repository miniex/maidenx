use crate::test_ops;
use maidenx_core::{
    device::{set_default_device, Device},
    dtype::DType,
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Constants for test data
const TEST_DATA_F32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const TEST_DATA_BOOL: [bool; 4] = [true, false, false, true];

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

fn verify_tensor<T: std::fmt::Debug + PartialEq + Default + Clone + 'static>(tensor: &Tensor, expected: Vec<T>) -> Result<()> {
    assert_eq!(tensor.to_flatten_vec::<T>()?, expected);
    Ok(())
}

// Core test functions
mod test_functions {
    use super::*;

    pub fn view_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let viewed = x.view(&[1, 2, 1, 2])?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);
                verify_tensor(&viewed, TEST_DATA_F32.to_vec())?;
            }
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let viewed = x.view(&[1, 2, 1, 2])?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);
                verify_tensor(&viewed, TEST_DATA_BOOL.to_vec())?;
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let viewed = x.view(&[1, 2, 1, 2])?;
                let y = x.view(&[2, 2])?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(viewed.shape(), &[1, 2, 1, 2]);
                verify_tensor(&viewed, TEST_DATA_F32.to_vec())?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![1.0; 4])?;
                }
            }
        }
        Ok(())
    }

    pub fn squeeze_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);
                verify_tensor(&squeezed, TEST_DATA_F32.to_vec())?;
            }
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);
                verify_tensor(&squeezed, TEST_DATA_BOOL.to_vec())?;
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze(2)?;
                let y = x.squeeze(1)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(squeezed.shape(), &[1, 2, 2]);
                verify_tensor(&squeezed, TEST_DATA_F32.to_vec())?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![1.0; 4])?;
                }
            }
        }
        Ok(())
    }

    pub fn squeeze_all_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;

                assert_eq!(squeezed.shape(), &[2, 2]);
                verify_tensor(&squeezed, TEST_DATA_F32.to_vec())?;
            }
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;

                assert_eq!(squeezed.shape(), &[2, 2]);
                verify_tensor(&squeezed, TEST_DATA_BOOL.to_vec())?;
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?.view(&[1, 2, 1, 2])?;
                let squeezed = x.squeeze_all()?;
                let y = x.squeeze_all()?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(squeezed.shape(), &[2, 2]);
                verify_tensor(&squeezed, TEST_DATA_F32.to_vec())?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![1.0; 4])?;
                }
            }
        }
        Ok(())
    }

    pub fn unsqueeze_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
                verify_tensor(&unsqueezed, TEST_DATA_F32.to_vec())?;
            }
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
                verify_tensor(&unsqueezed, TEST_DATA_BOOL.to_vec())?;
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?.view(&[2, 2])?;
                let unsqueezed = x.unsqueeze(1)?;
                let y = x.unsqueeze(0)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
                verify_tensor(&unsqueezed, TEST_DATA_F32.to_vec())?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![1.0; 4])?;
                }
            }
        }
        Ok(())
    }

    pub fn transpose_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let matrix_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
                let x = setup_tensor(matrix_data, dtype)?;
                let transposed = x.transpose(0, 1)?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);
                verify_tensor(&transposed, vec![1.0, 3.0, 2.0, 4.0])?;
            }
            DType::BOOL => {
                let matrix_data = vec![vec![true, false], vec![false, true]];
                let x = setup_tensor(matrix_data, dtype)?;
                let transposed = x.transpose(0, 1)?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);
                verify_tensor(&transposed, vec![true, false, false, true])?;
            }
            _ => {
                let matrix_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
                let x = setup_grad_tensor(matrix_data, dtype)?;
                let transposed = x.transpose(0, 1)?;
                let y = x.transpose(0, 1)?;
                let z = y.sum_all()?;
                z.backward()?;

                assert_eq!(transposed.shape(), &[2, 2]);
                assert_eq!(transposed.strides(), &[1, 2]);
                verify_tensor(&transposed, vec![1.0, 3.0, 2.0, 4.0])?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![1.0, 1.0, 1.0, 1.0])?;
                }
            }
        }
        Ok(())
    }

    pub fn slice_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_tensor(data.clone(), dtype)?.view(&[2, 3])?;

                let sliced1 = x.slice(0, 0, Some(1), 1)?;
                let sliced2 = x.slice(1, 1, Some(3), 1)?;
                let sliced3 = x.slice(1, 0, Some(3), 2)?;

                assert_eq!(sliced1.shape(), &[1, 3]);
                assert_eq!(sliced2.shape(), &[2, 2]);
                assert_eq!(sliced3.shape(), &[2, 2]);

                verify_tensor(&sliced1, vec![1.0, 2.0, 3.0])?;
                verify_tensor(&sliced2, vec![2.0, 3.0, 5.0, 6.0])?;
                verify_tensor(&sliced3, vec![1.0, 3.0, 4.0, 6.0])?;
            }
            DType::BOOL => {
                let data = vec![true, false, true, false, true, false];
                let x = setup_tensor(data.clone(), dtype)?.view(&[2, 3])?;

                let sliced1 = x.slice(0, 0, Some(1), 1)?;
                let sliced2 = x.slice(1, 1, Some(3), 1)?;
                let sliced3 = x.slice(1, 0, Some(3), 2)?;

                assert_eq!(sliced1.shape(), &[1, 3]);
                assert_eq!(sliced2.shape(), &[2, 2]);
                assert_eq!(sliced3.shape(), &[2, 2]);

                verify_tensor(&sliced1, vec![true, false, true])?;
                verify_tensor(&sliced2, vec![false, true, true, false])?;
                verify_tensor(&sliced3, vec![true, true, false, false])?;
            }
            _ => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_grad_tensor(data.clone(), dtype)?.view(&[2, 3])?;

                let sliced = x.slice(0, 1, Some(2), 1)?;
                let z = sliced.sum_all()?;
                z.backward()?;

                assert_eq!(sliced.shape(), &[1, 3]);
                verify_tensor(&sliced, vec![4.0, 5.0, 6.0])?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
                }

                let neg_sliced = x.slice(0, -1, Some(0), -1)?;
                assert_eq!(neg_sliced.shape(), &[1, 3]);
                verify_tensor(&neg_sliced, vec![4.0, 5.0, 6.0])?;
            }
        }
        Ok(())
    }

    pub fn unfold_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_tensor(data.clone(), dtype)?.view(&[2, 3])?;

                let unfolded1 = x.unfold(1, 2, 1)?;
                let unfolded2 = x.unfold(0, 1, 1)?;

                assert_eq!(unfolded1.shape(), &[2, 2, 2]);
                assert_eq!(unfolded2.shape(), &[2, 1, 3]);

                verify_tensor(&unfolded1, vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0])?;
                verify_tensor(&unfolded2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
            }
            DType::BOOL => {
                let data = vec![true, false, true, false, true, false];
                let x = setup_tensor(data.clone(), dtype)?.view(&[2, 3])?;

                let unfolded1 = x.unfold(1, 2, 1)?;
                let unfolded2 = x.unfold(0, 1, 1)?;

                assert_eq!(unfolded1.shape(), &[2, 2, 2]);
                assert_eq!(unfolded2.shape(), &[2, 1, 3]);

                verify_tensor(&unfolded1, vec![true, false, false, true, false, true, true, false])?;
                verify_tensor(&unfolded2, vec![true, false, true, false, true, false])?;
            }
            _ => {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x = setup_grad_tensor(data.clone(), dtype)?.view(&[2, 3])?;

                let unfolded = x.unfold(1, 2, 1)?;
                let z = unfolded.sum_all()?;
                z.backward()?;

                assert_eq!(unfolded.shape(), &[2, 2, 2]);
                verify_tensor(&unfolded, vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0])?;

                if let Some(g) = x.grad()? {
                    verify_tensor(&g, vec![1.0, 2.0, 1.0, 1.0, 2.0, 1.0])?;
                }

                let large_step_unfolded = x.unfold(1, 2, 2)?;
                assert_eq!(large_step_unfolded.shape(), &[2, 1, 2]);
                verify_tensor(&large_step_unfolded, vec![1.0, 2.0, 4.0, 5.0])?;
            }
        }
        Ok(())
    }

    pub fn reshape_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let reshaped = x.reshape(&[2, 2])?;

                assert_eq!(reshaped.shape(), &[2, 2]);
                verify_tensor(&reshaped, TEST_DATA_F32.to_vec())?;
            }
            DType::BOOL => {
                let x = setup_tensor(TEST_DATA_BOOL.to_vec(), dtype)?;
                let reshaped = x.reshape(&[2, 2])?;

                assert_eq!(reshaped.shape(), &[2, 2]);
                verify_tensor(&reshaped, TEST_DATA_BOOL.to_vec())?;
            }
            _ => {
                let x = setup_grad_tensor(TEST_DATA_F32.to_vec(), dtype)?;
                let reshaped = x.reshape(&[2, 2])?;
                reshaped.backward()?;

                assert_eq!(reshaped.shape(), &[2, 2]);
                verify_tensor(&reshaped, TEST_DATA_F32.to_vec())?;

                if let Some(g) = x.grad()? {
                    match dtype {
                        DType::BOOL => verify_tensor(&g, vec![true; 4])?,
                        _ => verify_tensor(&g, vec![1.0; 4])?,
                    }
                }
            }
        }
        Ok(())
    }

    pub fn broadcast_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(vec![1.0, 2.0], dtype)?;
                let broadcasted = x.broadcast(&[3, 2])?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                verify_tensor(&broadcasted, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0])?;
            }
            DType::BOOL => {
                let x = setup_tensor(vec![true, false], dtype)?;
                let broadcasted = x.broadcast(&[3, 2])?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                verify_tensor(&broadcasted, vec![true, false, true, false, true, false])?;
            }
            _ => {
                let x = setup_grad_tensor(vec![1.0, 2.0], dtype)?;
                let broadcasted = x.broadcast(&[3, 2])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 2]);
                verify_tensor(&broadcasted, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0])?;

                if let Some(g) = x.grad()? {
                    match dtype {
                        DType::BOOL => verify_tensor(&g, vec![true; 2])?,
                        _ => verify_tensor(&g, vec![3.0, 3.0])?,
                    }
                }
            }
        }
        Ok(())
    }

    pub fn broadcast_left_test(dtype: DType) -> Result<()> {
        match dtype {
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let x = setup_tensor(vec![1.0, 2.0], dtype)?;
                let broadcasted = x.broadcast_left(&[3, 4])?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0, 2.0].repeat(12);
                verify_tensor(&broadcasted, expected)?;
            }
            DType::BOOL => {
                let x = setup_tensor(vec![true, false], dtype)?;
                let broadcasted = x.broadcast_left(&[3, 4])?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [true, false].repeat(12);
                verify_tensor(&broadcasted, expected)?;
            }
            _ => {
                let x = setup_grad_tensor(vec![1.0, 2.0], dtype)?;
                let broadcasted = x.broadcast_left(&[3, 4])?;
                broadcasted.backward()?;

                assert_eq!(broadcasted.shape(), &[3, 4, 2]);
                let expected = [1.0, 2.0].repeat(12);
                verify_tensor(&broadcasted, expected)?;

                if let Some(g) = x.grad()? {
                    match dtype {
                        DType::BOOL => verify_tensor(&g, vec![true; 2])?,
                        _ => verify_tensor(&g, vec![12.0, 12.0])?,
                    }
                }
            }
        }
        Ok(())
    }
}

test_ops!([
    // view
    view,
    squeeze,
    squeeze_all,
    unsqueeze,
    transpose,
    slice,
    unfold,
    // reshape
    reshape,
    // broadcast
    broadcast,
    broadcast_left
]);
