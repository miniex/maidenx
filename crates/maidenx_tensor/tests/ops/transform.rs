use maidenx_core::{device::Device, dtype::DType, error::Result};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

#[test]
fn view() -> Result<()> {
    let mut x = Tensor::new_with_spec(vec![1.0f32, 2.0, 3.0, 4.0], Device::CPU)?;
    x.with_grad()?;
    let viewed = x.view(&[1, 2, 1, 2])?;
    let y = x.view(&[2, 2])?;
    let z = y.sum_all()?;
    z.backward()?;

    assert_eq!(viewed.shape(), &[1, 2, 1, 2]);

    if let Some(g) = x.grad()? {
        let grad_x = g.to_flatten_vec::<f32>()?;
        assert_eq!(grad_x, vec![1.0, 1.0, 1.0, 1.0]);
    }

    Ok(())
}

#[test]
fn squeeze() -> Result<()> {
    let mut x = Tensor::new_with_spec(vec![1.0f32, 2.0, 3.0, 4.0], Device::CPU)?.view(&[1, 2, 1, 2])?;
    x.with_grad()?;
    let squeezed = x.squeeze(2)?;
    let y = x.squeeze(1)?;
    let z = y.sum_all()?;
    z.backward()?;

    assert_eq!(squeezed.shape(), &[1, 2, 2]);

    if let Some(g) = x.grad()? {
        let grad_x = g.to_flatten_vec::<f32>()?;
        assert_eq!(grad_x, vec![1.0, 1.0, 1.0, 1.0]);
    }

    Ok(())
}

#[test]
fn squeeze_all() -> Result<()> {
    let mut x = Tensor::new_with_spec(vec![1.0f32, 2.0, 3.0, 4.0], Device::CPU)?.view(&[1, 2, 1, 2])?;
    x.with_grad()?;
    let squeezed = x.squeeze_all()?;
    let y = x.squeeze_all()?;
    let z = y.sum_all()?;
    z.backward()?;

    assert_eq!(squeezed.shape(), &[2, 2]);

    if let Some(g) = x.grad()? {
        let grad_x = g.to_flatten_vec::<f32>()?;
        assert_eq!(grad_x, vec![1.0, 1.0, 1.0, 1.0]);
    }

    Ok(())
}

#[test]
fn unsqueeze() -> Result<()> {
    let mut x = Tensor::new_with_spec(vec![1.0f32, 2.0, 3.0, 4.0], Device::CPU)?.view(&[2, 2])?;
    x.with_grad()?;
    let unsqueezed = x.unsqueeze(1)?;
    let y = x.unsqueeze(0)?; // [1, 2]
    let z = y.sum_all()?;
    z.backward()?;

    assert_eq!(unsqueezed.shape(), &[2, 1, 2]);

    if let Some(g) = x.grad()? {
        let grad_x = g.to_flatten_vec::<f32>()?;
        assert_eq!(grad_x, vec![1.0, 1.0, 1.0, 1.0]);
    }

    Ok(())
}

#[test]
fn transpose() -> Result<()> {
    let mut x = Tensor::new_with_spec(vec![vec![1.0f32, 2.0], vec![3.0, 4.0]], Device::CPU)?;
    x.with_grad()?;
    let transposed = x.transpose(0, 1)?;
    let y = x.transpose(0, 1)?;
    let z = y.sum_all()?;
    z.backward()?;

    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.strides(), &[1, 2]);
    assert_eq!(transposed.to_flatten_vec::<f32>()?, vec![1.0, 3.0, 2.0, 4.0]);

    if let Some(g) = x.grad()? {
        let grad_x = g.to_flatten_vec::<f32>()?;
        assert_eq!(grad_x, vec![1.0, 1.0, 1.0, 1.0]);
    }

    Ok(())
}

macro_rules! create_test_module {
    ($mod_name:ident, $test_fn:ident, $device:expr) => {
        mod $mod_name {
            use super::*;

            macro_rules! test_for_dtype {
                ($name:ident, $dtype:expr) => {
                    #[test]
                    fn $name() -> Result<()> {
                        $test_fn($device, $dtype)
                    }
                };
            }

            test_for_dtype!(bf16, DType::BF16);
            test_for_dtype!(f16, DType::F16);
            test_for_dtype!(f32, DType::F32);
            test_for_dtype!(f64, DType::F64);
            test_for_dtype!(bool, DType::BOOL);
            test_for_dtype!(u8, DType::U8);
            test_for_dtype!(u32, DType::U32);
            test_for_dtype!(i8, DType::I8);
            test_for_dtype!(i32, DType::I32);
            test_for_dtype!(i64, DType::I64);
        }
    };
}

mod reshape {
    use super::*;
    create_test_module!(cpu, reshape_, Device::CPU);
    #[cfg(feature = "cuda")]
    create_test_module!(cuda, reshape_, Device::CUDA(0));
}

mod broadcast {
    use super::*;
    create_test_module!(cpu, broadcast_, Device::CPU);
    #[cfg(feature = "cuda")]
    create_test_module!(cuda, broadcast_, Device::CUDA(0));
}

mod broadcast_left {
    use super::*;
    create_test_module!(cpu, broadcast_left_, Device::CPU);
    #[cfg(feature = "cuda")]
    create_test_module!(cuda, broadcast_left_, Device::CUDA(0));
}

fn reshape_(device: Device, dtype: DType) -> Result<()> {
    match dtype {
        DType::U8 | DType::U32 => {
            let x = setup_tensor(vec![1.0, 2.0, 3.0, 4.0], device, dtype)?;
            let reshaped = x.reshape(&[2, 2])?;

            assert_eq!(reshaped.shape(), &[2, 2]);
            verify_tensor(&reshaped, vec![1.0, 2.0, 3.0, 4.0])?;
        }
        DType::BOOL => {
            let x = setup_tensor(vec![true, false, false, true], device, dtype)?;
            let reshaped = x.reshape(&[2, 2])?;

            assert_eq!(reshaped.shape(), &[2, 2]);
            verify_tensor(&reshaped, vec![true, false, false, true])?;
        }
        _ => {
            let mut x = setup_tensor(vec![1.0, 2.0, 3.0, 4.0], device, dtype)?;
            x.with_grad()?;
            let reshaped = x.reshape(&[2, 2])?;
            reshaped.backward()?;

            assert_eq!(reshaped.shape(), &[2, 2]);
            verify_tensor(&reshaped, vec![1.0, 2.0, 3.0, 4.0])?;

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

fn broadcast_(device: Device, dtype: DType) -> Result<()> {
    match dtype {
        DType::U8 | DType::U32 => {
            let x = setup_tensor(vec![1.0, 2.0], device, dtype)?;
            let broadcasted = x.broadcast(&[3, 2])?;
            assert_eq!(broadcasted.shape(), &[3, 2]);
            verify_tensor(&broadcasted, [1.0, 2.0].repeat(3))?;
        }
        DType::BOOL => {
            let x = setup_tensor(vec![true, false], device, dtype)?;
            let broadcasted = x.broadcast(&[3, 2])?;
            assert_eq!(broadcasted.shape(), &[3, 2]);
            verify_tensor(&broadcasted, [true, false].repeat(3))?;
        }
        _ => {
            let mut x = setup_tensor(vec![1.0, 2.0], device, dtype)?;
            x.with_grad()?;
            let broadcasted = x.broadcast(&[3, 2])?;
            broadcasted.backward()?;
            assert_eq!(broadcasted.shape(), &[3, 2]);
            verify_tensor(&broadcasted, [1.0, 2.0].repeat(3))?;
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

fn broadcast_left_(device: Device, dtype: DType) -> Result<()> {
    match dtype {
        DType::U8 | DType::U32 => {
            let x = setup_tensor(vec![1.0, 2.0], device, dtype)?;
            let broadcasted = x.broadcast_left(&[3, 4])?;
            assert_eq!(broadcasted.shape(), &[3, 4, 2]);
            verify_tensor(&broadcasted, [1.0, 2.0].repeat(12))?;
        }
        DType::BOOL => {
            let x = setup_tensor(vec![true, false], device, dtype)?;
            let broadcasted = x.broadcast_left(&[3, 4])?;
            assert_eq!(broadcasted.shape(), &[3, 4, 2]);
            verify_tensor(&broadcasted, [true, false].repeat(12))?;
        }
        _ => {
            let mut x = setup_tensor(vec![1.0, 2.0], device, dtype)?;
            x.with_grad()?;
            let broadcasted = x.broadcast_left(&[3, 4])?;
            broadcasted.backward()?;
            assert_eq!(broadcasted.shape(), &[3, 4, 2]);
            verify_tensor(&broadcasted, [1.0, 2.0].repeat(12))?;
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

fn setup_tensor<T: Clone + 'static>(data: Vec<T>, device: Device, dtype: DType) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    let mut tensor = Tensor::new(data)?;
    tensor.with_device(device)?;
    tensor.with_dtype(dtype)?;
    Ok(tensor)
}

fn verify_tensor<T: std::fmt::Debug + PartialEq + Default + Clone + 'static>(tensor: &Tensor, expected: Vec<T>) -> Result<()> {
    assert_eq!(tensor.to_flatten_vec::<T>()?, expected);

    Ok(())
}
