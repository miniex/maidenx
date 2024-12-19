use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn scalar_add() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let result = tensor.scalar_add(5.0)?;
    assert_eq!(result.to_vec()?, vec![6.0, 7.0, 8.0]);
    Ok(())
}

#[test]
fn scalar_sub() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
    let result = tensor.scalar_sub(5.0)?;
    assert_eq!(result.to_vec()?, vec![5.0, 15.0, 25.0]);
    Ok(())
}

#[test]
fn scalar_mul() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![2.0, 4.0, 6.0], &device)?;
    let result = tensor.scalar_mul(3.0)?;
    assert_eq!(result.to_vec()?, vec![6.0, 12.0, 18.0]);
    Ok(())
}

#[test]
fn scalar_div() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
    let result = tensor.scalar_div(10.0)?;
    assert_eq!(result.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
#[should_panic(expected = "Cannot divide tensor by zero scalar value")]
fn scalar_div_zero() {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device).unwrap();
    tensor.scalar_div(0.0).unwrap(); // This should panic
}

#[test]
fn neg() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![1.0, -2.0, 3.0], &device)?;
    let result = tensor.neg()?;
    assert_eq!(result.to_vec()?, vec![-1.0, 2.0, -3.0]);
    Ok(())
}

#[test]
fn pow() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
    let result = tensor.pow(2.0)?;
    assert_eq!(result.to_vec()?, vec![4.0, 9.0, 16.0]);
    Ok(())
}
