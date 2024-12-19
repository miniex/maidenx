use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn scalar_add_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    a.with_grad();

    let b = a.scalar_add(5.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]); // dL/da = 1
    Ok(())
}

#[test]
fn scalar_add_chain_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    a.with_grad();

    let b = a.scalar_add(5.0)?.scalar_add(3.0)?.scalar_add(2.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
    Ok(())
}

#[test]
fn scalar_sub_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    a.with_grad();

    let b = a.scalar_sub(5.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]); // dL/da = 1
    Ok(())
}

#[test]
fn scalar_sub_chain_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
    a.with_grad();

    let b = a.scalar_sub(5.0)?.scalar_sub(3.0)?.scalar_sub(2.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
    Ok(())
}

#[test]
fn scalar_mul_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    a.with_grad();

    let b = a.scalar_mul(5.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![5.0, 5.0, 5.0]); // dL/da = scalar
    Ok(())
}

#[test]
fn scalar_mul_chain_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
    a.with_grad();

    let b = a.scalar_mul(2.0)?.scalar_mul(3.0)?.scalar_mul(4.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![24.0, 24.0, 24.0]);
    Ok(())
}

#[test]
fn scalar_div_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![10.0, 20.0, 30.0], &device)?;
    a.with_grad();

    let b = a.scalar_div(5.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![0.2, 0.2, 0.2]); // dL/da = 1/scalar
    Ok(())
}

#[test]
fn scalar_div_chain_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![8.0, 16.0, 32.0], &device)?;
    a.with_grad();

    let b = a.scalar_div(2.0)?.scalar_div(2.0)?.scalar_div(2.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![0.125, 0.125, 0.125]);
    Ok(())
}

#[test]
fn scalar_mix_chain_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
    a.with_grad();

    // (((a + 2) * 3) - 4) / 2
    let b = a
        .scalar_add(2.0)?
        .scalar_mul(3.0)?
        .scalar_sub(4.0)?
        .scalar_div(2.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    // dL/da = ((1 * 3) * 1) / 2 = 1.5
    assert_eq!(a_grad, vec![1.5, 1.5, 1.5]);
    Ok(())
}

#[test]
fn pow_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
    a.with_grad();

    let b = a.pow(2.0)?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    // gradient of x^2 is 2x
    assert_eq!(a_grad, vec![4.0, 6.0, 8.0]);
    Ok(())
}

#[test]
fn pow_chain_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
    a.with_grad();

    let b = a.pow(2.0)?.pow(2.0)?; // (x^2)^2 = x^4
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    // gradient of x^4 is 4x^3
    assert_eq!(a_grad, vec![32.0, 108.0, 256.0]);
    Ok(())
}
