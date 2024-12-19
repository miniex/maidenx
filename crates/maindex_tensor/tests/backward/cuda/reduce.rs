use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn mean_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
    a.with_grad();

    let b = a.mean()?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![0.25, 0.25, 0.25, 0.25]); // grad is 1/N for each element
    Ok(())
}

#[test]
fn sum_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
    a.with_grad();

    let b = a.sum()?;
    b.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    assert_eq!(a_grad, vec![1.0, 1.0, 1.0, 1.0]); // grad is 1 for each element
    Ok(())
}

#[test]
fn sum_with_dim_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    tensor.with_grad();

    let result = tensor.sum_with_dim(1)?;
    result.backward()?;

    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0; 6]); // Grad is 1 for each element since it's a sum

    Ok(())
}

#[test]
fn sum_to_shape_with_backward() -> TensorResult<()> {
    let device = Device::cuda(0);
    let mut tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    tensor.with_grad();

    // Test reduction to [1, 3]
    let reduced = tensor.sum_to_shape(&[1, 3])?;
    reduced.backward()?;

    // Gradient should be evenly distributed back to original tensor
    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    Ok(())
}
