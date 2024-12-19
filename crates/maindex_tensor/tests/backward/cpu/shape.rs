use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn transpose_backward() -> TensorResult<()> {
    let device = Device::cpu();
    let mut tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    tensor.with_grad();

    let transposed = tensor.transpose()?;
    transposed.backward()?;

    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]); // Each value contributes 1.0
    Ok(())
}

#[test]
fn transpose_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();
    let mut tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    tensor.with_grad();

    let transposed = tensor.transpose()?.transpose()?; // Double transpose
    transposed.backward()?;

    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]); // Same as original
    Ok(())
}

#[test]
fn transpose_dim_backward() -> TensorResult<()> {
    let device = Device::cpu();
    let mut tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    tensor.with_grad();

    let transposed = tensor.transpose_dim(0, 1)?;
    transposed.backward()?;

    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0; 6]); // Same contribution for all elements
    Ok(())
}

#[test]
fn transpose_dim_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut tensor = Tensor::from_vec_with_device(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
        &device,
    )?;
    tensor.with_grad();

    let transposed = tensor
        .transpose_dim(0, 1)? // 2 x 3 x 2 -> 3 x 2 x 2
        .transpose_dim(1, 2)? // 3 x 2 x 2 -> 3 x 2 x 2 (1, 2 swap)
        .transpose_dim(0, 2)?; // 3 x 2 x 2 -> 2 x 2 x 3

    transposed.backward()?;

    let grad = tensor.grad()?.unwrap().to_vec()?;

    println!("Original Tensor Grad: {:?}", grad);

    assert_eq!(grad, vec![1.0; 12]);
    Ok(())
}

#[test]
fn reshape_backward() -> TensorResult<()> {
    let device = Device::cpu();

    // Create a tensor with gradient tracking
    let mut tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    tensor.with_grad();

    // Reshape tensor and compute a gradient
    let reshaped = tensor.reshape(&[4])?;
    reshaped.backward()?;

    // Check that the gradient for the original tensor is correct
    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn reshape_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    // Create a tensor with gradient tracking
    let mut tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    tensor.with_grad();

    // Reshape tensor twice
    let reshaped = tensor.reshape(&[4])?.reshape(&[1, 4])?;
    reshaped.backward()?;

    // Check that the gradient for the original tensor is correct
    let grad = tensor.grad()?.unwrap().to_vec()?;
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]);

    Ok(())
}
