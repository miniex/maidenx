use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn transpose() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    let result = tensor.transpose()?;

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    Ok(())
}

#[test]
fn transpose_square() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    let result = tensor.transpose()?;

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec()?, vec![1.0, 3.0, 2.0, 4.0]);
    Ok(())
}

#[test]
fn transpose_invalid_dims() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(tensor.transpose_dim(-1, 1).is_err());
    assert!(tensor.transpose_dim(1, 2).is_err());
    assert!(tensor.transpose_dim(2, 1).is_err());
}

#[test]
fn transpose_non_2d() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(tensor.transpose().is_err());

    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 1, 2]).unwrap();
    assert!(tensor.transpose().is_err());
}

#[test]
fn transpose_chain() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    let transposed1 = tensor.transpose()?;
    let transposed2 = transposed1.transpose()?;

    // Double transpose should return the original tensor
    assert_eq!(transposed2.shape(), tensor.shape());
    assert_eq!(transposed2.to_vec()?, tensor.to_vec()?);
    Ok(())
}

#[test]
fn transpose_dim() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_vec_with_device(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
        &device,
    )?;
    let result = tensor.transpose_dim(1, 2)?;

    assert_eq!(result.shape(), &[2, 2, 3]);
    assert_eq!(
        result.to_vec()?,
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 8.0, 10.0, 12.0]
    );
    Ok(())
}

#[test]
fn transpose_dim_identity() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    let result = tensor.transpose_dim(1, 1)?;

    assert_eq!(result.shape(), tensor.shape());
    assert_eq!(result.to_vec()?, tensor.to_vec()?);
    Ok(())
}

#[test]
fn reshape() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

    // Reshape to 1D
    let reshaped = tensor.reshape(&[6])?;
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(reshaped.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Reshape back to 2D
    let reshaped_back = reshaped.reshape(&[3, 2])?;
    assert_eq!(reshaped_back.shape(), &[3, 2]);
    assert_eq!(reshaped_back.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    Ok(())
}

#[test]
fn reshape_invalid() {
    let device = Device::cuda(0);
    let tensor = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    // Invalid reshape
    assert!(tensor.unwrap().reshape(&[4]).is_err());
}
