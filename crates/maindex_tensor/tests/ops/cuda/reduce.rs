use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn mean() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
    let result = tensor.mean()?;
    assert_eq!(result.to_vec()?, vec![2.5]);
    Ok(())
}

#[test]
fn sum() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0, 4.0], &device)?;
    let result = tensor.sum()?;
    assert_eq!(result.to_vec()?, vec![10.0]);
    Ok(())
}

#[test]
fn sum_with_dim() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

    // Reduction along dimension 1 (columns)
    let result = tensor.sum_with_dim(1)?;
    assert_eq!(result.to_vec()?, vec![6.0, 15.0]);

    Ok(())
}

#[test]
fn sum_to_shape() -> TensorResult<()> {
    let device = Device::cuda(0);
    let tensor =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;

    // Test reduction to [1, 3]
    let reduced = tensor.sum_to_shape(&[1, 3])?;
    assert_eq!(reduced.shape(), &[1, 3]);
    assert_eq!(reduced.to_vec()?, vec![5.0, 7.0, 9.0]);

    // Test reduction to [2, 1]
    let reduced = tensor.sum_to_shape(&[2, 1])?;
    assert_eq!(reduced.shape(), &[2, 1]);
    assert_eq!(reduced.to_vec()?, vec![6.0, 15.0]);

    Ok(())
}
