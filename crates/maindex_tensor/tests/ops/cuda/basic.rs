use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn add() -> TensorResult<()> {
    let device = Device::cuda(0);

    // 1D tensor addition
    let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let c = a.add(&b)?;
    assert_eq!(c.to_vec()?, vec![5.0, 7.0, 9.0]);

    // 2D tensor addition
    let a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
    let c = a.add(&b)?;
    assert_eq!(c.to_vec()?, vec![6.0, 8.0, 10.0, 12.0]);

    Ok(())
}

#[test]
fn add_shape_mismatch() -> TensorResult<()> {
    let device = Device::cuda(0);

    let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let b = Tensor::from_device(vec![1.0, 2.0], &device)?;
    assert!(a.add(&b).is_err());
    Ok(())
}

#[test]
fn sub() -> TensorResult<()> {
    let device = Device::cuda(0);

    let a = Tensor::from_device(vec![5.0, 7.0, 9.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let c = a.sub(&b)?;
    assert_eq!(c.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn mul() -> TensorResult<()> {
    let device = Device::cuda(0);

    let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let c = a.mul(&b)?;
    assert_eq!(c.to_vec()?, vec![4.0, 10.0, 18.0]);
    Ok(())
}

#[test]
fn div() -> TensorResult<()> {
    let device = Device::cuda(0);

    let a = Tensor::from_device(vec![4.0, 10.0, 18.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let c = a.div(&b)?;
    let tolerance = 1e-6;
    assert!(c
        .to_vec()?
        .iter()
        .zip([1.0, 2.0, 3.0].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));
    Ok(())
}

#[test]
fn mat_mul() -> TensorResult<()> {
    let device = Device::cuda(0);

    let a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
    let c = a.mat_mul(&b)?;

    // Result should be:
    // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    assert_eq!(c.to_vec()?, vec![19.0, 22.0, 43.0, 50.0]);
    Ok(())
}
