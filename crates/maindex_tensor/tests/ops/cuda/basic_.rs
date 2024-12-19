use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn add_in_place() -> TensorResult<()> {
    let device = Device::cuda(0);

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.add_(&b)?;

    // Check if `a` is updated in place
    assert_eq!(a.to_vec()?, vec![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn add_in_place_with_broadcast() -> TensorResult<()> {
    let device = Device::cuda(0);

    let mut a = Tensor::from_device(vec![1.0], &device)?; // Shape: [1]
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?; // Shape: [3]

    a.add_(&b)?;

    // After broadcasting, `a` should match the shape and values
    assert_eq!(a.to_vec()?, vec![5.0, 6.0, 7.0]);
    Ok(())
}

#[test]
fn sub_in_place() -> TensorResult<()> {
    let device = Device::cuda(0);

    let mut a = Tensor::from_device(vec![5.0, 7.0, 9.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.sub_(&b)?;

    // Check if `a` is updated in place
    assert_eq!(a.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn mul_in_place() -> TensorResult<()> {
    let device = Device::cuda(0);

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.mul_(&b)?;

    // Check if `a` is updated in place
    assert_eq!(a.to_vec()?, vec![4.0, 10.0, 18.0]);
    Ok(())
}

#[test]
fn div_in_place() -> TensorResult<()> {
    let device = Device::cuda(0);

    let mut a = Tensor::from_device(vec![4.0, 10.0, 18.0], &device)?;
    let b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.div_(&b)?;

    // Check if `a` is updated in place
    let tolerance = 1e-6;
    assert!(a
        .to_vec()?
        .iter()
        .zip([1.0, 2.0, 3.0].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));
    Ok(())
}

#[test]
fn mat_mul_in_place() -> TensorResult<()> {
    let device = Device::cuda(0);

    let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

    a.mat_mul_(&b)?;

    // Check if `a` is updated in place
    // Result should be:
    // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    assert_eq!(a.to_vec()?, vec![19.0, 22.0, 43.0, 50.0]);
    Ok(())
}
