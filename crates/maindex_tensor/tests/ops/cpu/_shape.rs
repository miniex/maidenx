use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn broadcast_like() {
    let device = Device::cpu();

    // Case 1: [3] -> [1, 3]
    let a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device).unwrap();
    let b = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]], &device).unwrap();

    let broadcasted = a.broadcast_like(&b).unwrap();
    assert_eq!(broadcasted.shape(), &[1, 3]);
    assert_eq!(broadcasted.strides(), &[3, 1]);
    assert_eq!(broadcasted.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);

    // Case 2: [1, 3] -> [2, 3]
    let c = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]], &device).unwrap();
    let d = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]; 2], &device).unwrap();

    let broadcasted_c = c.broadcast_like(&d).unwrap();
    assert_eq!(broadcasted_c.shape(), &[2, 3]);
    assert_eq!(broadcasted_c.strides(), &[3, 1]);
    assert_eq!(
        broadcasted_c.to_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn broadcast_add() -> TensorResult<()> {
    let device = Device::cpu();

    // Case 1
    let tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let tensor2 = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let result = tensor1.add(&tensor2)?;
    assert_eq!(result.to_vec()?, vec![5.0, 7.0, 9.0]);

    // Case 2
    let tensor1d = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
    let tensor2d = Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;
    let result = tensor1d.add(&tensor2d)?;
    assert_eq!(result.to_vec()?, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);

    // Case 3
    let a = Tensor::from_device(vec![1.0, 2.0], &device)?.reshape(&[2, 1])?;
    let b = Tensor::from_device(vec![3.0, 4.0, 5.0], &device)?.reshape(&[1, 3])?;
    let result = a.add(&b)?;
    assert_eq!(result.to_vec()?, vec![4.0, 5.0, 6.0, 5.0, 6.0, 7.0]);

    Ok(())
}

#[test]
fn broadcast_sub() -> TensorResult<()> {
    let device = Device::cpu();

    // Case 1
    let tensor1 = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let tensor2 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let result = tensor1.sub(&tensor2)?;
    assert_eq!(result.to_vec()?, vec![3.0, 3.0, 3.0]);

    // Case 2
    let tensor1d = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
    let tensor2d = Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;
    let result = tensor1d.sub(&tensor2d)?;
    assert_eq!(result.to_vec()?, vec![0.0, 0.0, 0.0, -3.0, -3.0, -3.0]);

    // Case 3
    let a = Tensor::from_device(vec![1.0, 2.0], &device)?.reshape(&[2, 1])?;
    let b = Tensor::from_device(vec![3.0, 4.0, 5.0], &device)?.reshape(&[1, 3])?;
    let result = a.sub(&b)?;
    assert_eq!(result.to_vec()?, vec![-2.0, -3.0, -4.0, -1.0, -2.0, -3.0]);

    Ok(())
}

#[test]
fn broadcast_mul() -> TensorResult<()> {
    let device = Device::cpu();

    // Case 1: Same dimension vectors [3] * [3]
    let tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let tensor2 = Tensor::from_device(vec![2.0, 3.0, 4.0], &device)?;
    let result = tensor1.mul(&tensor2)?;
    assert_eq!(result.to_vec()?, vec![2.0, 6.0, 12.0]);

    // Case 2: 1D to 2D with reshape [1,2] * [2,2]
    let vector = Tensor::from_device(vec![2.0, 3.0], &device)?.reshape(&[1, 2])?;
    let matrix = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let result = vector.mul(&matrix)?;
    assert_eq!(result.to_vec()?, vec![2.0, 6.0, 6.0, 12.0]);

    Ok(())
}

#[test]
fn broadcast_div() -> TensorResult<()> {
    let device = Device::cpu();

    // Case 1: Same dimension vectors [3] / [3]
    let tensor1 = Tensor::from_device(vec![2.0, 4.0, 6.0], &device)?;
    let tensor2 = Tensor::from_device(vec![2.0, 2.0, 2.0], &device)?;
    let result = tensor1.div(&tensor2)?;
    assert_eq!(result.to_vec()?, vec![1.0, 2.0, 3.0]);

    // Case 2: Broadcasting with reshape [2,2] / [1,2]
    let matrix = Tensor::from_device(vec![vec![2.0, 4.0], vec![6.0, 8.0]], &device)?;
    let vector = Tensor::from_device(vec![2.0, 2.0], &device)?.reshape(&[1, 2])?;
    let result = matrix.div(&vector)?;
    assert_eq!(result.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[test]
fn broadcast_failure() -> TensorResult<()> {
    let device = Device::cpu();

    // Case 1: Incompatible broadcasting shapes
    let tensor1 = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let tensor2 = Tensor::from_device(vec![vec![1.0, 2.0, 3.0]], &device)?;
    assert!(tensor1.add(&tensor2).is_err());

    // Case 2: Cannot broadcast larger dimension to smaller
    let tensor3 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let tensor4 = Tensor::from_device(vec![1.0, 2.0], &device)?;
    assert!(tensor3.add(&tensor4).is_err());

    Ok(())
}
