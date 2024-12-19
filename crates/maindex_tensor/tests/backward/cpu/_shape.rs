use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn broadcast_add_backward() -> TensorResult<()> {
    let device = Device::cpu();

    // Test case: [1,3] + [2,3]
    let mut tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
    let mut tensor2 = Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;

    tensor1.with_grad();
    tensor2.with_grad();

    let result = tensor1.add(&tensor2)?;
    result.backward()?;

    // Check gradients
    let tensor1_grad = tensor1.grad()?.unwrap().to_vec()?;
    assert_eq!(tensor1_grad, vec![2.0, 2.0, 2.0]); // Sum across broadcast dimension

    let tensor2_grad = tensor2.grad()?.unwrap().to_vec()?;
    assert_eq!(tensor2_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn broadcast_mul_backward() -> TensorResult<()> {
    let device = Device::cpu();

    // Test case: [1,3] + [2,3]
    let mut tensor1 = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?.reshape(&[1, 3])?;
    let mut tensor2 = Tensor::from_device(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)?;

    tensor1.with_grad();
    tensor2.with_grad();

    let result = tensor1.mul(&tensor2)?;
    result.backward()?;

    // Check gradients
    let tensor1_grad = tensor1.grad()?.unwrap().to_vec()?;
    let tensor2_grad = tensor2.grad()?.unwrap().to_vec()?;

    // Verify gradient shapes and values
    assert_eq!(tensor1_grad.len(), 3);
    assert_eq!(tensor2_grad.len(), 6);

    Ok(())
}

#[test]
fn broadcast_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    // Setup tensors with compatible broadcasting shapes
    let mut tensor1 = Tensor::from_device(vec![1.0, 2.0], &device)?.reshape(&[1, 2])?;
    let mut tensor2 = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let mut tensor3 = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

    tensor1.with_grad();
    tensor2.with_grad();
    tensor3.with_grad();

    // (tensor1 * tensor2) + tensor3 where tensor1 is [1,2], tensor2 and tensor3 are [2,2]
    let result = tensor1.mul(&tensor2)?.add(&tensor3)?;
    result.backward()?;

    // Verify gradient shapes
    assert_eq!(tensor1.grad()?.unwrap().to_vec()?.len(), 2);
    assert_eq!(tensor2.grad()?.unwrap().to_vec()?.len(), 4);
    assert_eq!(tensor3.grad()?.unwrap().to_vec()?.len(), 4);

    Ok(())
}
