use maidenx_device::Device;
use maidenx_tensor::Tensor;

#[test]
fn from_device() {
    let device = Device::cpu();

    // 1D tensor
    let tensor = Tensor::from_device(vec![1.0, 2.0, 3.0], &device).unwrap();
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);

    // 2D tensor
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let tensor = Tensor::from_device(data, &device).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    // Scalar
    let tensor = Tensor::from_device(5.0f32, &device).unwrap();
    assert_eq!(tensor.shape(), &[1]);
    assert_eq!(tensor.to_vec().unwrap(), vec![5.0]);
}
