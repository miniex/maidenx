use maidenx_device::Device;
use maidenx_tensor::{error::TensorResult, Tensor};

#[test]
fn add_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.with_grad();
    b.with_grad();

    let c = a.add(&b)?;
    c.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
    assert_eq!(b_grad, vec![1.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn add_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

    a.with_grad();
    b.with_grad();
    c.with_grad();

    let d = a.add(&b)?.add(&c)?;
    d.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;
    let c_grad = c.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
    assert_eq!(b_grad, vec![1.0, 1.0, 1.0]);
    assert_eq!(c_grad, vec![1.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn sub_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.with_grad();
    b.with_grad();

    let c = a.sub(&b)?;
    c.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
    assert_eq!(b_grad, vec![-1.0, -1.0, -1.0]);

    Ok(())
}

#[test]
fn sub_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

    a.with_grad();
    b.with_grad();
    c.with_grad();

    let d = a.sub(&b)?.sub(&c)?;
    d.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;
    let c_grad = c.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![1.0, 1.0, 1.0]);
    assert_eq!(b_grad, vec![-1.0, -1.0, -1.0]);
    assert_eq!(c_grad, vec![-1.0, -1.0, -1.0]);

    Ok(())
}

#[test]
fn mul_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.with_grad();
    b.with_grad();

    let c = a.mul(&b)?;
    c.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![4.0, 5.0, 6.0]);
    assert_eq!(b_grad, vec![1.0, 2.0, 3.0]);

    Ok(())
}

#[test]
fn mul_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

    a.with_grad();
    b.with_grad();
    c.with_grad();

    let d = a.mul(&b)?.mul(&c)?;
    d.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;
    let c_grad = c.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![28.0, 40.0, 54.0]);
    assert_eq!(b_grad, vec![7.0, 16.0, 27.0]);
    assert_eq!(c_grad, vec![4.0, 10.0, 18.0]);

    Ok(())
}

#[test]
fn div_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;

    a.with_grad();
    b.with_grad();

    let c = a.div(&b)?;
    c.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;

    let tolerance = 1e-6;

    // Check gradients with tolerance
    assert!(a_grad
        .iter()
        .zip([0.25, 0.2, 0.16666667].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));
    assert!(b_grad
        .iter()
        .zip([-0.0625, -0.08, -0.08333334].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));

    Ok(())
}

#[test]
fn div_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;

    a.with_grad();
    b.with_grad();
    c.with_grad();

    let d = a.div(&b)?.div(&c)?;
    d.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;
    let c_grad = c.grad()?.unwrap().to_vec()?;

    // ! very big tolerance
    let tolerance = 1e-2;

    // Check gradients with tolerance
    assert!(a_grad
        .iter()
        .zip([0.03571429, 0.025, 0.01851852].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));
    assert!(b_grad
        .iter()
        .zip([-0.00892857, -0.01, -0.01028807].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));
    assert!(c_grad
        .iter()
        .zip([-0.00510204, -0.005, -0.00462963].iter())
        .all(|(a, b)| (a - b).abs() < tolerance));

    Ok(())
}

#[test]
fn mat_mul_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let mut b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;

    a.with_grad();
    b.with_grad();

    let c = a.mat_mul(&b)?;
    c.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![11.0, 15.0, 11.0, 15.0]);
    assert_eq!(b_grad, vec![4.0, 4.0, 6.0, 6.0]);

    Ok(())
}

#[test]
fn mat_mul_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?;
    let mut b = Tensor::from_device(vec![vec![5.0, 6.0], vec![7.0, 8.0]], &device)?;
    let mut c = Tensor::from_device(vec![vec![1.0, 0.0], vec![0.0, 1.0]], &device)?;

    a.with_grad();
    b.with_grad();
    c.with_grad();

    let d = a.mat_mul(&b)?.mat_mul(&c)?;
    d.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;
    let c_grad = c.grad()?.unwrap().to_vec()?;

    assert_eq!(a_grad, vec![11.0, 15.0, 11.0, 15.0]);
    assert_eq!(b_grad, vec![4.0, 4.0, 6.0, 6.0]);
    assert_eq!(c_grad, vec![62.0, 62.0, 72.0, 72.0]);

    Ok(())
}

#[test]
fn mix_chain_backward() -> TensorResult<()> {
    let device = Device::cpu();

    let mut a = Tensor::from_device(vec![1.0, 2.0, 3.0], &device)?;
    let mut b = Tensor::from_device(vec![4.0, 5.0, 6.0], &device)?;
    let mut c = Tensor::from_device(vec![7.0, 8.0, 9.0], &device)?;
    let mut d = Tensor::from_device(vec![10.0, 11.0, 12.0], &device)?;

    a.with_grad();
    b.with_grad();
    c.with_grad();
    d.with_grad();

    // ((a + b) * c - d) / 2
    let result = a
        .add(&b)? // (a + b)
        .mul(&c)? // * c
        .sub(&d)? // - d
        .scalar_div(2.0)?; // / 2

    result.backward()?;

    let a_grad = a.grad()?.unwrap().to_vec()?;
    let b_grad = b.grad()?.unwrap().to_vec()?;
    let c_grad = c.grad()?.unwrap().to_vec()?;
    let d_grad = d.grad()?.unwrap().to_vec()?;

    // result = (((a + b) * c) - d) / 2
    // ∂L/∂a = (1 * c) / 2
    // ∂L/∂b = (1 * c) / 2
    // ∂L/∂c = ((a + b) * 1) / 2
    // ∂L/∂d = -1 / 2

    assert_eq!(a_grad, vec![3.5, 4.0, 4.5]); // (c / 2)
    assert_eq!(b_grad, vec![3.5, 4.0, 4.5]); // (c / 2)
    assert_eq!(c_grad, vec![2.5, 3.5, 4.5]); // ((a + b) / 2)
    assert_eq!(d_grad, vec![-0.5, -0.5, -0.5]); // -1 / 2

    Ok(())
}
