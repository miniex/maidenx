use maidenx_core::{
    device::{set_default_device, Device},
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Helper functions
fn setup_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(feature = "mps")]
    set_default_device(Device::MPS);
    #[cfg(not(any(feature = "cuda", feature = "mps")))]
    set_default_device(Device::CPU);
}

fn setup_tensor<T: Clone + 'static>(data: Vec<T>) -> Result<Tensor>
where
    Vec<T>: TensorAdapter,
{
    setup_device();

    let tensor = Tensor::new(data)?;
    Ok(tensor)
}

#[test]
fn any() -> Result<()> {
    let x = setup_tensor(vec![false, false, true, false, false])?;
    assert!(x.any()?);

    let y = setup_tensor(vec![false, false, false, false, false])?;
    assert!(!y.any()?);

    let z = setup_tensor(vec![0.0f32, 0.0, 0.0, 1.0, 0.0])?;
    assert!(z.any()?);

    let w = setup_tensor(vec![0.0f32, 0.0, 0.0, 0.0, 0.0])?;
    assert!(!w.any()?);

    Ok(())
}

#[test]
fn get() -> Result<()> {
    let x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0])?;
    let scalar = x.get(&[4])?;

    assert_eq!(scalar.as_f32(), 7.0f32);

    Ok(())
}

#[test]
fn set() -> Result<()> {
    let mut x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0])?;
    x.set(&[4], 2.0)?;

    assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0f32, 4.0, 5.0, 9.0, 2.0, 3.0]);

    Ok(())
}

#[test]
fn selcet() -> Result<()> {
    let x = setup_tensor(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0])?;
    let select = x.select(0, 4)?;

    assert_eq!(select.to_flatten_vec::<f32>()?, vec![7.0]);

    Ok(())
}
