use maidenx_core::{
    device::{set_default_device, Device},
    error::Result,
};
use maidenx_tensor::{adapter::TensorAdapter, Tensor};

// Helper functions
fn setup_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(not(any(feature = "cuda")))]
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
