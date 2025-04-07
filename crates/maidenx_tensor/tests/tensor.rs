mod utils;

use maidenx_core::error::Result;
use utils::setup_tensor_without_dtype;

#[test]
fn any() -> Result<()> {
    let x = setup_tensor_without_dtype(vec![false, false, true, false, false])?;
    assert!(x.any()?);

    let y = setup_tensor_without_dtype(vec![false, false, false, false, false])?;
    assert!(!y.any()?);

    let z = setup_tensor_without_dtype(vec![0.0f32, 0.0, 0.0, 1.0, 0.0])?;
    assert!(z.any()?);

    let w = setup_tensor_without_dtype(vec![0.0f32, 0.0, 0.0, 0.0, 0.0])?;
    assert!(!w.any()?);

    Ok(())
}

#[test]
fn get() -> Result<()> {
    let x = setup_tensor_without_dtype(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0])?;
    let scalar = x.get(&[4])?;

    assert_eq!(scalar.as_f32(), 7.0f32);

    Ok(())
}

#[test]
fn set() -> Result<()> {
    let mut x = setup_tensor_without_dtype(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0])?;
    x.set(&[4], 2.0)?;

    assert_eq!(x.to_flatten_vec::<f32>()?, vec![3.0f32, 4.0, 5.0, 9.0, 2.0, 3.0]);

    Ok(())
}

#[test]
fn selcet() -> Result<()> {
    let x = setup_tensor_without_dtype(vec![3.0f32, 4.0, 5.0, 9.0, 7.0, 3.0])?;
    let select = x.select(0, 4)?;

    assert_eq!(select.to_flatten_vec::<f32>()?, vec![7.0]);

    Ok(())
}
