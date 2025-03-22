use crate::Tensor;
use maidenx_core::error::Result;

pub fn any_true(src: &Tensor) -> Result<bool> {
    let vector = src.to_flatten_vec::<bool>()?;

    Ok(vector.iter().any(|&x| x))
}
