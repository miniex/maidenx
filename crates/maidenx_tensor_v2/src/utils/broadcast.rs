use crate::Tensor;
use maidenx_core::error::{Error, Result};

pub fn broadcast_tensors(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    let output_ndim = a_ndim.max(b_ndim);
    let mut output_shape = vec![0; output_ndim];

    for i in 0..output_ndim {
        let a_dim = if i < a_ndim { a_shape[a_ndim - 1 - i] } else { 1 };
        let b_dim = if i < b_ndim { b_shape[b_ndim - 1 - i] } else { 1 };

        if a_dim == 1 || b_dim == 1 || a_dim == b_dim {
            output_shape[output_ndim - 1 - i] = a_dim.max(b_dim);
        } else {
            return Err(Error::InvalidShape {
                message: format!("Cannot broadcast shapes {:?} and {:?}", a_shape, b_shape),
            });
        }
    }

    let a_broadcasted = a.try_broadcast(&output_shape)?;
    let b_broadcasted = b.try_broadcast(&output_shape)?;

    Ok((a_broadcasted, b_broadcasted))
}

