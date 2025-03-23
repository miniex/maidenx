mod utils;

use maidenx_core::error::Result;
use maidenx_tensor::Tensor;
use utils::setup_device;

#[test]
fn new() -> Result<()> {
    setup_device();

    let x = Tensor::new(vec![1, 2, 3])?;

    assert_eq!(x.to_flatten_vec::<i32>()?, [1, 2, 3]);

    Ok(())
}
