mod utils;

use maidenx_core::{device::auto_set_device, error::Result};
use maidenx_tensor_v2::Tensor;

#[test]
fn adapter() -> Result<()> {
    auto_set_device();

    // Vec of Vec test case
    let x = Tensor::new(vec![vec![1, 2], vec![3, 4]]);
    assert_eq!(x.to_flatten_vec::<i32>(), [1, 2, 3, 4]);
    assert_eq!(x.to_vec2d::<i32>(), [[1, 2], [3, 4]]);

    // Fixed-size array reference test case
    let y = Tensor::new(&[[1, 2], [3, 4]]);
    assert_eq!(y.to_flatten_vec::<i32>(), [1, 2, 3, 4]);
    assert_eq!(y.to_vec2d::<i32>(), [[1, 2], [3, 4]]);

    // Slice of slices test case
    let row1: &[i32] = &[1, 2];
    let row2: &[i32] = &[3, 4];
    let matrix: &[&[i32]] = &[row1, row2];
    let z = Tensor::new(matrix);
    assert_eq!(z.to_flatten_vec::<i32>(), [1, 2, 3, 4]);
    assert_eq!(z.to_vec2d::<i32>(), [[1, 2], [3, 4]]);

    Ok(())
}
