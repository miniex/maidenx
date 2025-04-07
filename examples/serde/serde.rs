use maidenx::nn::*;
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // set_default_device(Device::CPU);
    // set_default_dtype(float32);
    // set_default_dtype(DType::F32);

    let tensor_x = Tensor::new(vec![1, 2, 3])?;
    let bytes = tensor_x.to_bytes()?;
    let tensor_y = Tensor::from_bytes(&bytes)?;

    // let x = Linear::new(3, 1, true)?;
    // x.save("assets/serde/x", "bytes")?;
    // x.save("assets/serde/x", "bin")?;
    // x.save("assets/serde/x", "json")?;
    let y = Linear::load("assets/serde/x.bin")?;
    let z = Linear::load("assets/serde/x.json")?;

    println!("y: {:?}\nz: {:?}", y.forward(&tensor_y)?, z.forward(&tensor_x)?);

    Ok(())
}
