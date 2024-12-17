use maidenx::nn::*;
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_device = Device::cuda(0);
    set_current_device(cuda_device)?;

    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let linear = Linear::new(2, 2, None, true)?;
    let linear_output = linear.forward(&input)?;

    let relu = ReLU::new();
    let relu_output = relu.forward(&linear_output)?;

    let sigmoid = Sigmoid::new();
    let sigmoid_output = sigmoid.forward(&relu_output)?;

    let tanh = Tanh::new();
    let tanh_output = tanh.forward(&sigmoid_output)?;

    println!("Input:\\n{}", input);
    println!("Linear output:\\n{}", linear_output);
    println!("ReLU output:\\n{}", relu_output);
    println!("Sigmoid output:\\n{}", sigmoid_output);
    println!("Tanh output:\\n{}", tanh_output);

    Ok(())
}
