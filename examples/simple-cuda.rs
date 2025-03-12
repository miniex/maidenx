use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_default_device(Device::CUDA(0));
    // set_default_dtype(float32);
    // set_default_dtype(DType::F32);

    let input_data: Vec<Vec<f32>> = (0..10000)
        .map(|i| vec![(i % 100) as f32 / 100.0, ((i % 100) + 1) as f32 / 100.0, ((i % 100) + 2) as f32 / 100.0])
        .collect();
    let target_data: Vec<Vec<f32>> = (0..10000).map(|i| vec![((i % 100) * 10) as f32 / 1000.0]).collect();

    let mut input = Tensor::new(input_data)?;
    let target = Tensor::new(target_data)?;
    input.with_grad()?;

    let mut linear = Linear::new(3, 1, true)?;
    let mse_loss = MSE::new();
    let mut optimizer = SGD::new(0.01);
    let epochs = 1000;

    let mut hundred_epochs_start = Instant::now();

    for epoch in 0..epochs {
        let pred = linear.forward(&input)?;
        let loss = mse_loss.forward((&pred, &target))?;
        loss.backward()?;

        optimizer.step(&mut linear.parameters())?;
        optimizer.zero_grad(&mut linear.parameters())?;

        if (epoch + 1) % 100 == 0 {
            let hundred_elapsed = hundred_epochs_start.elapsed();
            let params = linear.parameters();
            println!(
                "Epoch {}: Loss = {}, 100 Epochs Time = {:?}, Weight = {}, Bias = {}",
                epoch + 1,
                loss,
                hundred_elapsed,
                params[0],
                params.get(1).unwrap()
            );
            hundred_epochs_start = Instant::now();
        }
    }

    Ok(())
}
