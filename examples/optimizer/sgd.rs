use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("Training on CPU:");
        train_linear_regression(Device::cpu())?;
    }
    #[cfg(feature = "cuda")]
    {
        println!("\nTraining on CUDA:");
        train_linear_regression(Device::cuda(0))?;
    }

    Ok(())
}

fn train_linear_regression(device: Device) -> Result<(), Box<dyn std::error::Error>> {
    let input_data: Vec<Vec<f32>> = (0..10000)
        .map(|i| {
            vec![
                (i % 100) as f32 / 100.0,
                ((i % 100) + 1) as f32 / 100.0,
                ((i % 100) + 2) as f32 / 100.0,
            ]
        })
        .collect();
    let target_data: Vec<Vec<f32>> = (0..10000)
        .map(|i| vec![((i % 100) * 10) as f32 / 1000.0])
        .collect();

    let mut input = Tensor::from_device(input_data, &device)?;
    let target = Tensor::from_device(target_data, &device)?;

    input.with_grad();

    let linear = Linear::new_with_device(3, 1, None, true, device)?;

    let mse_loss = MSELoss::new();
    let mut optimizer = SGD::new(0.01);

    let epochs = 100;
    for epoch in 0..epochs {
        let start_time = Instant::now();

        let pred = linear.forward(&input)?;
        let loss = mse_loss.forward((&pred, &target))?;

        loss.backward()?;

        optimizer.step(&mut linear.parameters())?;
        optimizer.zero_grad(&mut linear.parameters())?;

        let elapsed = start_time.elapsed();
        println!(
            "Epoch {}: Loss = {:.4}, Time = {:?}, Weight = {:?}, Bias = {:?}",
            epoch,
            loss.to_vec()?[0],
            elapsed,
            linear.parameters()[0].to_vec()?,
            linear
                .parameters()
                .get(1)
                .map(|b| b.to_vec())
                .unwrap_or(Ok(vec![]))?
        );
    }

    Ok(())
}
