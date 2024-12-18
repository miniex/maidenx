use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training on CPU:");
    train_linear_regression(Device::cpu())?;

    #[cfg(feature = "cuda")]
    {
        println!("\nTraining on CUDA:");
        train_linear_regression(Device::cuda(0))?;
    }

    Ok(())
}

fn train_linear_regression(device: Device) -> Result<(), Box<dyn std::error::Error>> {
    let mut input =
        Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    let target = Tensor::from_vec_with_device(vec![10.0, 20.0], &[2, 1], &device)?;

    input.with_grad();

    let linear = Linear::new_with_device(3, 1, None, true, device)?;

    let mse_loss = MSELoss::new();
    let mut optimizer = SGD::new(0.01);

    let epochs = 25;
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
