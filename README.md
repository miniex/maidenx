# MaidenX

A Rust-based machine learning framework developed as part of the Maiden Engine project. MaidenX is designed with an educational focus, structured to mirror PyTorch's architecture to facilitate learning and understanding of ML framework implementations.
This library prioritizes code readability, ensuring that anyone can easily understand and work with the codebase.

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/miniex/maidenx#license)
[![Crates.io](https://img.shields.io/crates/v/maidenx.svg)](https://crates.io/crates/maidenx)

## Project Status 🚧

This is a personal learning and development project. As such:
- The framework is under active development
- Features may be experimental or incomplete
- Functionality is not guaranteed for production use

The project serves primarily as a testbed for AI engine development and learning purposes.

## Goals

MaidenX is being developed with a vision to create a lightweight, fast, and human-like artificial intelligence framework.
The library focuses on simplicity, performance, and user convenience, ensuring that developers can work effortlessly while enjoying robust machine learning capabilities.
As the project evolves, MaidenX aims to serve as a foundation for innovative AI solutions and advanced learning resources.

## Examples

```rust
use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
```
