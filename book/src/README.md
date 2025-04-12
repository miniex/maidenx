# Introduction

## What is MaidenX?

MaidenX is a Rust-based machine learning framework developed as part of the Maiden Engine project. It is designed with an educational focus, structured to mirror PyTorch's architecture to facilitate learning and understanding of ML framework implementations. The library prioritizes code readability, ensuring that anyone can easily understand and work with the codebase.

## Key Features

- **Pure Rust Implementation**: Built entirely in Rust, providing memory safety, concurrency, and performance benefits.
- **PyTorch-like API**: Familiar and intuitive API design for those coming from PyTorch.
- **Multiple Backends**: Support for CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon) computation.
- **Automatic Differentiation**: Built-in autograd system for gradient-based optimization.
- **Computational Graph**: Optional computational graph mode for deferred execution.
- **Serialization**: Integration with Rust's serde framework for model saving and loading.
- **Comprehensive Operations**: Rich set of tensor operations with autograd support.
- **Neural Network Layers**: Ready-to-use implementations of common neural network components.

## Architecture

MaidenX is organized into several core components:

### 1. Tensor System (`maidenx_tensor`)

The tensor module provides the foundation for all numerical operations. Key features include:

- Support for multiple data types (float32, float64, int32, etc.)
- Comprehensive tensor operations (arithmetic, transformation, reduction)
- Automatic broadcasting for compatible shapes
- In-place and out-of-place operations
- Efficient memory management and buffer handling

### 2. Neural Network Components (`maidenx_nn`)

The neural network module offers building blocks for constructing machine learning models:

- Common layers: Linear, Conv2d, LayerNorm, Dropout, Embedding
- Activation functions: ReLU, Sigmoid, Tanh, GELU, Softmax, etc.
- Loss functions: MSE, MAE, Huber, CrossEntropy
- Optimizers: SGD, Adam

### 3. Backend System (`maidenx_core`)

The core backend system provides device-specific implementations:

- CPU backend for universal compatibility
- CUDA backend for NVIDIA GPU acceleration
- MPS backend for Apple Silicon GPU acceleration
- Abstract device interface for consistent API across backends

## Getting Started

MaidenX organizes its functionality into separate features, allowing users to select only what they need:

### Default Features

These are included by default and recommended for most use cases:

- **nn**: Core neural network functionality
- **serde**: Serialization/deserialization support
- **graph**: Computational graph mode for deferred operations

### Optional Features

- **cuda**: GPU acceleration support using NVIDIA CUDA
- **mps**: Apple Metal Performance Shaders support for Apple Silicon

## Example Usage

Here's a simple example of training a linear model with MaidenX:

```rust
use maidenx::nn::*;
use maidenx::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create input and target data
    let input_data: Vec<Vec<f32>> = (0..10000)
        .map(|i| vec![(i % 100) as f32 / 100.0, ((i % 100) + 1) as f32 / 100.0, ((i % 100) + 2) as f32 / 100.0])
        .collect();
    let target_data: Vec<Vec<f32>> = (0..10000).map(|i| vec![((i % 100) * 10) as f32 / 1000.0]).collect();

    let mut input = Tensor::new(input_data)?;
    let target = Tensor::new(target_data)?;
    input.with_grad()?;

    // Create model, loss function, and optimizer
    let mut linear = Linear::new(3, 1, true)?;
    let mse_loss = MSE::new();
    let mut optimizer = SGD::new(0.01);
    let epochs = 1000;

    // Training loop
    for epoch in 0..epochs {
        let pred = linear.forward(&input)?;
        let loss = mse_loss.forward((&pred, &target))?;
        loss.backward()?;

        optimizer.step(&mut linear.parameters())?;
        optimizer.zero_grad(&mut linear.parameters())?;

        if (epoch + 1) % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch + 1, loss);
        }
    }

    Ok(())
}
```

## Supported Operations and Layers

MaidenX includes a comprehensive set of tensor operations and neural network layers, which we'll explore in more detail in the following chapters.
