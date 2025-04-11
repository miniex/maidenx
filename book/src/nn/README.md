# Neural Networks

The MaidenX neural networks module provides building blocks for creating deep learning models. It offers high-level abstractions for layers, optimizers, and loss functions, all of which integrate seamlessly with the tensor library's automatic differentiation system.

## Overview

MaidenX neural networks module provides:

- A consistent `Layer` trait for all neural network components
- Common layer implementations (Linear, Convolutional, etc.)
- Optimization algorithms (SGD, Adam)
- Loss functions (MSE, MAE, Cross Entropy, Huber)
- Support for training and evaluation modes

## Training Example

Here's a simple example of training a model with MaidenX:

```rust
// Create model layers
let mut linear1 = Linear::new(784, 128, true)?;
let mut linear2 = Linear::new(128, 10, true)?;

// Create optimizer
let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

// Training loop
for epoch in 0..num_epochs {
    // For each batch...
    let input = get_batch_input()?;  // Shape: [batch_size, 784]
    let target = get_batch_target()?;  // Shape: [batch_size, 10]
    
    // Forward pass
    let hidden = linear1.forward(&input)?;
    let output = linear2.forward(&hidden)?;
    
    // Compute loss
    let loss_fn = MSELoss::new();
    let loss = loss_fn.forward((&output, &target))?;
    
    // Backward pass
    loss.backward()?;
    
    // Collect parameters and update
    let mut params = Vec::new();
    params.append(&mut linear1.parameters());
    params.append(&mut linear2.parameters());
    
    // Update parameters
    optimizer.step(&mut params)?;
    optimizer.zero_grad(&mut params)?;
}
```

## Feature Support

MaidenX neural networks can run on different compute devices:
- CPU
- CUDA (GPU) with feature flag `cuda`
- MPS (Apple Silicon) with feature flag `mps`

## Serialization and Deserialization

MaidenX supports model serialization and deserialization through the `serde` feature flag. When enabled, all built-in layers can be saved to and loaded from files.

### Enabling Serialization

To enable serialization support, add the `serde` feature in your Cargo.toml:

```toml
[dependencies]
maidenx = { version = "0.1.0", features = ["serde"] }
```

### Saving and Loading Models

Built-in layers can be saved and loaded like this:

```rust
// Save a linear layer
let linear = Linear::new(784, 256, true)?;
linear.save("path/to/model.bin", "bin")?;  // Binary format
linear.save("path/to/model.json", "json")?;  // JSON format

// Load a linear layer
let loaded_linear = Linear::load("path/to/model.bin")?;
```

### Custom Model Serialization

For custom models or layers, you simply need to derive `Serialize` and `Deserialize` from the `serde` crate. MaidenX will automatically provide save/load functionality for your custom models:

```rust
use serde::{Serialize, Deserialize};

#[derive(Layer, Clone, Serialize, Deserialize)]
pub struct MyCustomModel {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
    state: LayerState,
}

impl MyCustomModel {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self> {
        Ok(Self {
            linear1: Linear::new(input_size, hidden_size, true)?,
            linear2: Linear::new(hidden_size, output_size, true)?,
            dropout: Dropout::new(0.5)?,
            state: LayerState::new(),
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(input)?;
        let hidden_dropped = self.dropout.forward(&hidden)?;
        self.linear2.forward(&hidden_dropped)
    }
    
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.append(&mut self.linear1.parameters());
        params.append(&mut self.linear2.parameters());
        params
    }
}
```

Once you've derived the required traits, you can save and load your custom models using the standard methods - no need to implement your own save/load functions:

```rust
// Save model
let model = MyCustomModel::new(784, 256, 10)?;
model.save("path/to/custom_model.bin", "bin")?; // Binary format
model.save("path/to/custom_model.json", "json")?; // JSON format

// Load model
let loaded_model = MyCustomModel::load("path/to/custom_model.bin")?;
```

### Implementation Notes

- All built-in MaidenX layers derive `Serialize` and `Deserialize` when the `serde` feature is enabled
- Only model structure and parameters are serialized, not the computational graph
- Custom models and layers must derive `Serialize` and `Deserialize` manually
- Binary serialization is more compact but less human-readable than JSON
- Saved models can be loaded across different platforms
