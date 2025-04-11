# Layer

The `Layer` trait is the foundation of neural network components in MaidenX. It defines the interface that all neural network layers must implement.

## Layer Trait Definition

```rust
pub trait Layer<I = &'static Tensor> {
    fn forward(&self, input: I) -> Result<Tensor>;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
    
    fn is_training(&self) -> bool;
    fn train(&mut self);
    fn eval(&mut self);
}
```

The `Layer` trait makes it easy to create custom layers and combine them into complex architectures. The generic parameter `I` allows layers to handle different input types, with the default being a reference to a `Tensor`.

## Core Methods

### forward

```rust
fn forward(&self, input: I) -> Result<Tensor>;
```

The `forward` method performs the layer's computation on the input and returns the output tensor. It's the primary function that defines the layer's behavior.

### parameters

```rust
fn parameters(&mut self) -> Vec<&mut Tensor>;
```

Returns all trainable parameters of the layer as mutable references, which can then be updated by optimizers during training.

## Training State Management

### is_training

```rust
fn is_training(&self) -> bool;
```

Returns whether the layer is in training mode (true) or evaluation mode (false).

### train

```rust
fn train(&mut self);
```

Sets the layer to training mode. This affects behaviors like dropout and batch normalization.

### eval

```rust
fn eval(&mut self);
```

Sets the layer to evaluation mode. This typically disables regularization techniques like dropout.

## LayerState

Most layer implementations use the `LayerState` structure to track their training state:

```rust
pub struct LayerState {
    training: bool,
}
```

`LayerState` provides a simple way to implement the training state methods:

```rust
impl LayerState {
    pub fn new() -> Self {
        Self { training: true }
    }

    pub fn is_training(&self) -> bool {
        self.training
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}
```

## Custom Layer Implementation

To implement a custom layer, you need to implement the `Layer` trait:

```rust
#[derive(Layer, Clone)]
struct MyCustomLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    state: LayerState,
}

impl Layer for MyCustomLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Custom forward computation
        let output = input.matmul(&self.weight)?;
        if let Some(ref bias) = self.bias {
            Ok(output.add(bias)?)
        } else {
            Ok(output)
        }
    }
    
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn is_training(&self) -> bool {
        self.state.is_training()
    }
    
    fn train(&mut self) {
        self.state.train();
    }
    
    fn eval(&mut self) {
        self.state.eval();
    }
}
```

## Using the Layer Macro

MaidenX provides a derive macro to simplify layer implementation:

```rust
#[derive(Layer, Clone)]
struct MySimpleLayer {
    weight: Tensor,
    state: LayerState,
}

// The Layer trait methods for training state are automatically implemented
impl MySimpleLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Your implementation here
    }
    
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}
