# Optimizer

The `Optimizer` trait defines the interface for all optimization algorithms in MaidenX. Optimizers update the parameters of neural network layers based on gradient information to minimize the loss function.

## Optimizer Trait Definition

```rust
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;
    fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>);
}
```

## Core Methods

### step

```rust
fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;
```

The `step` method updates the parameters based on their current gradients. This is the core method that performs the optimization algorithm's update rule.

### zero_grad

```rust
fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;
```

Resets the gradients of all parameters to zero, typically called before computing gradients for the next batch.

### set_learning_rate

```rust
fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>);
```

Allows dynamic adjustment of the learning rate during training, which can be useful for learning rate scheduling.

## Available Optimizers

### SGD (Stochastic Gradient Descent)

The SGD optimizer implements basic gradient descent with a configurable learning rate:

```rust
pub struct SGD {
    learning_rate: Scalar,
}

impl SGD {
    pub fn new(learning_rate: impl Into<Scalar>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
        }
    }
}
```

**Usage Example:**

```rust
let mut sgd = SGD::new(0.01);
// Training loop
for _ in 0..num_epochs {
    // Forward and backward pass
    // ...
    
    // Update parameters
    sgd.step(&mut parameters)?;
    sgd.zero_grad(&mut parameters)?;
}
```

### Adam (Adaptive Moment Estimation)

The Adam optimizer implements adaptive learning rates for each parameter with momentum and RMSProp-like behavior:

```rust
pub struct Adam {
    learning_rate: Scalar,
    beta1: Scalar,       // Exponential decay rate for first moment
    beta2: Scalar,       // Exponential decay rate for second moment
    epsilon: Scalar,     // Small constant for numerical stability
    t: usize,            // Timestep
    m: Vec<Tensor>,      // First moment vectors
    v: Vec<Tensor>,      // Second moment vectors
}

impl Adam {
    pub fn new(
        learning_rate: impl Into<Scalar>, 
        beta1: impl Into<Scalar>, 
        beta2: impl Into<Scalar>, 
        epsilon: impl Into<Scalar>
    ) -> Self {
        // Initialization
    }
}
```

**Usage Example:**

```rust
let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
// Training loop
for _ in 0..num_epochs {
    // Forward and backward pass
    // ...
    
    // Update parameters
    adam.step(&mut parameters)?;
    adam.zero_grad(&mut parameters)?;
}
```

## Implementing Custom Optimizers

To create a custom optimizer, implement the `Optimizer` trait:

```rust
#[derive(Optimizer)]
struct MyCustomOptimizer {
    learning_rate: Scalar,
    momentum: Scalar,
    velocity: Vec<Tensor>,
}

impl MyCustomOptimizer {
    pub fn new(learning_rate: impl Into<Scalar>, momentum: impl Into<Scalar>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            momentum: momentum.into(),
            velocity: Vec::new(),
        }
    }
    
    pub fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        // Initialize velocity vectors if needed
        if self.velocity.is_empty() {
            self.velocity = parameters
                .iter()
                .map(|param| Tensor::zeros_like(param))
                .collect::<Result<Vec<_>>>()?;
        }
        
        // Update rule with momentum
        for (param_idx, param) in parameters.iter_mut().enumerate() {
            if let Some(grad) = param.grad()? {
                // Update velocity
                self.velocity[param_idx] = self.velocity[param_idx]
                    .mul_scalar(self.momentum)?
                    .add(&grad)?;
                
                // Update parameter
                param.sub_(&self.velocity[param_idx].mul_scalar(self.learning_rate)?)?;
            }
        }
        Ok(())
    }
    
    pub fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        for param in parameters.iter_mut() {
            param.zero_grad()?;
        }
        Ok(())
    }
    
    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }
}
```

## Learning Rate Scheduling

You can implement learning rate scheduling by adjusting the learning rate during training:

```rust
let mut optimizer = SGD::new(0.1);

for epoch in 0..num_epochs {
    // Decay learning rate every 10 epochs
    if epoch > 0 && epoch % 10 == 0 {
        let current_lr = optimizer.learning_rate.to_f32();
        optimizer.set_learning_rate(current_lr * 0.1);
    }
    
    // Training loop
    // ...
}
```
