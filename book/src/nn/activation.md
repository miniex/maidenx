# Activation Layers

Activation layers apply non-linear functions to their input, which is essential for neural networks to learn complex patterns. MaidenX provides various activation functions as standalone layers.

## Available Activation Layers

### ReLU (Rectified Linear Unit)

```rust
pub struct ReLU {
    state: LayerState,
}
```

The ReLU activation function replaces negative values with zero.

**Constructor**:
```rust
let relu = ReLU::new()?;
```

**Mathematical Function**: f(x) = max(0, x)

**Example**:
```rust
let relu = ReLU::new()?;
let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0])?;
let y = relu.forward(&x)?; // [0.0, 0.0, 0.0, 1.0, 2.0]
```

### Sigmoid

```rust
pub struct Sigmoid {
    state: LayerState,
}
```

The Sigmoid activation squashes input values to the range (0, 1).

**Constructor**:
```rust
let sigmoid = Sigmoid::new()?;
```

**Mathematical Function**: f(x) = 1 / (1 + e^(-x))

**Example**:
```rust
let sigmoid = Sigmoid::new()?;
let x = Tensor::new(vec![-2.0, 0.0, 2.0])?;
let y = sigmoid.forward(&x)?; // [0.119, 0.5, 0.881]
```

### Tanh (Hyperbolic Tangent)

```rust
pub struct Tanh {
    state: LayerState,
}
```

The Tanh activation squashes input values to the range (-1, 1).

**Constructor**:
```rust
let tanh = Tanh::new()?;
```

**Mathematical Function**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Example**:
```rust
let tanh = Tanh::new()?;
let x = Tensor::new(vec![-2.0, 0.0, 2.0])?;
let y = tanh.forward(&x)?; // [-0.964, 0.0, 0.964]
```

### LeakyReLU

```rust
pub struct LeakyReLU {
    exponent: Scalar,
    state: LayerState,
}
```

LeakyReLU allows a small gradient for negative inputs to prevent "dying ReLU" problem.

**Constructor**:
```rust
let leaky_relu = LeakyReLU::new(0.01)?; // 0.01 is a common slope for negative values
```

**Mathematical Function**: f(x) = max(αx, x), where α is typically a small value like 0.01

**Example**:
```rust
let leaky_relu = LeakyReLU::new(0.01)?;
let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0])?;
let y = leaky_relu.forward(&x)?; // [-0.02, -0.01, 0.0, 1.0, 2.0]
```

### GELU (Gaussian Error Linear Unit)

```rust
pub struct GELU {
    state: LayerState,
}
```

GELU activation is used in recent transformer models like BERT and GPT.

**Constructor**:
```rust
let gelu = GELU::new()?;
```

**Mathematical Function**: f(x) = x * Φ(x), where Φ is the cumulative distribution function of the standard normal distribution

**Example**:
```rust
let gelu = GELU::new()?;
let x = Tensor::new(vec![-2.0, 0.0, 2.0])?;
let y = gelu.forward(&x)?; // [-0.046, 0.0, 1.954]
```

### ELU (Exponential Linear Unit)

```rust
pub struct ELU {
    exponent: Scalar,
    state: LayerState,
}
```

ELU uses an exponential function for negative values to allow negative outputs while maintaining smooth gradients.

**Constructor**:
```rust
let elu = ELU::new(1.0)?; // 1.0 is the alpha value
```

**Mathematical Function**: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0

**Example**:
```rust
let elu = ELU::new(1.0)?;
let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0])?;
let y = elu.forward(&x)?; // [-0.865, -0.632, 0.0, 1.0, 2.0]
```

### Softmax

The Softmax activation normalizes inputs into a probability distribution.

**Constructor**:
```rust
let softmax = Softmax::new(dim)?; // dim is the dimension along which to apply softmax
```

**Mathematical Function**: f(x)_i = e^(x_i) / Σ(e^(x_j)) for all j

**Example**:
```rust
let softmax = Softmax::new(-1)?; // Apply along the last dimension
let x = Tensor::new(vec![1.0, 2.0, 3.0])?;
let y = softmax.forward(&x)?; // [0.09, 0.24, 0.67]
```

## Choosing an Activation Function

Different activation functions have different properties and are suitable for different tasks:

- **ReLU**: General purpose, computationally efficient, but can suffer from "dying" neurons
- **LeakyReLU/ELU**: Improved versions of ReLU that help with the dying neuron problem
- **Sigmoid**: Useful for binary classification output layers
- **Tanh**: Similar to sigmoid but with outputs centered around 0
- **GELU**: Often used in transformer models like BERT, GPT, etc.
- **Softmax**: Used in output layers for multi-class classification

## Implementation Notes

All activation layers in MaidenX:

1. Implement the `Layer` trait
2. Require no trainable parameters
3. Support automatic differentiation for backpropagation
4. Have training and evaluation modes (though they behave the same in both modes)
