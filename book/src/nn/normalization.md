# Normalization Layer

Normalization layers help stabilize and accelerate neural network training by standardizing the inputs to each layer. MaidenX provides layer normalization, which normalizes inputs across feature dimensions.

## LayerNorm

Layer normalization normalizes the activations of a single sample, typically across the feature dimension(s). Unlike batch normalization, layer normalization operates on a single example, making it well-suited for scenarios with variable batch sizes or recurrent neural networks.

### Definition

```rust
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    normalized_shape: Vec<usize>,
    eps: f32,
    state: LayerState,
}
```

### Constructor

```rust
pub fn new(normalized_shape: Vec<usize>, with_bias: bool, eps: f32) -> Result<Self>
```

Creates a new layer normalization module.

**Parameters**:
- `normalized_shape`: The shape of the normalized dimensions (usually the feature dimensions)
- `with_bias`: Whether to include a bias term
- `eps`: Small constant added to the denominator for numerical stability

**Example**:
```rust
// For normalizing a feature vector of size 256
let layer_norm = LayerNorm::new(vec![256], true, 1e-5)?;

// For normalizing a 2D feature map of size [32, 64]
let layer_norm_2d = LayerNorm::new(vec![32, 64], true, 1e-5)?;
```

For more control over the initialization, you can use the extended constructor:

```rust
pub fn new_with_spec(
    normalized_shape: Vec<usize>, 
    with_bias: bool, 
    eps: f32, 
    device: Device, 
    dtype: DType
) -> Result<Self>
```

**Additional Parameters**:
- `device`: The device to place the layer's parameters on (CPU, CUDA, or MPS)
- `dtype`: The data type for the layer's parameters

**Example**:
```rust
let layer_norm = LayerNorm::new_with_spec(
    vec![512], 
    true, 
    1e-5,
    Device::CUDA(0),
    DType::F32
)?;
```

### Forward Pass

```rust
pub fn forward(&self, input: &Tensor) -> Result<Tensor>
```

Applies layer normalization to the input tensor.

**Parameters**:
- `input`: Input tensor with shape \[batch_size, ..., *normalized_shape\]

**Returns**: Output tensor with the same shape as input

**Example**:
```rust
let layer_norm = LayerNorm::new(vec![5], true, 1e-5)?;

// Input tensor with shape [2, 5]
let input = Tensor::new(vec![
    vec![1.0, 2.0, 3.0, 4.0, 5.0],
    vec![5.0, 4.0, 3.0, 2.0, 1.0]
])?;

let output = layer_norm.forward(&input)?; // Shape: [2, 5], normalized across dimension 1
```

### Parameter Access

```rust
pub fn weight(&self) -> &Tensor
pub fn bias(&self) -> Option<&Tensor>
```

Provides access to the layer's weight and bias parameters.

**Example**:
```rust
let layer_norm = LayerNorm::new(vec![10], true, 1e-5)?;
let weight = layer_norm.weight(); // Shape: [10]
let bias = layer_norm.bias().unwrap(); // Shape: [10]
```

Other accessor methods include:
- `normalized_shape()`
- `eps()`

### Layer Implementation

The LayerNorm layer implements the `Layer` trait, providing methods for parameter collection and training state management:

```rust
pub fn parameters(&mut self) -> Vec<&mut Tensor>
```

Returns all trainable parameters of the layer (weight and bias if present).

## Mathematical Operation

For an input tensor x of shape \[batch_size, ..., normalized_dims\], LayerNorm computes:

```
y = (x - E[x]) / sqrt(Var[x] + eps) * weight + bias
```

Where:
- E[x] is the mean across the normalized dimensions
- Var[x] is the variance across the normalized dimensions
- weight and bias are learnable parameters with the shape of normalized_dims

## Common Use Cases

### Normalizing Features in MLP

```rust
let mut linear1 = Linear::new(784, 256, true)?;
let mut layer_norm = LayerNorm::new(vec![256], true, 1e-5)?;
let mut linear2 = Linear::new(256, 10, true)?;

// Forward pass
let x1 = linear1.forward(&input)?;
let x2 = layer_norm.forward(&x1)?; // Apply normalization
let x3 = x2.relu()?; // Activation after normalization
let output = linear2.forward(&x3)?;
```

### Normalizing Features in Transformer

```rust
// With attention output of shape [batch_size, seq_len, hidden_size]
let attention_output = /* ... */;

// Normalize over the hidden dimension
let layer_norm = LayerNorm::new(vec![hidden_size], true, 1e-5)?;
let normalized_output = layer_norm.forward(&attention_output)?;
```

### Multi-Dimensional Normalization

```rust
// For a 2D feature map of shape [batch_size, channels, height, width]
let feature_map = /* ... */;

// Reshape to move spatial dimensions to the batch dimension
let reshaped = feature_map.reshape(&[batch_size * channels, height * width])?;

// Normalize over the flattened spatial dimensions
let layer_norm = LayerNorm::new(vec![height * width], true, 1e-5)?;
let normalized = layer_norm.forward(&reshaped)?;

// Reshape back to original shape
let output = normalized.reshape(&[batch_size, channels, height, width])?;
```

## Implementation Notes

- Unlike batch normalization, layer normalization operates independently on each sample
- The weight parameter is initialized to ones and the bias to zeros
- The normalization statistics (mean and variance) are computed at runtime, not stored
- LayerNorm behaves the same during training and evaluation (no separate statistics)
- The normalized_shape parameter specifies the dimensions over which normalization is applied
