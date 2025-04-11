# Linear Layer

The Linear layer (also known as a fully connected or dense layer) performs a linear transformation on the input data. It's one of the most fundamental building blocks in neural networks.

## Definition

```rust
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    state: LayerState,
}
```

## Constructor

```rust
pub fn new(in_features: usize, out_features: usize, with_bias: bool) -> Result<Self>
```

Creates a new Linear layer with the specified dimensions.

**Parameters**:
- `in_features`: The size of each input sample
- `out_features`: The size of each output sample
- `with_bias`: Whether to include a bias term

**Example**:
```rust
let linear = Linear::new(784, 256, true)?;
```

For more control over the initialization, you can use the extended constructor:

```rust
pub fn new_with_spec(
    in_features: usize, 
    out_features: usize, 
    with_bias: bool, 
    device: Device, 
    dtype: DType
) -> Result<Self>
```

**Additional Parameters**:
- `device`: The device to place the layer's parameters on (CPU, CUDA, or MPS)
- `dtype`: The data type for the layer's parameters

**Example**:
```rust
let linear = Linear::new_with_spec(
    784, 
    256, 
    true, 
    Device::CUDA(0), 
    DType::F32
)?;
```

## Forward Pass

```rust
pub fn forward(&self, input: &Tensor) -> Result<Tensor>
```

Applies the linear transformation y = xW + b.

**Parameters**:
- `input`: The input tensor with shape \[batch_size, ..., in_features\]

**Returns**: Output tensor with shape \[batch_size, ..., out_features\]

**Example**:
```rust
let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let linear = Linear::new(2, 3, true)?;
let output = linear.forward(&input)?; // Shape: [2, 3]
```

## Parameter Access

```rust
pub fn weight(&self) -> &Tensor
pub fn bias(&self) -> Option<&Tensor>
```

Provides access to the layer's weight and bias parameters.

**Example**:
```rust
let linear = Linear::new(2, 3, true)?;
let weight = linear.weight(); // Shape: [3, 2]
let bias = linear.bias().unwrap(); // Shape: [3]
```

## Layer Implementation

The Linear layer implements the `Layer` trait, providing methods for parameter collection and training state management:

```rust
pub fn parameters(&mut self) -> Vec<&mut Tensor>
```

Returns all trainable parameters of the layer (weight and bias if present).

## Mathematical Operation

For an input tensor x of shape \[batch_size, in_features\], the Linear layer computes:

```
output = x @ weight.T + bias
```

Where:
- @ represents the matrix multiplication
- weight.T is the transposed weight matrix of shape \[out_features, in_features\]
- bias is the bias vector of shape \[out_features\]

The output tensor has shape \[batch_size, out_features\].

## Broadcasting Support

The Linear layer supports broadcasting for batched inputs. If the input tensor has additional leading dimensions, they are preserved in the output:

```rust
let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[3, 2])?;
let linear = Linear::new(2, 4, true)?;
let output = linear.forward(&input)?; // Shape: [3, 4]
```

For a more complex batch structure:

```rust
// Input shape: [batch_size, sequence_length, in_features]
let input = Tensor::new(vec![/* values */])?.reshape(&[32, 10, 64])?;
let linear = Linear::new(64, 128, true)?;
let output = linear.forward(&input)?; // Shape: [32, 10, 128]
```
