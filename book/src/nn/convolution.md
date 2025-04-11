# Convolution Layer

The Convolution layer applies a convolution operation to the input data. It's particularly effective for processing grid-structured data such as images.

## Conv2d

The 2D convolution layer operates on 4D tensors with the shape \[batch_size, channels, height, width\].

### Definition

```rust
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    state: LayerState,
}
```

### Constructor

```rust
pub fn new(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    with_bias: bool
) -> Result<Self>
```

Creates a new 2D convolution layer with the specified parameters.

**Parameters**:
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Size of the convolving kernel as (height, width)
- `stride`: Stride of the convolution as (height, width)
- `padding`: Zero-padding added to both sides of the input as (height, width)
- `with_bias`: Whether to include a bias term

**Example**:
```rust
let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), true)?;
```

For more control over the initialization, you can use the extended constructor:

```rust
pub fn new_with_spec(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
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
let conv = Conv2d::new_with_spec(
    3, 
    64,
    (3, 3),
    (1, 1),
    (1, 1),
    true,
    Device::CUDA(0),
    DType::F32
)?;
```

### Forward Pass

```rust
pub fn forward(&self, input: &Tensor) -> Result<Tensor>
```

Applies the convolution operation to the input tensor.

**Parameters**:
- `input`: The input tensor with shape \[batch_size, in_channels, height, width\]

**Returns**: Output tensor with shape \[batch_size, out_channels, output_height, output_width\]

**Example**:
```rust
let input = Tensor::new(vec![/* values */])?.reshape(&[1, 3, 32, 32])?;
let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), true)?;
let output = conv.forward(&input)?; // Shape: [1, 64, 32, 32]
```

### Parameter Access

```rust
pub fn weight(&self) -> &Tensor
pub fn bias(&self) -> Option<&Tensor>
```

Provides access to the layer's weight and bias parameters.

**Example**:
```rust
let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1), true)?;
let weight = conv.weight(); // Shape: [64, 3, 3, 3]
let bias = conv.bias().unwrap(); // Shape: [64]
```

### Layer Implementation

The Conv2d layer implements the `Layer` trait, providing methods for parameter collection and training state management:

```rust
pub fn parameters(&mut self) -> Vec<&mut Tensor>
```

Returns all trainable parameters of the layer (weight and bias if present).

## Output Dimensions

For a given input dimensions, the output dimensions of the convolution are computed as:

```
output_height = (input_height + 2 * padding.0 - kernel_size.0) / stride.0 + 1
output_width = (input_width + 2 * padding.1 - kernel_size.1) / stride.1 + 1
```

## Implementation Details

The MaidenX Conv2d implementation uses the im2col algorithm for efficient computation:

1. The input tensor is transformed into a matrix where each column contains the values in a sliding window
2. Matrix multiplication is performed between this transformed matrix and the flattened kernel weights
3. The result is reshaped back to the expected output dimensions

This approach allows leveraging optimized matrix multiplication operations for convolution.

## Common Configurations

Here are some common Conv2d configurations:

### Basic Convolution (Same Padding)

```rust
// Maintains spatial dimensions
let conv = Conv2d::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), true)?;
```

### Strided Convolution (Downsampling)

```rust
// Reduces spatial dimensions by half
let conv = Conv2d::new(in_channels, out_channels, (3, 3), (2, 2), (1, 1), true)?;
```

### 1x1 Convolution (Channel Mixing)

```rust
// Changes channel dimensions only
let conv = Conv2d::new(in_channels, out_channels, (1, 1), (1, 1), (0, 0), true)?;
```
