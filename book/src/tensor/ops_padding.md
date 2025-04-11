# Padding Operations

Padding operations in maidenx add values around the borders of a tensor, expanding its dimensions.

## Basic Padding

### pad
```rust
fn pad(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor>
```
Pads a tensor with a constant value (alias for pad_with_constant).

- **Parameters**:
  - `paddings`: List of (before, after) padding pairs for each dimension
  - `pad_value`: The value to pad with
- **Returns**: A new tensor with padding applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[3])?;
let b = a.pad(&[(1, 2)], 0.0)?; // [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
```

## Padding Modes

### pad_with_constant
```rust
fn pad_with_constant(&self, paddings: &[(usize, usize)], pad_value: impl Into<Scalar>) -> Result<Tensor>
```
Pads a tensor with a constant value.

- **Parameters**:
  - `paddings`: List of (before, after) padding pairs for each dimension
  - `pad_value`: The value to pad with
- **Returns**: A new tensor with constant padding
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.pad_with_constant(&[(0, 1), (1, 0)], 0.0)?;
// [[0.0, 1.0, 2.0],
//  [0.0, 3.0, 4.0],
//  [0.0, 0.0, 0.0]]
```

### pad_with_reflection
```rust
fn pad_with_reflection(&self, paddings: &[(usize, usize)]) -> Result<Tensor>
```
Pads a tensor by reflecting the tensor values at the boundaries.

- **Parameters**:
  - `paddings`: List of (before, after) padding pairs for each dimension
- **Returns**: A new tensor with reflection padding
- **Supports Autograd**: Yes
- **Note**: Reflection padding requires the input dimension to be greater than 1
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let b = a.pad_with_reflection(&[(2, 2)])?;
// [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
```

### pad_with_replication
```rust
fn pad_with_replication(&self, paddings: &[(usize, usize)]) -> Result<Tensor>
```
Pads a tensor by replicating the edge values.

- **Parameters**:
  - `paddings`: List of (before, after) padding pairs for each dimension
- **Returns**: A new tensor with replication padding
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let b = a.pad_with_replication(&[(2, 2)])?;
// [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]
```

## Multi-dimensional Padding

For multi-dimensional tensors, padding is applied to each dimension separately based on the provided padding pairs. This allows for complex padding patterns to be created.

### Example: 2D Padding

```rust
// Create a 2D tensor
let a = Tensor::new(vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
])?.reshape(&[2, 3])?;

// Pad with zeros: 1 row at top, 1 row at bottom, 2 columns on left, 1 column on right
let b = a.pad_with_constant(&[(1, 1), (2, 1)], 0.0)?;
// Result:
// [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//  [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
//  [0.0, 0.0, 4.0, 5.0, 6.0, 0.0],
//  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
```

## Padding Behavior with Autograd

All padding operations support automatic differentiation (autograd). During the backward pass, gradients from the padded regions are properly handled:
- For constant padding, gradients in the padded regions are ignored
- For reflection and replication padding, gradients are properly accumulated into the original tensor

This makes padding operations safe to use in training neural networks or other gradient-based optimization tasks.