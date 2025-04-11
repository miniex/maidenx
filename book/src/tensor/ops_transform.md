# Transform Operations

Transform operations in maidenx modify the shape, layout, or organization of a tensor without changing its underlying values.

## View Operations

View operations create a new tensor that shares storage with the original tensor but has a different shape or organization.

### view
```rust
fn view<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Tensor>
```
Creates a view of the tensor with a new shape.

- **Parameters**:
  - `shape`: The new shape (use -1 for one dimension to be inferred)
- **Returns**: A new tensor with the same data but different shape
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
let b = a.view(&[6])?; // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
let c = a.view(&[3, 2])?; // [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
```

### squeeze
```rust
fn squeeze(&self, dim: impl Into<Scalar>) -> Result<Tensor>
```
Removes a dimension of size 1 from the tensor.

- **Parameters**:
  - `dim`: The dimension to remove
- **Returns**: A new tensor with the specified dimension removed if it's size 1
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[1, 3])?;
let b = a.squeeze(0)?; // [1.0, 2.0, 3.0]
```

### squeeze_all
```rust
fn squeeze_all(&self) -> Result<Tensor>
```
Removes all dimensions of size 1 from the tensor.

- **Returns**: A new tensor with all size 1 dimensions removed
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[1, 3, 1])?;
let b = a.squeeze_all()?; // [1.0, 2.0, 3.0]
```

### unsqueeze
```rust
fn unsqueeze(&self, dim: impl Into<Scalar>) -> Result<Tensor>
```
Adds a dimension of size 1 at the specified position.

- **Parameters**:
  - `dim`: The position to insert the new dimension
- **Returns**: A new tensor with an additional dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[3])?;
let b = a.unsqueeze(0)?; // [[1.0, 2.0, 3.0]]
let c = a.unsqueeze(1)?; // [[1.0], [2.0], [3.0]]
```

## Layout Operations

Layout operations modify how the tensor is stored in memory or accessed.

### transpose
```rust
fn transpose(&self, dim0: impl Into<Scalar>, dim1: impl Into<Scalar>) -> Result<Tensor>
```
Swaps two dimensions of a tensor.

- **Parameters**:
  - `dim0`: First dimension to swap
  - `dim1`: Second dimension to swap
- **Returns**: A new tensor with the dimensions swapped
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
// [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let b = a.transpose(0, 1)?;
// [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

### slice
```rust
fn slice(&self, dim: impl Into<Scalar>, start: impl Into<Scalar>, end: Option<impl Into<Scalar>>, step: impl Into<Scalar>) -> Result<Tensor>
```
Creates a view that is a slice of the tensor along a dimension.

- **Parameters**:
  - `dim`: The dimension to slice
  - `start`: The starting index
  - `end`: The ending index (exclusive, optional)
  - `step`: The step size
- **Returns**: A new tensor with the slice
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let b = a.slice(0, 1, Some(4), 1)?; // [2.0, 3.0, 4.0]
let c = a.slice(0, 0, Some(5), 2)?; // [1.0, 3.0, 5.0]
```

### unfold
```rust
fn unfold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Tensor>
```
Extracts sliding local blocks from a tensor along a dimension.

- **Parameters**:
  - `dim`: The dimension to unfold
  - `size`: The size of each slice
  - `step`: The step between each slice
- **Returns**: A new tensor with the dimension unfolded
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let b = a.unfold(0, 2, 1)?;
// [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
```

## Broadcasting Operations

Broadcasting operations expand a tensor to match a larger shape for element-wise operations.

### broadcast
```rust
fn broadcast(&self, shape: &[usize]) -> Result<Tensor>
```
Broadcasts a tensor to a new shape.

- **Parameters**:
  - `shape`: The target shape
- **Returns**: A new tensor expanded to the target shape
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[3])?;
let b = a.broadcast(&[2, 3])?;
// [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
```

### broadcast_like
```rust
fn broadcast_like(&self, other: &Tensor) -> Result<Tensor>
```
Broadcasts a tensor to match the shape of another tensor.

- **Parameters**:
  - `other`: The tensor to match shape with
- **Returns**: A new tensor with the shape of the other tensor
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[3])?;
let template = Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0])?.reshape(&[2, 3])?;
let b = a.broadcast_like(&template)?;
// [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
```

### broadcast_left
```rust
fn broadcast_left(&self, batch_dims: &[usize]) -> Result<Tensor>
```
Adds batch dimensions to the left of the tensor shape.

- **Parameters**:
  - `batch_dims`: The batch dimensions to add
- **Returns**: A new tensor with batch dimensions added
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[3])?;
let b = a.broadcast_left(&[2, 2])?;
// Shape: [2, 2, 3] (tensor of shape 3 repeated in a 2x2 grid)
```

## Reshape Operations

Reshape operations change the shape of a tensor, potentially rearranging elements.

### reshape
```rust
fn reshape<T: Into<Scalar> + Clone>(&self, shape: &[T]) -> Result<Tensor>
```
Reshapes a tensor to a new shape.

- **Parameters**:
  - `shape`: The new shape (use -1 for one dimension to be inferred)
- **Returns**: A new tensor with the specified shape
- **Supports Autograd**: Yes
- **Note**: Unlike `view`, `reshape` may copy data if the tensor is not contiguous
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
let b = a.reshape(&[3, 2])?; // [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
let c = a.reshape(&[-1])?; // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

## Broadcasting Rules

When using operations that support broadcasting (like binary operations), maidenx follows these rules:

1. If tensors have different number of dimensions, the shape of the tensor with fewer dimensions is padded with 1s on the left until both shapes have the same length.
2. For each dimension pair, they must either be equal or one of them must be 1.
3. In dimensions where one size is 1 and the other is greater, the tensor with size 1 is expanded to match the other.

For example:
- Shape [3] can broadcast with [2, 3] to produce [2, 3]
- Shape [2, 1] can broadcast with [2, 3] to produce [2, 3]
- Shape [3, 1] can broadcast with [1, 4] to produce [3, 4]