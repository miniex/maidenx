# Reduction Operations

Reduction operations in maidenx reduce tensors along specified dimensions, combining multiple elements into fewer outputs using a specific aggregation function.

## Sum Operations

### sum
```rust
fn sum(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor>
```
Computes the sum of elements along a specified dimension.

- **Parameters**:
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.sum(0, false)?; // [4.0, 6.0]
let c = a.sum(1, true)?; // [[3.0], [7.0]]
```

### sum_all
```rust
fn sum_all(&self) -> Result<Tensor>
```
Computes the sum of all elements in the tensor.

- **Returns**: A new scalar tensor containing the sum
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.sum_all()?; // [10.0]
```

### sum_to_shape
```rust
fn sum_to_shape(&self, shape: &[usize]) -> Result<Tensor>
```
Reduces a tensor to the specified shape by summing along dimensions where the size differs.

- **Parameters**:
  - `shape`: The target shape
- **Returns**: A new tensor with target shape
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
let b = a.sum_to_shape(&[1, 3])?; // [[4.0, 6.0, 8.0]]
```

## Mean Operations

### mean
```rust
fn mean(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor>
```
Computes the mean of elements along a specified dimension.

- **Parameters**:
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.mean(0, false)?; // [2.0, 3.0]
let c = a.mean(1, true)?; // [[1.5], [3.5]]
```

### mean_all
```rust
fn mean_all(&self) -> Result<Tensor>
```
Computes the mean of all elements in the tensor.

- **Returns**: A new scalar tensor containing the mean
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.mean_all()?; // [2.5]
```

## Fold and Unfold Operations

### fold
```rust
fn fold(&self, dim: impl Into<Scalar>, size: impl Into<Scalar>, step: impl Into<Scalar>) -> Result<Tensor>
```
Folds (combines) a tensor along a dimension, reversing the unfold operation.

- **Parameters**:
  - `dim`: The dimension to fold along
  - `size`: The size of each fold
  - `step`: The step size between each fold
- **Returns**: A new tensor with the dimension folded
- **Supports Autograd**: Yes
- **Example**:
```rust
// First unfold, then fold to demonstrate the operation
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let unfolded = a.unfold(0, 2, 1)?; // [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
let folded = unfolded.fold(0, 2, 1)?; // [1.0, 2.0, 3.0, 4.0, 5.0]
```

## Min/Max Operations

### max
```rust
fn max(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor>
```
Finds the maximum values along a specified dimension.

- **Parameters**:
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.max(0, false)?; // [3.0, 4.0]
let c = a.max(1, true)?; // [[2.0], [4.0]]
```

### max_all
```rust
fn max_all(&self) -> Result<Tensor>
```
Finds the maximum value across all elements in the tensor.

- **Returns**: A new scalar tensor containing the maximum
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.max_all()?; // [4.0]
```

### min
```rust
fn min(&self, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor>
```
Finds the minimum values along a specified dimension.

- **Parameters**:
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.min(0, false)?; // [1.0, 2.0]
let c = a.min(1, true)?; // [[1.0], [3.0]]
```

### min_all
```rust
fn min_all(&self) -> Result<Tensor>
```
Finds the minimum value across all elements in the tensor.

- **Returns**: A new scalar tensor containing the minimum
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.min_all()?; // [1.0]
```

## Norm Operations

### norm
```rust
fn norm(&self, p: impl Into<Scalar>, dim: impl Into<Scalar>, keep_dim: bool) -> Result<Tensor>
```
Computes the p-norm of elements along a specified dimension.

- **Parameters**:
  - `p`: The order of the norm (1 for L1, 2 for L2, etc.)
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![3.0, 4.0])?.reshape(&[2])?;
let b = a.norm(2.0, 0, false)?; // [5.0]  (L2 norm: sqrt(3^2 + 4^2))
```

### norm_all
```rust
fn norm_all(&self, p: impl Into<Scalar>) -> Result<Tensor>
```
Computes the p-norm of all elements in the tensor.

- **Parameters**:
  - `p`: The order of the norm (1 for L1, 2 for L2, etc.)
- **Returns**: A new scalar tensor containing the norm
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![3.0, 0.0, 4.0])?.reshape(&[3])?;
let b = a.norm_all(2.0)?; // [5.0]  (L2 norm: sqrt(3^2 + 0^2 + 4^2))
```

## Statistical Operations

### var
```rust
fn var(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Result<Tensor>
```
Computes the variance of elements along a specified dimension.

- **Parameters**:
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
  - `unbiased`: Whether to use Bessel's correction (N-1 divisor)
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.var(0, false, true)?; // [2.0, 2.0]
```

### std
```rust
fn std(&self, dim: impl Into<Scalar>, keep_dim: bool, unbiased: bool) -> Result<Tensor>
```
Computes the standard deviation of elements along a specified dimension.

- **Parameters**:
  - `dim`: The dimension to reduce
  - `keep_dim`: Whether to keep the reduced dimension as 1
  - `unbiased`: Whether to use Bessel's correction (N-1 divisor)
- **Returns**: A new tensor with reduced dimension
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = a.std(0, false, true)?; // [1.414..., 1.414...]  (sqrt of variance)
```