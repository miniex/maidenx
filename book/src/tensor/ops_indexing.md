# Indexing Operations

Indexing operations in maidenx allow for advanced manipulation and selection of tensor data using indices.

## Basic Indexing

### index
```rust
fn index(&self, indices: &Tensor) -> Result<Tensor>
```
Selects values along dimension 0 using indices (an alias for index_select with dim=0).

- **Parameters**:
  - `indices`: Tensor of indices to select
- **Returns**: A new tensor with selected values
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let indices = Tensor::new(vec![0, 2, 4])?.reshape(&[3])?;
let b = a.index(&indices)?; // [1.0, 3.0, 5.0]
```

## Advanced Indexing

### index_select
```rust
fn index_select(&self, dim: impl Into<Scalar>, indices: &Tensor) -> Result<Tensor>
```
Selects values along a specified dimension using indices.

- **Parameters**:
  - `dim`: The dimension to select from
  - `indices`: Tensor of indices to select
- **Returns**: A new tensor with selected values
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
let indices = Tensor::new(vec![0, 2])?.reshape(&[2])?;
let b = a.index_select(1, &indices)?;
// [[1.0, 3.0], [4.0, 6.0]]
```

### gather
```rust
fn gather(&self, dim: impl Into<Scalar>, index: &Tensor) -> Result<Tensor>
```
Gathers values from a tensor using an index tensor.

- **Parameters**:
  - `dim`: The dimension to gather from
  - `index`: Tensor of indices to gather
- **Returns**: A new tensor with gathered values
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
let indices = Tensor::new(vec![0, 0, 1])?.reshape(&[1, 3])?;
let b = a.gather(1, &indices)?;
// [[1.0, 1.0, 2.0]]
```

## In-place Modification Operations

### index_add_
```rust
fn index_add_(&mut self, dim: impl Into<Scalar>, indices: &Tensor, src: &Tensor) -> Result<()>
```
Adds values from the source tensor to specified indices along a dimension.

- **Parameters**:
  - `dim`: The dimension to add to
  - `indices`: Tensor of indices where to add
  - `src`: Tensor containing values to add
- **Returns**: Result indicating success
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let indices = Tensor::new(vec![0, 2])?.reshape(&[2])?;
let src = Tensor::new(vec![10.0, 20.0])?.reshape(&[2])?;
a.index_add_(0, &indices, &src)?;
// a becomes [11.0, 2.0, 23.0, 4.0, 5.0]
```

### index_put_
```rust
fn index_put_(&mut self, indices: &[usize], src: &Tensor) -> Result<()>
```
Puts values from the source tensor into the specified indices.

- **Parameters**:
  - `indices`: List of indices where to put values
  - `src`: Tensor containing values to put
- **Returns**: Result indicating success
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?.reshape(&[5])?;
let src = Tensor::new(vec![10.0, 20.0])?.reshape(&[2])?;
a.index_put_(&[1], &src)?;
// a becomes [1.0, 10.0, 20.0, 4.0, 5.0]
```

### scatter_add_
```rust
fn scatter_add_(&mut self, dim: impl Into<Scalar>, index: &Tensor, src: &Tensor) -> Result<()>
```
Adds values from the source tensor into self at specified indices along a dimension.

- **Parameters**:
  - `dim`: The dimension to scatter add to
  - `index`: Tensor of indices where to add
  - `src`: Tensor containing values to add
- **Returns**: Result indicating success
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::zeros(&[5])?;
let indices = Tensor::new(vec![0, 2, 0])?.reshape(&[3])?;
let src = Tensor::new(vec![1.0, 2.0, 3.0])?.reshape(&[3])?;
a.scatter_add_(0, &indices, &src)?;
// a becomes [4.0, 0.0, 2.0, 0.0, 0.0]
```

## Counting Operations

### bincount
```rust
fn bincount(&self, weights: Option<&Tensor>, minlength: Option<usize>) -> Result<Tensor>
```
Counts the frequency of each value in a tensor of non-negative integers.

- **Parameters**:
  - `weights`: Optional tensor of weights
  - `minlength`: Minimum length of the output tensor
- **Returns**: A new tensor with bin counts
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![0, 1, 1, 3, 2, 1, 3])?.reshape(&[7])?;
let b = a.bincount(None, None)?;
// [1, 3, 1, 2] (0 appears once, 1 appears three times, etc.)

// With weights
let weights = Tensor::new(vec![0.5, 1.0, 1.0, 2.0, 1.5, 1.0, 2.0])?.reshape(&[7])?;
let c = a.bincount(Some(&weights), None)?;
// [0.5, 3.0, 1.5, 4.0]
```

## Performance Considerations

When using indexing operations in performance-critical code, consider these tips:

1. **Avoid repeated indexing**: Instead of accessing individual elements in a loop, try to use vectorized operations.
2. **Use contiguous tensors**: Indexing operations are faster on contiguous tensors.
3. **Batch operations**: When possible, use batch operations like `index_select` rather than selecting individual elements.
4. **Consider in-place operations**: When appropriate, in-place operations like `index_add_` can be more memory-efficient.