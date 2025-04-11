# Tensor Operations

maidenx provides a comprehensive set of tensor operations for numerical computing and deep learning. This page provides an overview of the major operation categories.

## Operation Categories

maidenx tensor operations are organized into the following categories:

| Category | Description |
|----------|-------------|
| [Binary Operations](./ops_binary.md) | Operations between two tensors (add, mul, div, etc.) |
| [Unary Operations](./ops_unary.md) | Operations on a single tensor (neg, abs, exp, etc.) |
| [Reduction Operations](./ops_reduction.md) | Operations that reduce tensor dimensions (sum, mean, max, etc.) |
| [Transform Operations](./ops_transform.md) | Operations that transform tensor shape or layout |
| [Padding Operations](./ops_padding.md) | Operations that add padding around tensor boundaries |
| [Indexing Operations](./ops_indexing.md) | Operations for advanced indexing and selection |

## Common Operation Features

Most operations in maidenx share these common features:

### Automatic Differentiation (Autograd)

Many operations support automatic differentiation, which is crucial for training neural networks. Operations that support autograd will automatically track gradients when the tensor has `requires_grad` enabled.

```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?.with_grad()?;
let b = a.mul_scalar(2.0)?;  // 'b' will also have autograd enabled
```

### Type Promotion

When performing operations between tensors of different data types, maidenx automatically promotes types according to standard rules:

- If one tensor is floating point and one is integer, the integer tensor is converted to floating point
- When mixing different floating point precisions, the lower precision is promoted to the higher one

```rust
let a = Tensor::new(vec![1, 2, 3])?;  // i32 tensor
let b = Tensor::new(vec![1.0, 2.0, 3.0])?;  // f32 tensor
let c = a.add(&b)?;  // Result will be f32
```

### Broadcasting

Most operations support broadcasting, which allows operations between tensors of different shapes by implicitly expanding the smaller tensor:

```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![1.0])?;
let c = a.add(&b)?;  // [2.0, 3.0, 4.0]
```

### Error Handling

All operations return a `Result` type, which allows for clear error handling:

```rust
match tensor.add(&other_tensor) {
    Ok(result) => println!("Addition successful"),
    Err(e) => println!("Error: {}", e),
}
```

## Operation Examples

Here are some common operation examples:

```rust
// Create some tensors
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0])?.reshape(&[2, 2])?;

// Binary operations
let sum = a.add(&b)?;  // [[6.0, 8.0], [10.0, 12.0]]
let product = a.mul(&b)?;  // [[5.0, 12.0], [21.0, 32.0]]

// Unary operations
let neg_a = a.neg()?;  // [[-1.0, -2.0], [-3.0, -4.0]]
let exp_a = a.exp()?;  // Element-wise exponential

// Reduction operations
let sum_a = a.sum(0, false)?;  // [4.0, 6.0]
let max_a = a.max_all()?;  // 4.0

// Transform operations
let reshaped = a.reshape(&[4])?;  // [1.0, 2.0, 3.0, 4.0]
let transposed = a.transpose(0, 1)?;  // [[1.0, 3.0], [2.0, 4.0]]

// Indexing operations
let indices = Tensor::new(vec![0])?.reshape(&[1])?;
let first_row = a.index_select(0, &indices)?;  // [[1.0, 2.0]]
```

For detailed documentation on each operation category, please refer to the specific section pages linked above.