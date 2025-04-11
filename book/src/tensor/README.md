# Tensor

This chapter covers MaidenX's core tensor functionality, which provides multi-dimensional array operations with automatic differentiation (autograd) support.

## Core Tensor Features

MaidenX tensors provide:

- Multi-dimensional array representation
- Support for various data types (F32, F64, I32, I64, etc.)
- Automatic differentiation for gradient-based optimization
- Device support (CPU, CUDA, MPS)
- Broadcasting for performing operations between tensors of different shapes
- Extensive operation library (arithmetic, reduction, transformation, etc.)

## Tensor Structure

The `Tensor` struct is the primary data structure for representing multi-dimensional arrays:

```rust
pub struct Tensor {
    data: TensorData,          // Holds buffer and gradient information
    metadata: TensorMetadata,  // Holds device, dtype, layout, requires_grad
    node: Option<TensorNode>,  // Stores computational graph information for autograd
}
```

## Display and Debug Output

MaidenX tensors implement both the `Display` and `Debug` traits for convenient printing:

### Display Format

The Display format shows just the tensor's data in a nested array format:

```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
println!("{}", a);  // Outputs: [1.00000000, 2.00000000, 3.00000000]

let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
println!("{}", b);  // Outputs: [[1.00000000, 2.00000000], [3.00000000, 4.00000000]]
```

### Debug Format

The Debug format provides comprehensive information about the tensor, including shape, device, data type, data values, and gradient information:

```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0])?;
a.with_grad()?;
println!("{:?}", a);
// Outputs something like:
// Tensor(shape=[3], device=cpu, dtype=f32, data=[1.00000000, 2.00000000, 3.00000000], requires_grad=true, grad=[0.00000000, 0.00000000, 0.00000000])
```

## Serialization and Deserialization

MaidenX supports tensor serialization and deserialization through Serde (when the "serde" feature is enabled):

```rust
// Binary serialization
let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
let bytes = tensor.to_bytes()?;
let tensor_from_bytes = Tensor::from_bytes(&bytes)?;

// JSON serialization
let json = tensor.to_json()?;
let tensor_from_json = Tensor::from_json(&json)?;
```

The serialization preserves:
- Tensor data
- Shape and layout information
- Device information
- Data type
- Requires grad flag (but not gradient values)

However, computational graph information (the `node` field) is not serialized, so autograd history is not preserved.

## Getting Started

For detailed guides on tensor operations, see the following sections:
- [Tensor Creation](./creation.md): Ways to create and initialize tensors
- [Tensor Operations](./operation.md): Overview of tensor operations
- [Tensor Utilities](./utils.md): Utility functions for tensor manipulation
