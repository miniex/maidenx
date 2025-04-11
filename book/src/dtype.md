# Data Types (DType)

MaidenX supports a variety of data types for tensors, allowing you to optimize for memory usage, precision, and performance. The appropriate data type choice can significantly impact your model's accuracy and execution speed.

## Supported Data Types

| Category | Data Type | MaidenX Identifier | Size (bits) | Device Support | Use Cases |
|----------|-----------|-------------------|------------|----------------|-----------|
| **Floating Point** | BFloat16 | `maidenx::bfloat16` | 16 | All | Training with reduced precision |
| | Float16 | `maidenx::float16` | 16 | All | Memory-efficient inference |
| | Float32 | `maidenx::float32` | 32 | All | General training and inference |
| | Float64 | `maidenx::float64` | 64 | CPU, CUDA | High-precision scientific computing |
| **Integer (Unsigned)** | UInt8 | `maidenx::uint8` | 8 | All | Quantized models, image processing |
| | UInt16 | `maidenx::uint16` | 16 | All | Compact indexing |
| | UInt32 | `maidenx::uint32` | 32 | All | Large indices |
| | UInt64 | `maidenx::uint64` | 64 | CPU, CUDA | Very large indices |
| **Integer (Signed)** | Int8 | `maidenx::int8` | 8 | All | Quantized models, efficient storage |
| | Int16 | `maidenx::int16` | 16 | All | Compact representation with sign |
| | Int32 | `maidenx::int32` | 32 | All | General integer operations |
| | Int64 | `maidenx::int64` | 64 | CPU, CUDA | Large integer ranges |
| **Boolean** | Bool | `maidenx::bool` | 1 | All | Masks, condition operations |

## Setting the Default Data Type

You can set the default data type for all tensor operations:

```rust
use maidenx::prelude::*;

// Set default dtype to Float32
set_default_dtype(DType::F32);

// Create tensor with default dtype
let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
```

## Explicit Data Type Specification

You can create tensors with specific data types regardless of the default:

```rust
// Create a tensor with float64 precision
let high_precision = Tensor::new_with_spec(
    vec![1.0, 2.0, 3.0],
    Device::CPU, 
    DType::F64
)?;

// Create an integer tensor
let int_tensor = Tensor::new_with_spec(
    vec![1, 2, 3], 
    Device::CPU, 
    DType::I32
)?;
```

## Type Conversion

Tensors can be converted between data types using `to_dtype`:

```rust
// Convert float32 to float64
let f32_tensor = Tensor::new(vec![1.0f32, 2.0, 3.0])?;
let f64_tensor = f32_tensor.to_dtype(DType::F64)?;

// Convert float to int
let int_tensor = f32_tensor.to_dtype(DType::I32)?;
```

## Boolean Type Handling

Boolean tensors are handled specially depending on the context:

1. **Logical Operations**: Remain as `maidenx::bool`
   ```rust
   let a = Tensor::new(vec![true, false, true])?;
   let b = Tensor::new(vec![false, true, false])?;
   let logical_and = a.logical_and(&b)?; // Still boolean type
   ```

2. **Arithmetic Operations**: Promoted to `maidenx::uint8`
   ```rust
   let bool_tensor = Tensor::new(vec![true, false, true])?;
   let added = bool_tensor.add_scalar(1)?; // Converted to uint8 for addition
   ```

3. **Operations with Floating-Point**: Promoted to `maidenx::float32`
   ```rust
   let bool_tensor = Tensor::new(vec![true, false])?;
   let float_tensor = Tensor::new(vec![1.5, 2.5])?;
   let result = bool_tensor.mul(&float_tensor)?; // Converted to float32
   ```

## Automatic Differentiation and Data Types

Only floating-point data types support automatic differentiation (autograd):

```rust
// This works - float types support gradients
let mut float_tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
float_tensor.with_grad()?; // Enables gradient tracking

// This would fail - integer types don't support gradients
let mut int_tensor = Tensor::new(vec![1, 2, 3])?.to_dtype(DType::I32)?;
// int_tensor.with_grad()?; // Would return an error
```

## Device Compatibility

Not all data types are supported on all devices:

- **64-bit types** (`F64`, `I64`, `U64`) are not supported on MPS (Apple Silicon)
- When using MPS, use 32-bit or smaller types

## Memory and Performance Considerations

- **Float16/BFloat16**: Half-precision can significantly reduce memory usage with minimal accuracy loss for many applications
- **Int8/UInt8**: Quantized models often use 8-bit integers for dramatic memory and performance improvements
- **Float64**: Double precision is rarely needed for machine learning but may be crucial for scientific computing
- **Bool**: Most memory-efficient for masks and conditions, but may be promoted to larger types during operations

## Example: Mixed Precision Training

```rust
use maidenx::prelude::*;

fn main() -> Result<()> {
    // Use float16 for weights to save memory
    let mut weights = Tensor::randn(&[1024, 1024])?.to_dtype(DType::F16)?;
    weights.with_grad()?;
    
    // Use float32 for activation and loss computation for stability
    let input = Tensor::randn(&[32, 1024])?;
    
    // Forward pass (operations automatically convert as needed)
    let output = input.matmul(&weights)?;
    
    // Loss computation in float32 for accuracy
    let target = Tensor::randn(&[32, 1024])?;
    let loss = output.sub(&target)?.pow(2.0)?.mean_all()?;
    
    // Backward pass (gradients handled in appropriate precision)
    loss.backward()?;
    
    Ok(())
}
```

This approach balances memory efficiency with numerical stability.
