# Creation

MaidenX provides multiple ways to create tensors. This section covers the various tensor creation methods available in the library.

## Creating Tensors from Data

### new

Creates a new tensor from data using default device and data type.

```rust
use maidenx_tensor::Tensor;

// Create a tensor from a vector of integers
let x = Tensor::new(vec![1, 2, 3])?;
```

### new_with_spec

Creates a new tensor with specified device and data type. This is useful when you need more control over the tensor's properties.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create a tensor with specific device and data type
let x = Tensor::new_with_spec(vec![1, 2, 3], Device::CPU, DType::I32)?;

// Create a tensor with type conversion (integers to float)
let y = Tensor::new_with_spec(vec![1, 2, 3], Device::CPU, DType::F32)?;
assert_eq!(y.to_flatten_vec::<f32>()?, [1.0, 2.0, 3.0]);
```

## Creating Empty Tensors

### empty

Creates an uninitialized tensor with the specified shape, using default device and data type.

```rust
use maidenx_tensor::Tensor;

// Create an empty tensor of shape [2, 3]
let x = Tensor::empty(&[2, 3])?;
```

### empty_like

Creates an uninitialized tensor with the same shape, device, and data type as the provided tensor.

```rust
use maidenx_tensor::Tensor;

let src = Tensor::new(vec![1, 2, 3, 4, 5, 6])?;
src.with_shape(&[2, 3])?;

// Create an empty tensor with same properties as src
let y = Tensor::empty_like(&src)?;
```

### empty_with_spec

Creates an uninitialized tensor with the specified shape, device, and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create an empty tensor with specific shape, device and data type
let x = Tensor::empty_with_spec(&[2, 3], Device::CPU, DType::F32)?;
```

## Creating Tensors with Constant Values

### zeros

Creates a tensor of the specified shape filled with zeros, using default device and data type.

```rust
use maidenx_tensor::Tensor;

// Create a tensor of shape [2, 3] filled with zeros
let x = Tensor::zeros(&[2, 3])?;
assert_eq!(x.to_flatten_vec::<f32>()?, vec![0.0; 6]);
```

### zeros_like

Creates a tensor filled with zeros with the same shape, device, and data type as the provided tensor.

```rust
use maidenx_tensor::Tensor;

let src = Tensor::new(vec![1, 2, 3, 4, 5, 6])?;
src.with_shape(&[2, 3])?;

// Create a tensor of zeros with same properties as src
let y = Tensor::zeros_like(&src)?;
```

### zeros_with_spec

Creates a tensor filled with zeros with the specified shape, device, and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create a zeros tensor with specific shape, device and data type
let x = Tensor::zeros_with_spec(&[2, 3], Device::CPU, DType::F32)?;
```

### ones

Creates a tensor of the specified shape filled with ones, using default device and data type.

```rust
use maidenx_tensor::Tensor;

// Create a tensor of shape [2, 3] filled with ones
let x = Tensor::ones(&[2, 3])?;
assert_eq!(x.to_flatten_vec::<f32>()?, vec![1.0; 6]);
```

### ones_like

Creates a tensor filled with ones with the same shape, device, and data type as the provided tensor.

```rust
use maidenx_tensor::Tensor;

let src = Tensor::new(vec![1, 2, 3, 4, 5, 6])?;
src.with_shape(&[2, 3])?;

// Create a tensor of ones with same properties as src
let y = Tensor::ones_like(&src)?;
```

### ones_with_spec

Creates a tensor filled with ones with the specified shape, device, and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create a ones tensor with specific shape, device and data type
let x = Tensor::ones_with_spec(&[2, 3], Device::CPU, DType::F32)?;
```

### fill

Creates a tensor of the specified shape filled with a specified value, using default device and data type.

```rust
use maidenx_tensor::Tensor;

// Create a tensor of shape [2, 3] filled with value 5.0
let x = Tensor::fill(&[2, 3], 5.0)?;
assert_eq!(x.to_flatten_vec::<f32>()?, vec![5.0; 6]);
```

### fill_like

Creates a tensor filled with a specified value with the same shape, device, and data type as the provided tensor.

```rust
use maidenx_tensor::Tensor;

let src = Tensor::new(vec![1, 2, 3, 4, 5, 6])?;
src.with_shape(&[2, 3])?;

// Create a tensor filled with 7 with same properties as src
let y = Tensor::fill_like(&src, 7)?;
```

### fill_with_spec

Creates a tensor filled with a specified value with the specified shape, device, and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create a tensor filled with 5.0 with specific shape, device and data type
let x = Tensor::fill_with_spec(&[2, 3], 5.0, Device::CPU, DType::F32)?;
```

## Creating Tensors with Random Values

### randn

Creates a tensor of the specified shape filled with values sampled from a standard normal distribution (mean = 0, std = 1), using default device and data type.

```rust
use maidenx_tensor::Tensor;

// Create a tensor of shape [2, 3] with random normal values
let x = Tensor::randn(&[2, 3])?;
```

### randn_like

Creates a tensor filled with values sampled from a standard normal distribution with the same shape, device, and data type as the provided tensor.

```rust
use maidenx_tensor::Tensor;

let src = Tensor::new(vec![1, 2, 3, 4, 5, 6])?;
src.with_shape(&[2, 3])?;

// Create a tensor with random normal values with same properties as src
let y = Tensor::randn_like(&src)?;
```

### randn_with_spec

Creates a tensor filled with values sampled from a standard normal distribution with the specified shape, device, and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create a random normal tensor with specific shape, device and data type
let x = Tensor::randn_with_spec(&[2, 3], Device::CPU, DType::F32)?;
```

## Creating Sequences

### range

Creates a 1D tensor with values [0, 1, 2, ..., n-1].

```rust
use maidenx_tensor::Tensor;

// Create a tensor with values [0, 1, 2, 3, 4]
let x = Tensor::range(5)?;
assert_eq!(x.to_flatten_vec::<f32>()?, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
```

### range_with_spec

Creates a 1D tensor with values [0, 1, 2, ..., n-1] with the specified device and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create a range tensor with specific device and data type
let x = Tensor::range_with_spec(5, Device::CPU, DType::I32)?;
assert_eq!(x.to_flatten_vec::<i32>()?, vec![0, 1, 2, 3, 4]);
```

### arange

Creates a 1D tensor with values [start, start+step, start+2*step, ...] up to but not including end.

```rust
use maidenx_tensor::Tensor;

// Create a tensor with values [1.0, 2.0, 3.0, 4.0]
let x = Tensor::arange(1.0, 5.0, 1.0)?;

// Create a tensor with values [0.0, 2.0, 4.0, 6.0, 8.0]
let y = Tensor::arange(0, 10, 2)?;

// Create a tensor with negative step [5.0, 4.0, 3.0, 2.0, 1.0]
let z = Tensor::arange(5, 0, -1)?;
```

### arange_with_spec

Creates a 1D tensor with values [start, start+step, start+2*step, ...] up to but not including end, with the specified device and data type.

```rust
use maidenx_core::{device::Device, dtype::DType};
use maidenx_tensor::Tensor;

// Create an arange tensor with specific device and data type
let x = Tensor::arange_with_spec(1, 5, 1, Device::CPU, DType::I32)?;
assert_eq!(x.to_flatten_vec::<i32>()?, vec![1, 2, 3, 4]);

// Create a tensor with float values [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
let y = Tensor::arange_with_spec(0.5, 4.0, 0.5, Device::CPU, DType::F32)?;
```

## Other Creation Methods

### share_buffer

Creates a new tensor that shares the underlying buffer with the provided tensor.

```rust
use maidenx_tensor::Tensor;

let x = Tensor::new(vec![1, 2, 3, 4])?;
let y = Tensor::share_buffer(&x)?;

// Both tensors share the same buffer
assert_eq!(y.to_flatten_vec::<i32>()?, [1, 2, 3, 4]);
```

## Creation Pattern

MaidenX follows a consistent pattern for tensor creation functions:

1. **Basic function**: Takes minimal arguments and uses default device and data type
   ```rust
   Tensor::zeros(&[2, 3])?;
   ```

2. **_like function**: Creates a tensor with the same properties as another tensor
   ```rust
   Tensor::zeros_like(&x)?;
   ```

3. **_with_spec function**: Provides complete control over shape, device, and data type
   ```rust
   Tensor::zeros_with_spec(&[2, 3], Device::CPU, DType::F32)?;
   ```

This consistent pattern makes it easy to understand and use the various creation methods in MaidenX.