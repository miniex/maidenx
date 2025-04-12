# Create Tensors

MaidenX provides a variety of methods for creating and initializing tensors. This guide covers the most common tensor creation patterns and provides examples for each method.

## Creating Tensors from Data

The most direct way to create a tensor is from existing data using the `new` method:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor from a vector of values
    let vec_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(vec_data)?;
    println!("1D Tensor: {}", tensor);
    
    // Create a tensor from a 2D vector (matrix)
    let matrix_data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0]
    ];
    let matrix = Tensor::new(matrix_data)?;
    println!("2D Tensor shape: {:?}", matrix.shape());
    println!("2D Tensor: {}", matrix);
    
    Ok(())
}
```

## Creating Tensors with Specific Device and Data Type

You can explicitly specify the device and data type when creating a tensor:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor with specific device and data type
    let data = vec![1, 2, 3, 4];
    let tensor = Tensor::new_with_spec(data, Device::CPU, DType::I32)?;
    
    println!("Device: {:?}", tensor.device());
    println!("Data type: {:?}", tensor.dtype());
    
    Ok(())
}
```

## Creating Pre-initialized Tensors

MaidenX provides several factory methods for creating tensors with pre-initialized values:

### Zeros and Ones

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor filled with zeros
    let zeros = Tensor::zeros(&[2, 3])?;
    println!("Zeros tensor: {}", zeros);
    
    // Create a tensor filled with ones
    let ones = Tensor::ones(&[2, 3])?;
    println!("Ones tensor: {}", ones);
    
    // Create a tensor filled with a specific value
    let filled = Tensor::fill(&[2, 3], 5.0)?;
    println!("Filled tensor: {}", filled);
    
    Ok(())
}
```

### Random Tensors

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor with random values from a normal distribution
    let random = Tensor::randn(&[2, 3])?;
    println!("Random tensor: {}", random);
    
    Ok(())
}
```

## Creating Range Tensors

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor with values [0, 1, 2, 3, 4]
    let range = Tensor::range(5)?;
    println!("Range tensor: {}", range);
    
    // Create a tensor with custom range [1, 3, 5, 7, 9]
    let arange = Tensor::arange(1, 11, 2)?;
    println!("Arange tensor: {}", arange);
    
    Ok(())
}
```

## Creating Tensors Based on Existing Tensors

MaidenX follows PyTorch's pattern of providing `_like` methods that create new tensors with the same properties as existing ones:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a source tensor
    let source = Tensor::randn(&[2, 3])?.to_device(Device::CPU)?.with_dtype(DType::F32)?;
    
    // Create tensors with the same properties as source
    let zeros_like = Tensor::zeros_like(&source)?;
    let ones_like = Tensor::ones_like(&source)?;
    let randn_like = Tensor::randn_like(&source)?;
    let empty_like = Tensor::empty_like(&source)?;
    
    println!("Source shape: {:?}, device: {:?}, dtype: {:?}", 
             source.shape(), source.device(), source.dtype());
    println!("Zeros like shape: {:?}, device: {:?}, dtype: {:?}", 
             zeros_like.shape(), zeros_like.device(), zeros_like.dtype());
    
    Ok(())
}
```

## Configuring Device and Data Type Settings

MaidenX provides functions to manage the default device and data type settings for tensor creation:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get current default settings
    println!("Default device: {:?}", get_default_device());
    println!("Default dtype: {:?}", get_default_dtype());
    
    // Set new defaults
    set_default_device(Device::CPU);
    set_default_dtype(DType::F32);
    
    // All tensors created after this will use these defaults
    let tensor = Tensor::ones(&[2, 3])?;
    println!("Device: {:?}, dtype: {:?}", tensor.device(), tensor.dtype());
    
    // Auto-select the best available device
    auto_set_device();
    println!("Auto-selected device: {:?}", get_default_device());
    
    // Create a tensor using the auto-selected device
    let auto_tensor = Tensor::ones(&[2, 3])?;
    println!("Auto tensor device: {:?}", auto_tensor.device());
    
    Ok(())
}
```

When you call `auto_set_device()`, MaidenX will:
1. Check for CUDA availability if the `cuda` feature is enabled
2. Check for MPS availability if the `mps` feature is enabled 
3. Fall back to CPU if no accelerated device is available

## Hardware Acceleration

If you have enabled the appropriate features, you can create tensors directly on GPU:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor on CUDA device (requires "cuda" feature)
    #[cfg(feature = "cuda")]
    let cuda_tensor = Tensor::ones_with_spec(&[2, 3], Device::CUDA(0), DType::F32)?;
    
    // Create a tensor on MPS device (requires "mps" feature)
    #[cfg(feature = "mps")]
    let mps_tensor = Tensor::ones_with_spec(&[2, 3], Device::MPS, DType::F32)?;
    
    Ok(())
}
```

## Moving Tensors Between Devices

You can move tensors between devices after creation:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor on CPU
    let cpu_tensor = Tensor::ones(&[2, 3])?;
    
    // Move to CUDA device if available
    #[cfg(feature = "cuda")]
    let cuda_tensor = cpu_tensor.to_device(Device::CUDA(0))?;
    
    // Move to MPS device if available
    #[cfg(feature = "mps")]
    let mps_tensor = cpu_tensor.to_device(Device::MPS)?;
    
    Ok(())
}
```

## Enabling Autograd

To make a tensor track gradients for automatic differentiation, use the `with_grad` method:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor that requires gradients
    let mut tensor = Tensor::randn(&[2, 3])?;
    tensor.with_grad()?;
    
    println!("Requires grad: {}", tensor.requires_grad());
    
    Ok(())
}
```
