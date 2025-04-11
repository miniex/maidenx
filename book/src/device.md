# Device

MaidenX supports multiple computing devices to run tensor operations, allowing you to choose the most suitable hardware for your specific use case. This flexibility lets you develop on one platform and deploy on another without changing your code.

## Supported Devices

| Device | Description | Availability |
|--------|-------------|--------------|
| **CPU** | Standard CPU execution | Always available |
| **CUDA** | NVIDIA GPU acceleration via CUDA | Available with `cuda` feature flag |
| **MPS** | Apple Silicon GPU acceleration via Metal Performance Shaders | Available with `mps` feature flag |
| **Vulkan** | Cross-platform GPU acceleration | Planned for future release |

## Device Selection

You can set the default device for tensor operations using:

```rust
use maidenx::prelude::*;

// Set default device to CPU
set_default_device(Device::CPU);

// Set default device to first CUDA GPU
#[cfg(feature = "cuda")]
set_default_device(Device::CUDA(0));

// Set default device to Apple Silicon GPU
#[cfg(feature = "mps")]
set_default_device(Device::MPS);
```

## Per-Tensor Device Placement

You can also create tensors on specific devices, regardless of the default:

```rust
// Create a tensor on CPU
let cpu_tensor = Tensor::new_with_spec(
    vec![1.0, 2.0, 3.0], 
    Device::CPU, 
    DType::F32
)?;

// Create a tensor on CUDA (if available)
#[cfg(feature = "cuda")]
let cuda_tensor = Tensor::new_with_spec(
    vec![1.0, 2.0, 3.0], 
    Device::CUDA(0), 
    DType::F32
)?;
```

## Moving Tensors Between Devices

Tensors can be moved between devices using the `to_device` method:

```rust
// Move tensor to CPU
let tensor_on_cpu = tensor.to_device(Device::CPU)?;

// Move tensor to CUDA (if available)
#[cfg(feature = "cuda")]
let tensor_on_cuda = tensor.to_device(Device::CUDA(0))?;
```

## Device-Specific Considerations

### CPU

- Available on all platforms
- Good for development and debugging
- Slower for large-scale computations
- No special requirements

### CUDA

- Requires NVIDIA GPU and CUDA toolkit
- Best performance for large models and batch sizes
- Enabled with the `cuda` feature flag
- Supports multiple GPU selection via `Device::CUDA(device_id)`

### MPS (Metal Performance Shaders)

- Available on Apple Silicon (M1/M2/M3) devices
- Good performance on Apple hardware
- Enabled with the `mps` feature flag
- Does not support 64-bit data types (F64, I64, U64)

### Vulkan (Planned)

- Will provide cross-platform GPU acceleration
- Intended to work on various GPUs (NVIDIA, AMD, Intel)
- Not yet implemented

## Example: Multi-Device Code

Here's how to write code that can run on any available device:

```rust
use maidenx::prelude::*;

fn main() -> Result<()> {
    // Choose the best available device
    auto_set_device();
    
    println!("Using device: {}", get_default_device().name());
    
    // Create a tensor (will use the default device)
    let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
    let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
    
    // Operations run on the tensor's device
    let c = a.add(&b)?;
    
    println!("Result: {}", c);
    
    Ok(())
}
```

This code automatically selects the best available device based on feature flags, with CUDA preferred over MPS, and MPS preferred over CPU.
