# Installation

MaidenX is a Rust machine learning framework that's available through [crates.io](https://crates.io/crates/maidenx). This guide will walk you through the installation process, including setting up optional hardware acceleration features.

## Basic Installation

To add MaidenX to your Rust project, add it as a dependency in your `Cargo.toml` file:

```toml
[dependencies]
maidenx = "*"
```

This will include the default features (`nn`, `serde`, and `graph`), which are suitable for most use cases.

## Feature Configuration

MaidenX provides several optional features that you can enable based on your needs:

### Default Features

These are included automatically and provide core functionality:

| Feature | Description |
|---------|-------------|
| `nn` | Neural network components (layers, optimizers, activations) |
| `serde` | Serialization/deserialization for saving and loading models |
| `graph` | Computational graph for deferred tensor operations |

### Hardware Acceleration

For improved performance, you can enable hardware-specific backends:

| Feature | Description | Requirements |
|---------|-------------|--------------|
| `cuda` | NVIDIA GPU acceleration | NVIDIA GPU, CUDA toolkit |
| `mps` | Apple Silicon GPU acceleration | Apple Silicon Mac |

To enable specific features, modify your dependency in `Cargo.toml`:

```toml
[dependencies]
maidenx = { version = "*", features = ["cuda"] }  # For NVIDIA GPU support
```

Or:

```toml
[dependencies]
maidenx = { version = "*", features = ["mps"] }  # For Apple Silicon GPU support
```

## Hardware-Specific Setup

### CUDA Backend (NVIDIA GPUs)

To use the CUDA backend:

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (compatible with your NVIDIA GPU)
2. Ensure your system's PATH includes the CUDA binaries
3. Enable the `cuda` feature in your Cargo.toml

### MPS Backend (Apple Silicon)

To use the Metal Performance Shaders backend:

1. Ensure you're using macOS on Apple Silicon hardware (M1/M2/M3)
2. Have Xcode and the Command Line Tools installed
3. Enable the `mps` feature in your Cargo.toml

## Setting Default Device and Data Type

MaidenX allows you to configure the global default device and data type for tensor operations:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check current default device and dtype
    println!("Default device: {:?}", get_default_device());
    println!("Default dtype: {:?}", get_default_dtype());
    
    // Set new defaults
    set_default_device(Device::CPU);
    set_default_dtype(DType::F32);
    
    // Create a tensor using the defaults
    let tensor = Tensor::ones(&[2, 3])?;
    println!("Device: {:?}, dtype: {:?}", tensor.device(), tensor.dtype());
    
    // Automatic device selection based on available hardware
    auto_set_device();
    println!("Auto-selected device: {:?}", get_default_device());
    
    Ok(())
}
```

The `auto_set_device()` function will select the best available device in this order:
1. CUDA if available and the `cuda` feature is enabled
2. MPS if available and the `mps` feature is enabled
3. CPU as fallback

## Verifying Installation

To verify that MaidenX is correctly installed and configured, you can run a simple example:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple tensor
    let tensor = Tensor::ones(&[2, 3])?;
    println!("Tensor shape: {:?}", tensor.shape());
    println!("Tensor device: {:?}", tensor.device());
    
    Ok(())
}
```

If you've enabled hardware acceleration, you can explicitly create tensors on specific devices:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors on different devices
    let cpu_tensor = Tensor::ones(&[2, 3])?.to_device(Device::CPU)?;
    
    #[cfg(feature = "cuda")]
    let cuda_tensor = Tensor::ones(&[2, 3])?.to_device(Device::CUDA(0))?;
    
    #[cfg(feature = "mps")]
    let mps_tensor = Tensor::ones(&[2, 3])?.to_device(Device::MPS)?;
    
    println!("CPU Tensor: {:?}", cpu_tensor);
    
    Ok(())
}
```

## Next Steps

Once you've successfully installed MaidenX, you're ready to start creating and manipulating tensors. Continue to the [Creating Tensors](./create-tensors.md) guide to learn the basics of working with MaidenX's tensor system.