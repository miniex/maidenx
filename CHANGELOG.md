# Changelog for MaidenX 0.1.1

## Added Features

### Tensor Operations
- **Scalar Operations**:
  - Added `read_scalar` and `write_scalar` for both CPU and CUDA devices. [[bd3d4b5](https://github.com/miniex/maidenx/commit/bd3d4b5)]
- **Memory Management**:
  - Implemented tensor offset support for memory-efficient views. [[82a2e30](https://github.com/miniex/maidenx/commit/82a2e30)]
- **Padding Operations**:
  - Introduced new padding operations for flexible tensor dimension management. [[5531e67](https://github.com/miniex/maidenx/commit/5531e67)]
- **Operator Overriding**:
  - Added operator overriding capabilities for customized tensor operations. [[f1c957b](https://github.com/miniex/maidenx/commit/f1c957b)]
- **Metadata Support**:
  - Added metadata support in tensors for storing additional information. [[030b9f8](https://github.com/miniex/maidenx/commit/030b9f8)]

### Neural Network Components
- **Activations**:
  - Added `LeakyReLU`, `GELU`, and `ELU` activation functions. [[06ff9d2](https://github.com/miniex/maidenx/commit/06ff9d2)]

## Enhancements
- **Buffer Management**:
  - Improved performance by replacing RwLock with Arc for buffer management in tensors. [[af8b5d5](https://github.com/miniex/maidenx/commit/af8b5d5)]
- **Tensor Indexing**:
  - Updated tensor indexing methods to support offsets. [[89a382b](https://github.com/miniex/maidenx/commit/89a382b)]

## Fixed Issues
- Fixed borrowing issues in inplace operations. [[af8b5d5](https://github.com/miniex/maidenx/commit/af8b5d5)]


---

This release introduces additional tensor operations, new activation functions, and performance improvements to MaidenX.

# Changelog for MaidenX 0.1.0

## Added Features

### DType

- **Floating-Point Types**:
  - `bf16`: Brain Floating Point (bfloat16) - 16-bit precision optimized for neural networks
  - `f16`: IEEE 754 Half-Precision (float16) - 16-bit floating-point
  - `f32`: IEEE 754 Single-Precision (float32) - 32-bit floating-point
  - `f64`: IEEE 754 Double-Precision (float64) - 64-bit floating-point

- **Integer Types**:
  - `u8`: Unsigned 8-bit integer
  - `u32`: Unsigned 32-bit integer
  - `i8`: Signed 8-bit integer
  - `i32`: Signed 32-bit integer
  - `i64`: Signed 64-bit integer

### Neural Network Components
- **Modules**:
  - Added `Linear`, `Bilinear`, and `Conv2d` layers supporting CPU and CUDA backends.
- **Activations**:
  - Introduced `ReLU`, `Sigmoid`, and `Tanh` activation functions.
- **Loss Functions**:
  - Added `MSE`, `MAE`, and `Huber` loss functions.
- **Optimizers**:
  - Implemented `SGD` and `Adam` optimizers.

### Tensor Operations
- **Basic Operations**:
  - Element-wise addition, subtraction, multiplication, division, and matrix multiplication with gradient tracking and broadcasting.
- **Shape Operations**:
  - Added operations like `transpose`, `reshape`, and broadcasting utilities.
- **Reduction Operations**:
  - Implemented `sum`, `mean`, and related dimensional operations.
- **Scalar Operations**:
  - Enabled addition, subtraction, multiplication, and division with scalars.
- **Comparison Operations**:
  - Added scalar-based comparisons like `le_scalar` and `gt_scalar`.

## Enhancements
- **Gradient Tracking**: Automatic differentiation support for all operations.
- **Broadcasting**: Fully integrated tensor broadcasting for seamless operations.

## Known Issues
- None reported for this version.

---

This foundational release introduces core tensor operations and neural network components to MaidenX.
