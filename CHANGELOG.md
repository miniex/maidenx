# Changelog for MaidenX 0.3.0

# Changelog for MaidenX 0.2.0

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
  - `i32`: Signed 32-bit integer
  - `i64`: Signed 64-bit integer

## Enhancements

## Bug Fixes

## Known Issues

## Deprecated

## Removed

<br />

# Changelog for MaidenX 0.1.0

## Added Features

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
