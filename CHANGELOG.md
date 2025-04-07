# MaidenX Changelog

## Version 0.1.4

### New Features
- **Performance**: Enjoy enhanced performance on Apple Silicon devices with the new Metal Performance Shaders (MPS) backend support, enabling hardware acceleration for various tensor operations. [[#9](https://github.com/miniex/maidenx/pull/9)]
- **Buffer Management**: Experience improved data handling with the new BufferManager, designed to streamline memory management and enhance performance. [[993a075](https://github.com/miniex/maidenx/commit/993a075)]
- **Serialization**: Benefit from the new serialization and deserialization support, allowing for easier data handling and storage across different formats. [[b726e79](https://github.com/miniex/maidenx/commit/b726e79)]
- **Examples**: Explore new examples demonstrating serialization features and reorganized device examples for better clarity and usability. [[0a26aa8](https://github.com/miniex/maidenx/commit/0a26aa8)]

### Improvements
- **Core Functionality**: The buffer interface is redesigned to include offset support, providing more flexibility in data management and safer memory access. [[c9b8180](https://github.com/miniex/maidenx/commit/c9b8180)]
- **Tensor Operations**: The to_flatten_vec method now supports additional data types (u16, u64, i16), enhancing its versatility for various applications. [[cffeaf1](https://github.com/miniex/maidenx/commit/cffeaf1)]

### Others
- **Documentation**: The feature support tables are updated to replace CPU, CUDA, and MPS columns with a new Notes column for clearer information presentation. [[6dc7c23](https://github.com/miniex/maidenx/commit/6dc7c23)]

## Version 0.1.3

### New Features

#### Neural Network
- Added Embedding layer with support for padding and gradient scaling [[71915be](https://github.com/miniex/maidenx/commit/71915be)]

#### Tensor Operations
- Added tensor reduction operations including norm, variance, and standard deviation [[0076a86](https://github.com/miniex/maidenx/commit/0076a86)]
- Implemented stride and offset support for padding operations to handle non-contiguous tensors [[d52daa9](https://github.com/miniex/maidenx/commit/d52daa9)]

#### Indexing Operations
- Added index_select and index_add_ functions to enhance tensor indexing capabilities [[aaebba8](https://github.com/miniex/maidenx/commit/aaebba8)]
- Added bincount operation for efficient counting of occurrences in tensors [[222d9e4](https://github.com/miniex/maidenx/commit/222d9e4)]

#### Data Types
- Added support for u16, u64, and i16 data types for more flexibility in tensor operations [[7bb020c](https://github.com/miniex/maidenx/commit/7bb020c)]

### Improvements
- Enhanced performance of the contiguous operation for CUDA tensors [[#6](https://github.com/miniex/maidenx/issues/6), [68b342f](https://github.com/miniex/maidenx/commit/68b342f)]
- Unified type promotion across all tensor operations, enhancing compatibility and reducing errors [[316dd26](https://github.com/miniex/maidenx/commit/316dd26)]

### Fixes
- Fixed gather operation to ensure the output uses the source tensor's data type [[91128cd](https://github.com/miniex/maidenx/commit/91128cd)]
- Fixed alignment issues in CPU kernels to match the CUDA implementation [[040e2f1](https://github.com/miniex/maidenx/commit/040e2f1)]
- Fixed overflow issues in read/write_scalar operations for u8 and i8 tensor types [[#5](https://github.com/miniex/maidenx/issues/5), [fd901bc](https://github.com/miniex/maidenx/commit/fd901bc)]

## Version 0.1.2

### New Features

#### Mathematical Operations
- Added trigonometric functions (sine, cosine, tangent) [[6ae44d3](https://github.com/miniex/maidenx/commit/6ae44d3)]
- Added logarithmic and exponential functions (ln, log10, log2, exp, exp2, exp10) [[bcd79bb](https://github.com/miniex/maidenx/commit/bcd79bb)]

#### Neural Network
- Implemented LayerNorm with keepdims option for mean function [[d7face4](https://github.com/miniex/maidenx/commit/d7face4)]
- Added Dropout layer and LayerState for independent training mode tracking [[4981cac](https://github.com/miniex/maidenx/commit/4981cac)]
- Added CrossEntropyLoss implementation [[31aa816](https://github.com/miniex/maidenx/commit/31aa816)]

#### Tensor Operations
- Added arange and range functions for evenly spaced tensors [[8b9ba03](https://github.com/miniex/maidenx/commit/8b9ba03)]
- Added gather and scatter operations for indexed tensor manipulation [[076f2a0](https://github.com/miniex/maidenx/commit/076f2a0)]
- Added min and max operations [[89fda3b](https://github.com/miniex/maidenx/commit/89fda3b)]
- Added recip operation for calculating reciprocals [[93dd124](https://github.com/miniex/maidenx/commit/93dd124)]
- Added softmax function [[f39cbdf](https://github.com/miniex/maidenx/commit/f39cbdf)]
- Added broadcast_like function [[b2f4d4e](https://github.com/miniex/maidenx/commit/b2f4d4e)]
- Added log function with ln alias [[b4259f0](https://github.com/miniex/maidenx/commit/b4259f0)]
- Enjoy boolean support for scalar comparison operations, enhancing flexibility in your computations. [[316dd26](https://github.com/miniex/maidenx/commit/316dd26)]

### Improvements
- Optimized im2col and col2im CUDA kernels [[a01cca5](https://github.com/miniex/maidenx/commit/a01cca5)]
- Added keep_dims parameter to the sum function [[d2a717f](https://github.com/miniex/maidenx/commit/d2a717f)]
- Improved naming of loss function classes [[5344285](https://github.com/miniex/maidenx/commit/5344285)]
- Fixed boolean promotion issues and scalar type conversion in arange function [[c31db84](https://github.com/miniex/maidenx/commit/c31db84)]

### Fixes
- Fixed compilation errors in softmax CUDA kernel [[14aea5e](https://github.com/miniex/maidenx/commit/14aea5e)]
- Fixed namespace from nn::functional to nn::alias [[5d2fbbc](https://github.com/miniex/maidenx/commit/5d2fbbc)]
- Corrected minor typos in the codebase [[d290383](https://github.com/miniex/maidenx/commit/d290383), [aa7041f](https://github.com/miniex/maidenx/commit/aa7041f), [65affd1](https://github.com/miniex/maidenx/commit/65affd1)]

## Version 0.1.1

### Added Features

#### Tensor Operations
- Added read_scalar and write_scalar for both CPU and CUDA devices [[bd3d4b5](https://github.com/miniex/maidenx/commit/bd3d4b5)]
- Implemented tensor offset support for memory-efficient views [[82a2e30](https://github.com/miniex/maidenx/commit/82a2e30)]
- Introduced new padding operations [[5531e67](https://github.com/miniex/maidenx/commit/5531e67)]
- Added operator overriding capabilities [[f1c957b](https://github.com/miniex/maidenx/commit/f1c957b)]
- Added metadata support in tensors [[030b9f8](https://github.com/miniex/maidenx/commit/030b9f8)]

#### Neural Network Components
- Added LeakyReLU, GELU, and ELU activation functions [[06ff9d2](https://github.com/miniex/maidenx/commit/06ff9d2)]

### Enhancements
- Improved performance by replacing RwLock with Arc for buffer management [[af8b5d5](https://github.com/miniex/maidenx/commit/af8b5d5)]
- Updated tensor indexing methods to support offsets [[89a382b](https://github.com/miniex/maidenx/commit/89a382b)]

### Fixes
- Fixed borrowing issues in inplace operations [[af8b5d5](https://github.com/miniex/maidenx/commit/af8b5d5)]

## Version 0.1.0

### Added Features

#### DType Support
- Floating-Point Types: bf16, f16, f32, f64
- Integer Types: u8, u32, i8, i32, i64

#### Neural Network Components
- Modules: Linear, Bilinear, Conv2d layers (CPU and CUDA)
- Activations: ReLU, Sigmoid, Tanh
- Loss Functions: MSE, MAE, Huber
- Optimizers: SGD, Adam

#### Tensor Operations
- Basic Operations: element-wise operations with gradient tracking and broadcasting
- Shape Operations: transpose, reshape, and broadcasting utilities
- Reduction Operations: sum, mean
- Scalar Operations: addition, subtraction, multiplication, division
- Comparison Operations: scalar-based comparisons

### Enhancements
- Gradient Tracking via automatic differentiation
- Full tensor broadcasting support
