# Tensor Utilities

## Device and Type Conversion

These utilities allow you to change a tensor's device or data type.

### with_device / to_device
```rust
pub fn with_device(&mut self, device: Device) -> Result<()>
pub fn to_device(&self, device: Device) -> Result<Self>
```
Changes the device where a tensor is stored.

- **Parameters**:
  - `device`: The target device (CPU, CUDA, MPS)
- **Returns**: 
  - `with_device`: Modifies the tensor in-place and returns Result
  - `to_device`: Returns a new tensor on the specified device
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.to_device(Device::CPU)?;  // Copy to CPU

// In-place version
let mut c = Tensor::new(vec![1.0, 2.0, 3.0])?;
c.with_device(Device::CPU)?;  // Move to CPU in-place
```

### with_dtype / to_dtype
```rust
pub fn with_dtype(&mut self, dtype: DType) -> Result<()>
pub fn to_dtype(&self, dtype: DType) -> Result<Self>
```
Changes the data type of a tensor.

- **Parameters**:
  - `dtype`: The target data type (F32, F64, I32, etc.)
- **Returns**: 
  - `with_dtype`: Modifies the tensor in-place and returns Result
  - `to_dtype`: Returns a new tensor with the specified data type
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.to_dtype(DType::F64)?;  // Convert to 64-bit float

// In-place version
let mut c = Tensor::new(vec![1.0, 2.0, 3.0])?;
c.with_dtype(DType::I32)?;  // Convert to 32-bit int in-place
```

### with_shape / to_shape
```rust
pub fn with_shape(&mut self, shape: &[usize]) -> Result<()>
pub fn to_shape(&self, shape: &[usize]) -> Result<Self>
```
Changes the shape of a tensor without modifying the data.

- **Parameters**:
  - `shape`: The new shape dimensions
- **Returns**: 
  - `with_shape`: Modifies the tensor in-place and returns Result
  - `to_shape`: Returns a new tensor with the specified shape
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
let b = a.to_shape(&[2, 2])?;  // Reshape to 2x2

// In-place version
let mut c = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
c.with_shape(&[2, 2])?;  // Reshape to 2x2 in-place
```

### with_grad
```rust
pub fn with_grad(&mut self) -> Result<()>
```
Enables gradient computation for a tensor.

- **Returns**: Modifies the tensor in-place and returns Result
- **Example**:
```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0])?;
a.with_grad()?;  // Enable gradients
```
