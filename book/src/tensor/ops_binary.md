# Binary Operations

Binary operations in maidenx are operations that take two tensors as input and produce a single output tensor. These operations typically apply the specified mathematical operation element-wise between corresponding elements of the input tensors.

## Arithmetic Operations

### add
```rust
fn add(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise addition between two tensors.

- **Parameters**:
  - `rhs`: The tensor to add to the current tensor
- **Returns**: A new tensor containing the sum of the elements
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
let c = a.add(&b)?; // [5.0, 7.0, 9.0]
```

### sub
```rust
fn sub(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise subtraction between two tensors.

- **Parameters**:
  - `rhs`: The tensor to subtract from the current tensor
- **Returns**: A new tensor containing the difference of the elements
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![5.0, 7.0, 9.0])?;
let b = Tensor::new(vec![1.0, 2.0, 3.0])?;
let c = a.sub(&b)?; // [4.0, 5.0, 6.0]
```

### mul
```rust
fn mul(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise multiplication between two tensors.

- **Parameters**:
  - `rhs`: The tensor to multiply with the current tensor
- **Returns**: A new tensor containing the product of the elements
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
let c = a.mul(&b)?; // [4.0, 10.0, 18.0]
```

### div
```rust
fn div(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise division between two tensors.

- **Parameters**:
  - `rhs`: The tensor to divide the current tensor by
- **Returns**: A new tensor containing the quotient of the elements
- **Supports Autograd**: Yes
- **Note**: When dividing integer tensors, the result will be promoted to F32
- **Example**:
```rust
let a = Tensor::new(vec![4.0, 10.0, 18.0])?;
let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
let c = a.div(&b)?; // [1.0, 2.0, 3.0]
```

### maximum
```rust
fn maximum(&self, rhs: &Tensor) -> Result<Tensor>
```
Takes the element-wise maximum of two tensors.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new tensor containing the maximum value at each position
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 5.0, 3.0])?;
let b = Tensor::new(vec![4.0, 2.0, 6.0])?;
let c = a.maximum(&b)?; // [4.0, 5.0, 6.0]
```

### minimum
```rust
fn minimum(&self, rhs: &Tensor) -> Result<Tensor>
```
Takes the element-wise minimum of two tensors.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new tensor containing the minimum value at each position
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 5.0, 3.0])?;
let b = Tensor::new(vec![4.0, 2.0, 6.0])?;
let c = a.minimum(&b)?; // [1.0, 2.0, 3.0]
```

## Logical Operations

### logical_and
```rust
fn logical_and(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise logical AND between two tensors.

- **Parameters**:
  - `rhs`: The tensor to combine with the current tensor
- **Returns**: A new boolean tensor with logical AND applied
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![true, false, true])?;
let b = Tensor::new(vec![true, true, false])?;
let c = a.logical_and(&b)?; // [true, false, false]
```

### logical_or
```rust
fn logical_or(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise logical OR between two tensors.

- **Parameters**:
  - `rhs`: The tensor to combine with the current tensor
- **Returns**: A new boolean tensor with logical OR applied
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![true, false, true])?;
let b = Tensor::new(vec![true, true, false])?;
let c = a.logical_or(&b)?; // [true, true, true]
```

### logical_xor
```rust
fn logical_xor(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs element-wise logical XOR between two tensors.

- **Parameters**:
  - `rhs`: The tensor to combine with the current tensor
- **Returns**: A new boolean tensor with logical XOR applied
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![true, false, true])?;
let b = Tensor::new(vec![true, true, false])?;
let c = a.logical_xor(&b)?; // [false, true, true]
```

## Comparison Operations

### eq
```rust
fn eq(&self, rhs: &Tensor) -> Result<Tensor>
```
Compares two tensors for element-wise equality.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new boolean tensor with true where elements are equal
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![1.0, 5.0, 3.0])?;
let c = a.eq(&b)?; // [true, false, true]
```

### ne
```rust
fn ne(&self, rhs: &Tensor) -> Result<Tensor>
```
Compares two tensors for element-wise inequality.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new boolean tensor with true where elements are not equal
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![1.0, 5.0, 3.0])?;
let c = a.ne(&b)?; // [false, true, false]
```

### lt
```rust
fn lt(&self, rhs: &Tensor) -> Result<Tensor>
```
Compares if elements in the first tensor are less than the corresponding elements in the second tensor.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new boolean tensor with true where elements are less than the corresponding elements
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![2.0, 2.0, 1.0])?;
let c = a.lt(&b)?; // [true, false, false]
```

### le
```rust
fn le(&self, rhs: &Tensor) -> Result<Tensor>
```
Compares if elements in the first tensor are less than or equal to the corresponding elements in the second tensor.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new boolean tensor with true where elements are less than or equal to the corresponding elements
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![2.0, 2.0, 1.0])?;
let c = a.le(&b)?; // [true, true, false]
```

### gt
```rust
fn gt(&self, rhs: &Tensor) -> Result<Tensor>
```
Compares if elements in the first tensor are greater than the corresponding elements in the second tensor.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new boolean tensor with true where elements are greater than the corresponding elements
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![2.0, 2.0, 1.0])?;
let c = a.gt(&b)?; // [false, false, true]
```

### ge
```rust
fn ge(&self, rhs: &Tensor) -> Result<Tensor>
```
Compares if elements in the first tensor are greater than or equal to the corresponding elements in the second tensor.

- **Parameters**:
  - `rhs`: The tensor to compare with the current tensor
- **Returns**: A new boolean tensor with true where elements are greater than or equal to the corresponding elements
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![2.0, 2.0, 1.0])?;
let c = a.ge(&b)?; // [false, true, true]
```

## Matrix Multiplication

### matmul
```rust
fn matmul(&self, rhs: &Tensor) -> Result<Tensor>
```
Performs matrix multiplication between two tensors.

- **Parameters**:
  - `rhs`: The tensor to multiply with the current tensor
- **Returns**: A new tensor containing the result of matrix multiplication
- **Supports Autograd**: Yes
- **Shape Rules**:
  - For 1D tensors (vectors): Returns the dot product as a scalar
  - For 2D tensors (matrices): Standard matrix multiplication (M×K * K×N → M×N)
  - For batched tensors: Applied to the last two dimensions with broadcasting
  - If a tensor has only 1 dimension, it's treated as:
    - 1D * 1D: Both are treated as vectors, resulting in a scalar (dot product)
    - 1D * 2D: The 1D tensor is treated as a 1×K matrix, resulting in a 1×N vector
    - 2D * 1D: The 1D tensor is treated as a K×1 matrix, resulting in a M×1 vector
  - For N-D * M-D tensors: Leading dimensions are broadcast
- **Examples**:
```rust
// Basic matrix multiplication (2D * 2D)
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0])?.reshape(&[2, 2])?;
let c = a.matmul(&b)?; // [[19.0, 22.0], [43.0, 50.0]]

// Vector-vector multiplication (1D * 1D)
let v1 = Tensor::new(vec![1.0, 2.0, 3.0])?;
let v2 = Tensor::new(vec![4.0, 5.0, 6.0])?;
let dot = v1.matmul(&v2)?; // [32.0] (dot product: 1*4 + 2*5 + 3*6)

// Matrix-vector multiplication (2D * 1D)
let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
let v = Tensor::new(vec![5.0, 6.0])?;
let mv = m.matmul(&v)?; // [17.0, 39.0]

// Batched matrix multiplication with broadcasting
let batch1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?.reshape(&[2, 2, 2])?;
let batch2 = Tensor::new(vec![9.0, 10.0, 11.0, 12.0])?.reshape(&[1, 2, 2])?;
let result = batch1.matmul(&batch2)?; // [[[29, 32], [67, 74]], [[65, 72], [159, 176]]]
```

## In-place Operations

### add_
```rust
fn add_(&mut self, rhs: &Tensor) -> Result<()>
```
Performs in-place element-wise addition.

- **Parameters**:
  - `rhs`: The tensor to add to the current tensor
- **Returns**: Result indicating success or failure
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
a.add_(&b)?; // a becomes [5.0, 7.0, 9.0]
```

### sub_
```rust
fn sub_(&mut self, rhs: &Tensor) -> Result<()>
```
Performs in-place element-wise subtraction.

- **Parameters**:
  - `rhs`: The tensor to subtract from the current tensor
- **Returns**: Result indicating success or failure
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::new(vec![5.0, 7.0, 9.0])?;
let b = Tensor::new(vec![1.0, 2.0, 3.0])?;
a.sub_(&b)?; // a becomes [4.0, 5.0, 6.0]
```

### mul_
```rust
fn mul_(&mut self, rhs: &Tensor) -> Result<()>
```
Performs in-place element-wise multiplication.

- **Parameters**:
  - `rhs`: The tensor to multiply with the current tensor
- **Returns**: Result indicating success or failure
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
a.mul_(&b)?; // a becomes [4.0, 10.0, 18.0]
```

### div_
```rust
fn div_(&mut self, rhs: &Tensor) -> Result<()>
```
Performs in-place element-wise division.

- **Parameters**:
  - `rhs`: The tensor to divide the current tensor by
- **Returns**: Result indicating success or failure
- **Supports Autograd**: No
- **Example**:
```rust
let mut a = Tensor::new(vec![4.0, 10.0, 18.0])?;
let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
a.div_(&b)?; // a becomes [1.0, 2.0, 3.0]
```

## Broadcasting

All binary operations in maidenx support broadcasting, allowing operations between tensors of different shapes. Broadcasting automatically expands dimensions of the smaller tensor to match the larger one where possible, following these rules:

1. Trailing dimensions are aligned
2. Each dimension either has the same size or one of them is 1 (which gets expanded)
3. If a tensor has fewer dimensions, it is padded with dimensions of size 1 at the beginning

This allows for flexible and concise operations without unnecessary tensor reshaping or copying.