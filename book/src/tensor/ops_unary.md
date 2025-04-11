# Unary Operations

Unary operations in maidenx are operations that take a single tensor as input and produce a single output tensor. These operations apply the specified mathematical function to each element of the input tensor.

## Basic Unary Operations

### neg
```rust
fn neg(&self) -> Result<Tensor>
```
Negates each element in the tensor.

- **Returns**: A new tensor with each element negated
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, -2.0, 3.0])?;
let b = a.neg()?; // [-1.0, 2.0, -3.0]
```

### abs
```rust
fn abs(&self) -> Result<Tensor>
```
Computes the absolute value of each element in the tensor.

- **Returns**: A new tensor with absolute values
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![-1.0, 2.0, -3.0])?;
let b = a.abs()?; // [1.0, 2.0, 3.0]
```

### sign
```rust
fn sign(&self) -> Result<Tensor>
```
Returns the sign of each element in the tensor (-1 for negative, 0 for zero, 1 for positive).

- **Returns**: A new tensor with the sign of each element
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![-2.0, 0.0, 3.0])?;
let b = a.sign()?; // [-1.0, 0.0, 1.0]
```

### square
```rust
fn square(&self) -> Result<Tensor>
```
Squares each element in the tensor.

- **Returns**: A new tensor with each element squared
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.square()?; // [1.0, 4.0, 9.0]
```

### sqrt
```rust
fn sqrt(&self) -> Result<Tensor>
```
Computes the square root of each element in the tensor.

- **Returns**: A new tensor with the square root of each element
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 4.0, 9.0])?;
let b = a.sqrt()?; // [1.0, 2.0, 3.0]
```

## Activation Functions

### relu
```rust
fn relu(&self) -> Result<Tensor>
```
Applies the Rectified Linear Unit function to each element (max(0, x)).

- **Returns**: A new tensor with ReLU applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![-1.0, 0.0, 2.0])?;
let b = a.relu()?; // [0.0, 0.0, 2.0]
```

### sigmoid
```rust
fn sigmoid(&self) -> Result<Tensor>
```
Applies the sigmoid function (1 / (1 + exp(-x))) to each element.

- **Returns**: A new tensor with sigmoid applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0])?;
let b = a.sigmoid()?; // [0.5]
```

### tanh
```rust
fn tanh(&self) -> Result<Tensor>
```
Applies the hyperbolic tangent function to each element.

- **Returns**: A new tensor with tanh applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0])?;
let b = a.tanh()?; // [0.0]
```

### gelu
```rust
fn gelu(&self) -> Result<Tensor>
```
Applies the Gaussian Error Linear Unit function to each element.

- **Returns**: A new tensor with GELU applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, 1.0, -1.0])?;
let b = a.gelu()?; // [0.0, 0.841..., -0.159...]
```

### softplus
```rust
fn softplus(&self) -> Result<Tensor>
```
Applies the softplus function (log(1 + exp(x))) to each element.

- **Returns**: A new tensor with softplus applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, 1.0])?;
let b = a.softplus()?; // [0.693..., 1.313...]
```

## Trigonometric Functions

### sin
```rust
fn sin(&self) -> Result<Tensor>
```
Computes the sine of each element.

- **Returns**: A new tensor with sine applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, std::f32::consts::PI/2.0])?;
let b = a.sin()?; // [0.0, 1.0]
```

### cos
```rust
fn cos(&self) -> Result<Tensor>
```
Computes the cosine of each element.

- **Returns**: A new tensor with cosine applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, std::f32::consts::PI/2.0])?;
let b = a.cos()?; // [1.0, 0.0]
```

### tan
```rust
fn tan(&self) -> Result<Tensor>
```
Computes the tangent of each element.

- **Returns**: A new tensor with tangent applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, std::f32::consts::PI/4.0])?;
let b = a.tan()?; // [0.0, 1.0]
```

## Logarithmic and Exponential Functions

### ln
```rust
fn ln(&self) -> Result<Tensor>
```
Computes the natural logarithm of each element.

- **Returns**: A new tensor with natural log applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, std::f32::consts::E])?;
let b = a.ln()?; // [0.0, 1.0]
```

### log
```rust
fn log(&self) -> Result<Tensor>
```
Alias for `ln()` - computes the natural logarithm.

- **Returns**: A new tensor with natural log applied
- **Supports Autograd**: Yes

### log10
```rust
fn log10(&self) -> Result<Tensor>
```
Computes the base-10 logarithm of each element.

- **Returns**: A new tensor with base-10 log applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 10.0, 100.0])?;
let b = a.log10()?; // [0.0, 1.0, 2.0]
```

### log2
```rust
fn log2(&self) -> Result<Tensor>
```
Computes the base-2 logarithm of each element.

- **Returns**: A new tensor with base-2 log applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 4.0, 8.0])?;
let b = a.log2()?; // [0.0, 1.0, 2.0, 3.0]
```

### exp
```rust
fn exp(&self) -> Result<Tensor>
```
Computes the exponential (e^x) of each element.

- **Returns**: A new tensor with exponential applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, 1.0])?;
let b = a.exp()?; // [1.0, 2.718...]
```

### exp10
```rust
fn exp10(&self) -> Result<Tensor>
```
Computes 10 raised to the power of each element.

- **Returns**: A new tensor with 10^x applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, 1.0, 2.0])?;
let b = a.exp10()?; // [1.0, 10.0, 100.0]
```

### exp2
```rust
fn exp2(&self) -> Result<Tensor>
```
Computes 2 raised to the power of each element.

- **Returns**: A new tensor with 2^x applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![0.0, 1.0, 2.0, 3.0])?;
let b = a.exp2()?; // [1.0, 2.0, 4.0, 8.0]
```

### recip
```rust
fn recip(&self) -> Result<Tensor>
```
Computes the reciprocal (1/x) of each element.

- **Returns**: A new tensor with reciprocal applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 4.0])?;
let b = a.recip()?; // [1.0, 0.5, 0.25]
```

## Logical Operations

### logical_not
```rust
fn logical_not(&self) -> Result<Tensor>
```
Computes the logical NOT of each element.

- **Returns**: A new boolean tensor with values negated
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![true, false])?;
let b = a.logical_not()?; // [false, true]
```

## Operations with Scalar Values

### add_scalar
```rust
fn add_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Adds a scalar value to each element in the tensor.

- **Parameters**: 
  - `scalar`: The scalar value to add
- **Returns**: A new tensor with scalar added to each element
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.add_scalar(5.0)?; // [6.0, 7.0, 8.0]
```

### sub_scalar
```rust
fn sub_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Subtracts a scalar value from each element in the tensor.

- **Parameters**: 
  - `scalar`: The scalar value to subtract
- **Returns**: A new tensor with scalar subtracted from each element
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![6.0, 7.0, 8.0])?;
let b = a.sub_scalar(5.0)?; // [1.0, 2.0, 3.0]
```

### mul_scalar
```rust
fn mul_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Multiplies each element in the tensor by a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to multiply by
- **Returns**: A new tensor with each element multiplied by scalar
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.mul_scalar(2.0)?; // [2.0, 4.0, 6.0]
```

### div_scalar
```rust
fn div_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Divides each element in the tensor by a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to divide by
- **Returns**: A new tensor with each element divided by scalar
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![2.0, 4.0, 6.0])?;
let b = a.div_scalar(2.0)?; // [1.0, 2.0, 3.0]
```

### maximum_scalar
```rust
fn maximum_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Takes the maximum of each element in the tensor and a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new tensor with maximum values
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 3.0, 2.0])?;
let b = a.maximum_scalar(2.0)?; // [2.0, 3.0, 2.0]
```

### minimum_scalar
```rust
fn minimum_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Takes the minimum of each element in the tensor and a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new tensor with minimum values
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 3.0, 2.0])?;
let b = a.minimum_scalar(2.0)?; // [1.0, 2.0, 2.0]
```

### pow
```rust
fn pow(&self, exponent: impl Into<Scalar>) -> Result<Tensor>
```
Raises each element in the tensor to the power of the exponent.

- **Parameters**: 
  - `exponent`: The exponent to raise elements to
- **Returns**: A new tensor with each element raised to the power
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.pow(2.0)?; // [1.0, 4.0, 9.0]
```

### leaky_relu
```rust
fn leaky_relu(&self, negative_slope: impl Into<Scalar>) -> Result<Tensor>
```
Applies the Leaky ReLU function to each element.

- **Parameters**: 
  - `negative_slope`: The slope for negative input values
- **Returns**: A new tensor with Leaky ReLU applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![-2.0, 0.0, 3.0])?;
let b = a.leaky_relu(0.1)?; // [-0.2, 0.0, 3.0]
```

### elu
```rust
fn elu(&self, alpha: impl Into<Scalar>) -> Result<Tensor>
```
Applies the Exponential Linear Unit function to each element.

- **Parameters**: 
  - `alpha`: The alpha parameter for ELU
- **Returns**: A new tensor with ELU applied
- **Supports Autograd**: Yes
- **Example**:
```rust
let a = Tensor::new(vec![-2.0, 0.0, 3.0])?;
let b = a.elu(1.0)?; // [-0.865..., 0.0, 3.0]
```

## Comparison Operations with Scalar

### eq_scalar
```rust
fn eq_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Compares each element for equality with a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new boolean tensor with comparison results
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 2.0])?;
let b = a.eq_scalar(2.0)?; // [false, true, true]
```

### ne_scalar
```rust
fn ne_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Compares each element for inequality with a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new boolean tensor with comparison results
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 2.0])?;
let b = a.ne_scalar(2.0)?; // [true, false, false]
```

### lt_scalar
```rust
fn lt_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Checks if each element is less than a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new boolean tensor with comparison results
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.lt_scalar(2.0)?; // [true, false, false]
```

### le_scalar
```rust
fn le_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Checks if each element is less than or equal to a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new boolean tensor with comparison results
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.le_scalar(2.0)?; // [true, true, false]
```

### gt_scalar
```rust
fn gt_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Checks if each element is greater than a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new boolean tensor with comparison results
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.gt_scalar(2.0)?; // [false, false, true]
```

### ge_scalar
```rust
fn ge_scalar(&self, scalar: impl Into<Scalar>) -> Result<Tensor>
```
Checks if each element is greater than or equal to a scalar value.

- **Parameters**: 
  - `scalar`: The scalar value to compare with
- **Returns**: A new boolean tensor with comparison results
- **Supports Autograd**: No
- **Example**:
```rust
let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
let b = a.ge_scalar(2.0)?; // [false, true, true]
```