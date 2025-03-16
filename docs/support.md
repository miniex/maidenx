# Supported Functions

Blank: Upcoming support
✓: Supported
✗: Not supported

## DType

| Data type | DType |
|---|---|
| 16-bit floating point | `maidenx::bfloat16` |
| 16-bit floating point | `maidenx::float16` or `maidenx::half` |
| 32-bit floating point | `maidenx::float32` |
| 64-bit floating point | `maidenx::float64` |
| boolean | `maidenx::bool` (only support logical operations) |
| 8-bit integer (unsigned) | `maidenx::uint8` |
| 16-bit integer (unsigned) | not supported |
| 32-bit integer (unsigned) | `maidenx::uint32` |
| 64-bit integer (unsigned) | not supported |
| 8-bit integer (signed) | `maidenx::int8` |
| 16-bit integer (signed) | not supported |
| 32-bit integer (signed) | `maidenx::int32` |
| 64-bit integer (signed) | `maidenx::int64` |

<br/>

## Tensor

### Tensor Operations

| Category | Name | Autograd | CPU | CUDA |
|---|---|---|---|---|
| **Binary Operations** |
|            | `add`              | ✓ | ✓ | ✓ |
|            | `sub`              | ✓ | ✓ | ✓ |
|            | `mul`              | ✓ | ✓ | ✓ |
|            | `div`              | ✓ | ✓ | ✓ |
| logical operations |
|            | `logical_and`      | ✗ | ✓ | ✓ |
|            | `logical_or`       | ✗ | ✓ | ✓ |
|            | `logical_xor`      | ✗ | ✓ | ✓ |
| comparison operations |
|            | `eq`               | ✗ | ✓ | ✓ |
|            | `ne`               | ✗ | ✓ | ✓ |
|            | `lt`               | ✗ | ✓ | ✓ |
|            | `le`               | ✗ | ✓ | ✓ |
|            | `gt`               | ✗ | ✓ | ✓ |
|            | `ge`               | ✗ | ✓ | ✓ |
| matrix multiplication |
|            | `matmul`           | ✓ | ✓ | ✓ |
| in-place |
|            | `add_`             | ✗ | ✓ | ✓ |
|            | `sub_`             | ✗ | ✓ | ✓ |
|            | `mul_`             | ✗ | ✓ | ✓ |
|            | `div_`             | ✗ | ✓ | ✓ |
| **Unary Operations** |
|            | `neg`              | ✓ | ✓ | ✓ |
|            | `abs`              | ✓ | ✓ | ✓ |
|            | `sign`             | ✗ | ✓ | ✓ |
|            | `square`           | ✓ | ✓ | ✓ |
|            | `sqrt`             | ✓ | ✓ | ✓ |
|            | `relu`             | ✓ | ✓ | ✓ |
|            | `sigmoid`          | ✓ | ✓ | ✓ |
|            | `tanh`             | ✓ | ✓ | ✓ |
|            | `gelu`             | ✓ | ✓ | ✓ |
|            | `sin`              | ✓ | ✓ | ✓ |
|            | `cos`              | ✓ | ✓ | ✓ |
|            | `tan`              | ✓ | ✓ | ✓ |
|            | `ln`               | ✓ | ✓ | ✓ |
|            | `log10`            | ✓ | ✓ | ✓ |
|            | `log2`             | ✓ | ✓ | ✓ |
|            | `exp`              | ✓ | ✓ | ✓ |
|            | `exp10`            | ✓ | ✓ | ✓ |
|            | `exp2`             | ✓ | ✓ | ✓ |
|            | `softplus`         | ✓ | ✓ | ✓ |
|            | `recip`            | ✓ | ✓ | ✓ |
| logical operations |
|            | `logical_not`      | ✗ | ✓ | ✓ |
| with constant |
|            | `add_scalar`       | ✓ | ✓ | ✓ |
|            | `sub_scalar`       | ✓ | ✓ | ✓ |
|            | `mul_scalar`       | ✓ | ✓ | ✓ |
|            | `div_scalar`       | ✓ | ✓ | ✓ |
|            | `pow`              | ✓ | ✓ | ✓ |
|            | `leaky_relu`       | ✓ | ✓ | ✓ |
|            | `elu`              | ✓ | ✓ | ✓ |
| comparison operations |
|            | `eq_scalar`        | ✗ | ✓ | ✓ |
|            | `ne_scalar`        | ✗ | ✓ | ✓ |
|            | `lt_scalar`        | ✗ | ✓ | ✓ |
|            | `le_scalar`        | ✗ | ✓ | ✓ |
|            | `gt_scalar`        | ✗ | ✓ | ✓ |
|            | `ge_scalar`        | ✗ | ✓ | ✓ |
| **Reduction Operations** |
|            | `sum`              | ✓ | ✓ | ✓ |
|            | `sum_all`          | ✓ | ✓ | ✓ |
|            | `sum_to_shape`     | ✓ | ✓ | ✓ |
|            | `mean`             | ✓ | ✓ | ✓ |
|            | `mean_all`         | ✓ | ✓ | ✓ |
|            | `fold`             | ✓ | ✓ | ✓ |
| **Transform Operations** |
| view operations |
|            | `view`             | ✓ | ✓ | ✓ |
|            | `squeeze`          | ✓ | ✓ | ✓ |
|            | `squeeze_all`      | ✓ | ✓ | ✓ |
|            | `unsqueeze`        | ✓ | ✓ | ✓ |
|            | `transpose`        | ✓ | ✓ | ✓ |
|            | `slice`            | ✓ | ✓ | ✓ |
|            | `unfold`           | ✓ | ✓ | ✓ |
| reshape operations |
|            | `reshape`          | ✓ | ✓ | ✓ |
| broadcast operations |
|            | `broadcast`        | ✓ | ✓ | ✓ |
|            | `broadcast_like`   | ✓ | ✓ | ✓ |
|            | `broadcast_left`   | ✓ | ✓ | ✓ |
| **Padding Operations** |
|            | `pad`                  | ✓ | ✓ | ✓ |
|            | `pad_with_constant`    | ✓ | ✓ | ✓ |
|            | `pad_with_reflection`  | ✓ | ✓ | ✓ |
|            | `pad_with_replication` | ✓ | ✓ | ✓ |


<br/>

### Tensor Functions

| Category | Name | CPU | CUDA |
|---|---|---|---|
| **Util Functions** |
|            | `is_contiguous`    | ✓ | ✓ |
|            | `contiguous`       | ✓ | ✓ |
|            | `to_flatten_vec`   | ✓ | ✓ |
|            | `get`              | ✓ | ✓ |
|            | `set`              | ✓ | ✓ |
|            | `select`           | ✓ | ✓ |

<br/>

## Neural Networks

### Layers

| Category | Name | CPU | CUDA |
|---|---|---|---|
| **Linear Layers** |
|            | `Linear`           | ✓ | ✓ |
| **Convolution Layers** |
|            | `Conv2d`           | ✓ | ✓ |
| **Normalization Layers** |
|            | `LayerNorm`        | ✓ | ✓ |
| **Dropout Layers** |
|            | `Dropout`          | ✓ | ✓ |
| **Activation Layers** |
|            | `Softmax`          | ✓ | ✓ | 
| tensor ops aliases |
|            | `ReLU`             | ✓ | ✓ |
|            | `Sigmoid`          | ✓ | ✓ |
|            | `Tanh`             | ✓ | ✓ |
|            | `LeakyReLU`        | ✓ | ✓ | 
|            | `GELU`             | ✓ | ✓ | 
|            | `ELU`              | ✓ | ✓ | 

<br/>

### Loss Layers

| Category | Name | CPU | CUDA |
|---|---|---|---|
|            | `Huber`            | ✓ | ✓ |
|            | `MAE`              | ✓ | ✓ |
|            | `MSE`              | ✓ | ✓ |

<br/>

### Optimizer

| Category | Name | CPU | CUDA |
|---|---|---|---|
|            | `Adam`             | ✓ | ✓ |
|            | `SGD`              | ✓ | ✓ |
