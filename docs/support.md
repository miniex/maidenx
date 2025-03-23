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
| boolean | `maidenx::bool` |
| 8-bit integer (unsigned) | `maidenx::uint8` |
| 16-bit integer (unsigned) | `maidenx::uint16` |
| 32-bit integer (unsigned) | `maidenx::uint32` |
| 64-bit integer (unsigned) | `maidenx::uint64` |
| 8-bit integer (signed) | `maidenx::int8` |
| 16-bit integer (signed) | `maidenx::int16` |
| 32-bit integer (signed) | `maidenx::int32` |
| 64-bit integer (signed) | `maidenx::int64` |

> [!NOTE]
> The boolean type in the MaidenX framework is promoted based on the type of operation:
> 
> - For logical operations: Remains as maidenx::bool
> - For arithmetic operations: Promoted to maidenx::uint8 (8-bit unsigned integer)
> - For operations involving floating-point numbers: Promoted to maidenx::float32 (32-bit floating point)
>
> This conversion happens automatically in the framework to ensure type compatibility during different operations, as boolean values (true/false) need to be represented as numeric values (1/0) when used in mathematical contexts.

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
|            | `maximum`          | ✓ | ✓ | ✓ |
|            | `minimum`          | ✓ | ✓ | ✓ |
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
|            | `log` (ln alias)   | ✓ | ✓ | ✓ |
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
|            | `maximum_scalar`   | ✓ | ✓ | ✓ |
|            | `minimum_scalar`   | ✓ | ✓ | ✓ |
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
|            | `max`              | ✓ | ✓ | ✓ | 
|            | `max_all`          | ✓ | ✓ | ✓ | 
|            | `min`              | ✓ | ✓ | ✓ | 
|            | `min_all`          | ✓ | ✓ | ✓ | 
|            | `norm`             | ✓ | ✓ | ✓ | 
|            | `norm_all`         | ✓ | ✓ | ✓ | 
|            | `std`              | ✓ | ✓ | ✓ | 
|            | `var`              | ✓ | ✓ | ✓ | 
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
|            | `pad`                                | ✓ | ✓ | ✓ |
|            | `pad_with_constant`                  | ✓ | ✓ | ✓ |
|            | `pad_with_reflection`                | ✓ | ✓ | ✓ |
|            | `pad_with_replication`               | ✓ | ✓ | ✓ |
| **Indexing Operations** |
|            | `index` (dim 0 index_select alias)   | ✓ | ✓ | ✓ |
|            | `index_add_`       | ✗ | ✓ | ✓ |
|            | `index_select`     | ✓ | ✓ | ✓ |
|            | `index_put_`       | ✗ | ✓ | ✓ |
|            | `gather`           | ✓ | ✓ | ✓ |
|            | `scatter_add_`     | ✗ | ✓ | ✓ |
|            | `bincount`         | ✗ | ✓ | ✓ |
| **NN Layer Aliases** |
| activation |
|            | `softmax`          | ✓ | ✓ | ✓ | 


<br/>

### Tensor Functions

| Category | Name | CPU | CUDA |
|---|---|---|---|
| **Util Functions** |
|            | `is_contiguous`    | ✓ | ✓ |
|            | `contiguous`       | ✓ | ✓ |
|            | `to_flatten_vec`   | ✓ | ✓ |
|            | `any`              | ✓ | ✓ |
|            | `get`              | ✓ | ✓ |
|            | `set`              | ✓ | ✓ |
|            | `select`           | ✓ | ✓ |
|            | `item`             | ✓ | ✓ |

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
| **Embedding Layers** |
|            | `Embedding`        | ✓ | ✓ |
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
|            | `CrossEntropyLoss` | ✓ | ✓ |

<br/>

### Optimizer

| Category | Name | CPU | CUDA |
|---|---|---|---|
|            | `Adam`             | ✓ | ✓ |
|            | `SGD`              | ✓ | ✓ |
