# Supported

## Device

- CPU
- CUDA
- MPS
- Vulkan (planned)

## DType

| Data type | DType | Notes |
|---|---|---|
| 16-bit floating point     | `maidenx::bfloat16`   |                      |
| 16-bit floating point     | `maidenx::float16`    |                      |
| 32-bit floating point     | `maidenx::float32`    |                      |
| 64-bit floating point     | `maidenx::float64`    | Not supported on MPS |
| boolean                   | `maidenx::bool`       |                      |
| 8-bit integer (unsigned)  | `maidenx::uint8`      |                      |
| 16-bit integer (unsigned) | `maidenx::uint16`     |                      |
| 32-bit integer (unsigned) | `maidenx::uint32`     |                      |
| 64-bit integer (unsigned) | `maidenx::uint64`     | Not supported on MPS |
| 8-bit integer (signed)    | `maidenx::int8`       |                      |
| 16-bit integer (signed)   | `maidenx::int16`      |                      |
| 32-bit integer (signed)   | `maidenx::int32`      |                      |
| 64-bit integer (signed)   | `maidenx::int64`      | Not supported on MPS |

> [!NOTE]
> The boolean type in the MaidenX framework is promoted based on the type of operation:
> 
> - For logical operations: Remains as maidenx::bool
> - For arithmetic operations: Promoted to maidenx::uint8 (8-bit unsigned integer)
> - For operations involving floating-point numbers: Promoted to maidenx::float32 (32-bit floating point)
>
> This conversion happens automatically in the framework to ensure type compatibility during different operations, as boolean values (true/false) need to be represented as numeric values (1/0) when used in mathematical contexts.

> [!IMPORTANT]
> Automatic differentiation in MaidenX only supports floating-point types.
> Integer and boolean types cannot be used with gradient computation.

> [!IMPORTANT]
> MPS (Metal Performance Shaders) does not support 64-bit data types (uint64, int64, float64).
> When using MPS as your compute device, please use 32-bit or lower precision data types instead.
