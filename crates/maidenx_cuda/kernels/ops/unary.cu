#include "../cuda_utils.cuh"
#include <cstdint>
#include <stdint.h>

template <typename T> __device__ T maximum(T x, T y) { return (x > y) ? x : y; }

template <typename T> __device__ T minimum(T x, T y) { return (x < y) ? x : y; }

#define UNARY_OP_OUTPUT(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)              \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims, const size_t *metadata,     \
      const IN_TYPENAME *input, OUT_TYPENAME *output) {                        \
    const size_t *dims = metadata;                                             \
    const size_t *strides = metadata + num_dims;                               \
    const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;           \
                                                                               \
    if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {       \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        IN_TYPENAME x = input ? input[offset + i] : (IN_TYPENAME)output[i];    \
        output[i] = FUNC;                                                      \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned strided_i =                                                   \
            offset + get_strided_index(i, num_dims, dims, strides);            \
        IN_TYPENAME x = input ? input[strided_i] : (IN_TYPENAME)output[i];     \
        output[i] = FUNC;                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els, size_t num_dims, const size_t *metadata,                 \
      const IN_TYPENAME *input, OUT_TYPENAME *output) {                        \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(num_els, num_dims,        \
                                                     metadata, input, output); \
  }

#define UNARY_OP(TYPENAME, FN_NAME, FUNC)                                      \
  UNARY_OP_OUTPUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

#define UNARY_OP_WITH_CONSTANT(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)       \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims, const size_t *metadata,     \
      const IN_TYPENAME *input, const IN_TYPENAME constant,                    \
      OUT_TYPENAME *output) {                                                  \
    const size_t *dims = metadata;                                             \
    const size_t *strides = metadata + num_dims;                               \
    const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;           \
                                                                               \
    if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {       \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        IN_TYPENAME x;                                                         \
        if (input) {                                                           \
          x = input[offset + i];                                               \
        } else {                                                               \
          x = output[i];                                                       \
        }                                                                      \
        output[i] = FUNC;                                                      \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned strided_i =                                                   \
            offset + get_strided_index(i, num_dims, dims, strides);            \
        IN_TYPENAME x;                                                         \
        if (input) {                                                           \
          x = input[strided_i];                                                \
        } else {                                                               \
          x = output[i];                                                       \
        }                                                                      \
        output[i] = FUNC;                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els, size_t num_dims, const size_t *metadata,                 \
      const IN_TYPENAME *input, const IN_TYPENAME constant,                    \
      OUT_TYPENAME *output) {                                                  \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els, num_dims, metadata, input, constant, output);                 \
  }

UNARY_OP(float, neg_f32, -x);
UNARY_OP(double, neg_f64, -x);
UNARY_OP(bool, neg_bool, !x);
UNARY_OP(int8_t, neg_i8, -x);
UNARY_OP(int32_t, neg_i32, -x);
UNARY_OP(int64_t, neg_i64, -x);

UNARY_OP(float, abs_f32, fabsf(x));
UNARY_OP(double, abs_f64, fabs(x));
UNARY_OP(bool, abs_bool, x);
UNARY_OP(int8_t, abs_i8, abs(x));
UNARY_OP(int32_t, abs_i32, abs(x));
UNARY_OP(int64_t, abs_i64, abs(x));

UNARY_OP(float, sign_f32, (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f));
UNARY_OP(double, sign_f64, (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0));
UNARY_OP(bool, sign_bool, x ? 1 : 0);
UNARY_OP(uint8_t, sign_u8, (x > 0) ? 1 : 0);
UNARY_OP(uint32_t, sign_u32, (x > 0) ? 1 : 0);
UNARY_OP(int8_t, sign_i8, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int32_t, sign_i32, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int64_t, sign_i64, (x > 0) ? 1 : ((x < 0) ? -1 : 0));

UNARY_OP(float, square_f32, x *x);
UNARY_OP(double, square_f64, x *x);
UNARY_OP(bool, square_bool, x);
UNARY_OP(uint8_t, square_u8, x *x);
UNARY_OP(uint32_t, square_u32, x *x);
UNARY_OP(int8_t, square_i8, x *x);
UNARY_OP(int32_t, square_i32, x *x);
UNARY_OP(int64_t, square_i64, x *x);

UNARY_OP(float, sqrt_f32, sqrtf(x));
UNARY_OP(double, sqrt_f64, sqrt(x));
UNARY_OP(bool, sqrt_bool, x);
UNARY_OP(uint8_t, sqrt_u8, __float2int_rn(sqrtf(__int2float_rn(x))));
UNARY_OP(uint32_t, sqrt_u32, __float2int_rn(sqrtf(__int2float_rn(x))));
UNARY_OP(int8_t, sqrt_i8, __float2int_rn(sqrtf(__int2float_rn(abs(x)))));
UNARY_OP(int32_t, sqrt_i32, __float2int_rn(sqrtf(__int2float_rn(abs(x)))));
UNARY_OP(int64_t, sqrt_i64, __double2ll_rn(sqrt(__ll2double_rn(abs(x)))));

UNARY_OP(float, relu_f32, x > 0 ? x : 0);
UNARY_OP(double, relu_f64, x > 0 ? x : 0);
UNARY_OP(bool, relu_bool, x);

UNARY_OP(float, sigmoid_f32, 1.0f / (1.0f + expf(-x)));
UNARY_OP(double, sigmoid_f64, 1.0 / (1.0 + exp(-x)));
UNARY_OP(bool, sigmoid_bool, x);

UNARY_OP(float, tanh_f32, tanhf(x));
UNARY_OP(double, tanh_f64, tanh(x));
UNARY_OP(bool, tanh_bool, x);

UNARY_OP(float, gelu_f32,
         0.5f * x *
             (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))));
UNARY_OP(double, gelu_f64,
         0.5 * x *
             (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))));

UNARY_OP(float, sin_f32, sinf(x));
UNARY_OP(double, sin_f64, sin(x));

UNARY_OP(float, cos_f32, cosf(x));
UNARY_OP(double, cos_f64, cos(x));

UNARY_OP(float, tan_f32, tanf(x));
UNARY_OP(double, tan_f64, tan(x));

UNARY_OP(float, ln_f32, logf(x));
UNARY_OP(double, ln_f64, log(x));

UNARY_OP(float, log10_f32, log10f(x));
UNARY_OP(double, log10_f64, log10(x));

UNARY_OP(float, log2_f32, log2f(x));
UNARY_OP(double, log2_f64, log2(x));

UNARY_OP(float, exp_f32, expf(x));
UNARY_OP(double, exp_f64, exp(x));

UNARY_OP(float, exp10_f32, exp10f(x));
UNARY_OP(double, exp10_f64, exp10(x));

UNARY_OP(float, exp2_f32, exp2f(x));
UNARY_OP(double, exp2_f64, exp2(x));

UNARY_OP(float, softplus_f32, logf(1.0f + expf(x)));
UNARY_OP(double, softplus_f64, log(1.0 + exp(x)));

UNARY_OP(float, recip_f32, 1.0f / x);
UNARY_OP(double, recip_f64, 1.0 / x);

UNARY_OP_OUTPUT(float, bool, logical_not_f32, x == 0.0f);
UNARY_OP_OUTPUT(double, bool, logical_not_f64, x == 0.0f);
UNARY_OP_OUTPUT(bool, bool, logical_not_bool, !x);
UNARY_OP_OUTPUT(uint8_t, bool, logical_not_u8, x == 0u);
UNARY_OP_OUTPUT(uint32_t, bool, logical_not_u32, x == 0u);
UNARY_OP_OUTPUT(int8_t, bool, logical_not_i8, x == 0);
UNARY_OP_OUTPUT(int32_t, bool, logical_not_i32, x == 0);
UNARY_OP_OUTPUT(int64_t, bool, logical_not_i64, x == 0);

// __half
UNARY_OP(__half, neg_f16, -x);
UNARY_OP(__half, abs_f16, __habs(x));
UNARY_OP(__half, sign_f16,
         __half((x > __half(0)) ? 1.0f : ((x < __half(0)) ? -1.0f : 0.0f)));
UNARY_OP(__half, square_f16, x *x);
UNARY_OP(__half, sqrt_f16, hsqrt(x));
UNARY_OP(__half, relu_f16, x > __half(0) ? x : __half(0));
UNARY_OP(__half, sigmoid_f16,
         __float2half(1.0f / (1.0f + expf(-__half2float(x)))));
UNARY_OP(__half, tanh_f16, __float2half(tanhf(__half2float(x))));
UNARY_OP(__half, gelu_f16,
         __hmul(__hmul(__float2half(0.5f), x),
                __hadd(__float2half(1.0f),
                       __float2half(tanhf(0.7978845608028654f *
                                          (__half2float(x) +
                                           0.044715f * __half2float(x) *
                                               __half2float(x) *
                                               __half2float(x)))))));
UNARY_OP(__half, sin_f16, __float2half(sinf(__half2float(x))));
UNARY_OP(__half, cos_f16, __float2half(cosf(__half2float(x))));
UNARY_OP(__half, tan_f16, __float2half(tanf(__half2float(x))));
UNARY_OP(__half, ln_f16, __float2half(logf(__half2float(x))));
UNARY_OP(__half, log10_f16, __float2half(log10f(__half2float(x))));
UNARY_OP(__half, log2_f16, __float2half(log2f(__half2float(x))));
UNARY_OP(__half, exp_f16, __float2half(expf(__half2float(x))));
UNARY_OP(__half, exp10_f16, __float2half(exp10f(__half2float(x))));
UNARY_OP(__half, exp2_f16, __float2half(exp2f(__half2float(x))));
UNARY_OP(__half, softplus_f16,
         __float2half(logf(1.0f + expf(__half2float(x)))));
UNARY_OP(__half, recip_f16, __float2half(1.0f / __half2float(x)));

UNARY_OP_OUTPUT(__half, bool, logical_not_f16, x == __half(0.0f));

// __nv_bfloat16
UNARY_OP(__nv_bfloat16, neg_bf16, -x);
UNARY_OP(__nv_bfloat16, abs_bf16, __habs(x));
UNARY_OP(__nv_bfloat16, sign_bf16,
         __nv_bfloat16((x > __nv_bfloat16(0))
                           ? 1.0f
                           : ((x < __nv_bfloat16(0)) ? -1.0f : 0.0f)));
UNARY_OP(__nv_bfloat16, square_bf16, x *x);
UNARY_OP(__nv_bfloat16, sqrt_bf16,
         __float2bfloat16(sqrtf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, relu_bf16, x > __nv_bfloat16(0) ? x : __nv_bfloat16(0));
UNARY_OP(__nv_bfloat16, sigmoid_bf16,
         __float2bfloat16(1.0f / (1.0f + expf(-__bfloat162float(x)))));
UNARY_OP(__nv_bfloat16, tanh_bf16,
         __float2bfloat16(tanhf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, gelu_bf16,
         __hmul(__hmul(__float2bfloat16(0.5f), x),
                __hadd(__float2bfloat16(1.0f),
                       __float2bfloat16(tanhf(0.7978845608028654f *
                                              (__bfloat162float(x) +
                                               0.044715f * __bfloat162float(x) *
                                                   __bfloat162float(x) *
                                                   __bfloat162float(x)))))));
UNARY_OP(__nv_bfloat16, sin_bf16, __float2bfloat16(sinf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, cos_bf16, __float2bfloat16(cosf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, tan_bf16, __float2bfloat16(tanf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, ln_bf16, __float2bfloat16(logf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, log10_bf16,
         __float2bfloat16(log10f(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, log2_bf16,
         __float2bfloat16(log2f(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, exp_bf16, __float2bfloat16(expf(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, exp10_bf16,
         __float2bfloat16(exp10f(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, exp2_bf16,
         __float2bfloat16(exp2f(__bfloat162float(x))));
UNARY_OP(__nv_bfloat16, softplus_bf16,
         __float2bfloat16(logf(1.0f + expf(__bfloat162float(x)))));
UNARY_OP(__nv_bfloat16, recip_bf16,
         __float2bfloat16(1.0f / __bfloat162float(x)));

UNARY_OP_OUTPUT(__nv_bfloat16, bool, logical_not_bf16,
                x == __nv_bfloat16(0.0f));

// ==== with constant ====

UNARY_OP_WITH_CONSTANT(float, float, add_scalar_f32, x + constant);
UNARY_OP_WITH_CONSTANT(float, float, sub_scalar_f32, x - constant);
UNARY_OP_WITH_CONSTANT(float, float, mul_scalar_f32, x *constant);
UNARY_OP_WITH_CONSTANT(float, float, div_scalar_f32, x / constant);
UNARY_OP_WITH_CONSTANT(double, double, add_scalar_f64, x + constant);
UNARY_OP_WITH_CONSTANT(double, double, sub_scalar_f64, x - constant);
UNARY_OP_WITH_CONSTANT(double, double, mul_scalar_f64, x *constant);
UNARY_OP_WITH_CONSTANT(double, double, div_scalar_f64, x / constant);
UNARY_OP_WITH_CONSTANT(bool, bool, add_scalar_bool, x || constant);
UNARY_OP_WITH_CONSTANT(bool, bool, sub_scalar_bool, x ^ constant);
UNARY_OP_WITH_CONSTANT(bool, bool, mul_scalar_bool, x &&constant);
UNARY_OP_WITH_CONSTANT(bool, bool, div_scalar_bool, x &&constant);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, add_scalar_u8, x + constant);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, sub_scalar_u8, x - constant);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, mul_scalar_u8, x *constant);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, div_scalar_u8, x / constant);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, add_scalar_u32, x + constant);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, sub_scalar_u32, x - constant);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, mul_scalar_u32, x *constant);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, div_scalar_u32, x / constant);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, add_scalar_i8, x + constant);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, sub_scalar_i8, x - constant);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, mul_scalar_i8, x *constant);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, div_scalar_i8, x / constant);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, add_scalar_i32, x + constant);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, sub_scalar_i32, x - constant);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, mul_scalar_i32, x *constant);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, div_scalar_i32, x / constant);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, add_scalar_i64, x + constant);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, sub_scalar_i64, x - constant);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, mul_scalar_i64, x *constant);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, div_scalar_i64, x / constant);

UNARY_OP_WITH_CONSTANT(float, float, maximum_scalar_f32, maximum(x, constant));
UNARY_OP_WITH_CONSTANT(double, double, maximum_scalar_f64,
                       maximum(x, constant));
UNARY_OP_WITH_CONSTANT(bool, bool, maximum_scalar_bool, x || constant);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, maximum_scalar_u8,
                       maximum(x, constant));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, maximum_scalar_u32,
                       maximum(x, constant));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, maximum_scalar_i8, maximum(x, constant));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, maximum_scalar_i32,
                       maximum(x, constant));
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, maximum_scalar_i64,
                       maximum(x, constant));

UNARY_OP_WITH_CONSTANT(float, float, minimum_scalar_f32, minimum(x, constant));
UNARY_OP_WITH_CONSTANT(double, double, minimum_scalar_f64,
                       minimum(x, constant));
UNARY_OP_WITH_CONSTANT(bool, bool, minimum_scalar_bool, x &&constant);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, minimum_scalar_u8,
                       minimum(x, constant));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, minimum_scalar_u32,
                       minimum(x, constant));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, minimum_scalar_i8, minimum(x, constant));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, minimum_scalar_i32,
                       minimum(x, constant));
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, minimum_scalar_i64,
                       minimum(x, constant));

UNARY_OP_WITH_CONSTANT(float, float, pow_f32, powf(x, constant));
UNARY_OP_WITH_CONSTANT(double, double, pow_f64, pow(x, constant));
UNARY_OP_WITH_CONSTANT(bool, bool, pow_bool, x && (constant != 0));
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, pow_u8, pow(x, constant));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, pow_u32, pow(x, constant));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, pow_i8, pow(x, constant));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, pow_i32, pow(x, constant));
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, pow_i64, pow(x, constant));
UNARY_OP_WITH_CONSTANT(float, float, leaky_relu_f32, x > 0 ? x : constant * x);
UNARY_OP_WITH_CONSTANT(double, double, leaky_relu_f64,
                       x > 0 ? x : constant * x);
UNARY_OP_WITH_CONSTANT(float, float, elu_f32,
                       x > 0 ? x : constant * (expf(x) - 1.0f));
UNARY_OP_WITH_CONSTANT(double, double, elu_f64,
                       x > 0 ? x : constant * (exp(x) - 1.0));

UNARY_OP_WITH_CONSTANT(float, bool, eq_scalar_f32, x == constant);
UNARY_OP_WITH_CONSTANT(float, bool, ne_scalar_f32, x != constant);
UNARY_OP_WITH_CONSTANT(float, bool, lt_scalar_f32, x < constant);
UNARY_OP_WITH_CONSTANT(float, bool, le_scalar_f32, x <= constant);
UNARY_OP_WITH_CONSTANT(float, bool, gt_scalar_f32, x > constant);
UNARY_OP_WITH_CONSTANT(float, bool, ge_scalar_f32, x >= constant);
UNARY_OP_WITH_CONSTANT(double, bool, eq_scalar_f64, x == constant);
UNARY_OP_WITH_CONSTANT(double, bool, ne_scalar_f64, x != constant);
UNARY_OP_WITH_CONSTANT(double, bool, lt_scalar_f64, x < constant);
UNARY_OP_WITH_CONSTANT(double, bool, le_scalar_f64, x <= constant);
UNARY_OP_WITH_CONSTANT(double, bool, gt_scalar_f64, x > constant);
UNARY_OP_WITH_CONSTANT(double, bool, ge_scalar_f64, x >= constant);
UNARY_OP_WITH_CONSTANT(bool, bool, eq_scalar_bool, x == constant);
UNARY_OP_WITH_CONSTANT(bool, bool, ne_scalar_bool, x != constant);
UNARY_OP_WITH_CONSTANT(bool, bool, lt_scalar_bool, !x && constant);
UNARY_OP_WITH_CONSTANT(bool, bool, le_scalar_bool, !x || constant);
UNARY_OP_WITH_CONSTANT(bool, bool, gt_scalar_bool, x && !constant);
UNARY_OP_WITH_CONSTANT(bool, bool, ge_scalar_bool, x || !constant);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, eq_scalar_u8, x == constant);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, ne_scalar_u8, x != constant);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, lt_scalar_u8, x < constant);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, le_scalar_u8, x <= constant);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, gt_scalar_u8, x > constant);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, ge_scalar_u8, x >= constant);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, eq_scalar_u32, x == constant);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, ne_scalar_u32, x != constant);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, lt_scalar_u32, x < constant);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, le_scalar_u32, x <= constant);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, gt_scalar_u32, x > constant);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, ge_scalar_u32, x >= constant);
UNARY_OP_WITH_CONSTANT(int8_t, bool, eq_scalar_i8, x == constant);
UNARY_OP_WITH_CONSTANT(int8_t, bool, ne_scalar_i8, x != constant);
UNARY_OP_WITH_CONSTANT(int8_t, bool, lt_scalar_i8, x < constant);
UNARY_OP_WITH_CONSTANT(int8_t, bool, le_scalar_i8, x <= constant);
UNARY_OP_WITH_CONSTANT(int8_t, bool, gt_scalar_i8, x > constant);
UNARY_OP_WITH_CONSTANT(int8_t, bool, ge_scalar_i8, x >= constant);
UNARY_OP_WITH_CONSTANT(int32_t, bool, eq_scalar_i32, x == constant);
UNARY_OP_WITH_CONSTANT(int32_t, bool, ne_scalar_i32, x != constant);
UNARY_OP_WITH_CONSTANT(int32_t, bool, lt_scalar_i32, x < constant);
UNARY_OP_WITH_CONSTANT(int32_t, bool, le_scalar_i32, x <= constant);
UNARY_OP_WITH_CONSTANT(int32_t, bool, gt_scalar_i32, x > constant);
UNARY_OP_WITH_CONSTANT(int32_t, bool, ge_scalar_i32, x >= constant);
UNARY_OP_WITH_CONSTANT(int64_t, bool, eq_scalar_i64, x == constant);
UNARY_OP_WITH_CONSTANT(int64_t, bool, ne_scalar_i64, x != constant);
UNARY_OP_WITH_CONSTANT(int64_t, bool, lt_scalar_i64, x < constant);
UNARY_OP_WITH_CONSTANT(int64_t, bool, le_scalar_i64, x <= constant);
UNARY_OP_WITH_CONSTANT(int64_t, bool, gt_scalar_i64, x > constant);
UNARY_OP_WITH_CONSTANT(int64_t, bool, ge_scalar_i64, x >= constant);

// __half
UNARY_OP_WITH_CONSTANT(__half, __half, add_scalar_f16, x + constant);
UNARY_OP_WITH_CONSTANT(__half, __half, sub_scalar_f16, x - constant);
UNARY_OP_WITH_CONSTANT(__half, __half, mul_scalar_f16, x *constant);
UNARY_OP_WITH_CONSTANT(__half, __half, div_scalar_f16, x / constant);
UNARY_OP_WITH_CONSTANT(__half, __half, maximum_scalar_f16,
                       maximum(x, constant));
UNARY_OP_WITH_CONSTANT(__half, __half, minimum_scalar_f16,
                       minimum(x, constant));
UNARY_OP_WITH_CONSTANT(__half, __half, pow_f16,
                       __float2half(powf(__half2float(x),
                                         __half2float(constant))));
UNARY_OP_WITH_CONSTANT(__half, __half, leaky_relu_f16,
                       x > __half(0) ? x : constant * x);
UNARY_OP_WITH_CONSTANT(__half, __half, elu_f16,
                       x > __half(0)
                           ? x
                           : __hmul(constant,
                                    __hsub(__float2half(expf(__half2float(x))),
                                           __float2half(1.0f))));

UNARY_OP_WITH_CONSTANT(__half, bool, eq_scalar_f16, x == constant);
UNARY_OP_WITH_CONSTANT(__half, bool, ne_scalar_f16, x != constant);
UNARY_OP_WITH_CONSTANT(__half, bool, lt_scalar_f16, x < constant);
UNARY_OP_WITH_CONSTANT(__half, bool, le_scalar_f16, x <= constant);
UNARY_OP_WITH_CONSTANT(__half, bool, gt_scalar_f16, x > constant);
UNARY_OP_WITH_CONSTANT(__half, bool, ge_scalar_f16, x >= constant);

// __nv_bfloat16
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, add_scalar_bf16,
                       x + constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, sub_scalar_bf16,
                       x - constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, mul_scalar_bf16,
                       x *constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, div_scalar_bf16,
                       x / constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, maximum_scalar_bf16,
                       maximum(x, constant));
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, minimum_scalar_bf16,
                       minimum(x, constant));
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, pow_bf16,
                       __float2bfloat16(powf(__bfloat162float(x),
                                             __bfloat162float(constant))));
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, __nv_bfloat16, leaky_relu_bf16,
                       x > __nv_bfloat16(0) ? x : constant * x);
UNARY_OP_WITH_CONSTANT(
    __nv_bfloat16, __nv_bfloat16, elu_bf16,
    x > __nv_bfloat16(0)
        ? x
        : __hmul(constant, __hsub(__float2bfloat16(expf(__bfloat162float(x))),
                                  __float2bfloat16(1.0f))));

UNARY_OP_WITH_CONSTANT(__nv_bfloat16, bool, eq_scalar_bf16, x == constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, bool, ne_scalar_bf16, x != constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, bool, lt_scalar_bf16, x < constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, bool, le_scalar_bf16, x <= constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, bool, gt_scalar_bf16, x > constant);
UNARY_OP_WITH_CONSTANT(__nv_bfloat16, bool, ge_scalar_bf16, x >= constant);
