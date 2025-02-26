#include "../cuda_utils.cuh"
#include <stdint.h>
#include <stdio.h>

template <typename T> __device__ T sub_with_clamp(T x, T y) {
  return (x > y) ? (x - y) : 0;
}

#define BINARY_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)                    \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims,                             \
      const size_t *dims_and_strides, const IN_TYPENAME *lhs,                  \
      const IN_TYPENAME *rhs, OUT_TYPENAME *out) {                             \
    const size_t *dims = dims_and_strides;                                     \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;               \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;               \
    bool lhs_cont = dims_and_strides == nullptr ||                             \
                    is_contiguous(num_dims, dims, lhs_strides);                \
    bool rhs_cont = dims_and_strides == nullptr ||                             \
                    is_contiguous(num_dims, dims, rhs_strides);                \
    if (lhs_cont && rhs_cont) {                                                \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        IN_TYPENAME x = lhs[i];                                                \
        IN_TYPENAME y = rhs[i];                                                \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else if (lhs_cont) {                                                     \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned int tmp_i = i;                                                \
        unsigned int rhs_i = 0;                                                \
        for (int d = num_dims - 1; d >= 0; d--) {                              \
          unsigned int i_dim = tmp_i % dims[d];                                \
          rhs_i += i_dim * rhs_strides[d];                                     \
          tmp_i /= dims[d];                                                    \
        }                                                                      \
        IN_TYPENAME x = lhs[i];                                                \
        IN_TYPENAME y = rhs[rhs_i];                                            \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else if (rhs_cont) {                                                     \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned int tmp_i = i;                                                \
        unsigned int lhs_i = 0;                                                \
        for (int d = num_dims - 1; d >= 0; d--) {                              \
          unsigned int i_dim = tmp_i % dims[d];                                \
          lhs_i += i_dim * lhs_strides[d];                                     \
          tmp_i /= dims[d];                                                    \
        }                                                                      \
        IN_TYPENAME x = lhs[lhs_i];                                            \
        IN_TYPENAME y = rhs[i];                                                \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned int tmp_i = i;                                                \
        unsigned int lhs_i = 0;                                                \
        unsigned int rhs_i = 0;                                                \
        for (int d = num_dims - 1; d >= 0; d--) {                              \
          unsigned int i_dim = tmp_i % dims[d];                                \
          lhs_i += i_dim * lhs_strides[d];                                     \
          rhs_i += i_dim * rhs_strides[d];                                     \
          tmp_i /= dims[d];                                                    \
        }                                                                      \
        IN_TYPENAME x = lhs[lhs_i];                                            \
        IN_TYPENAME y = rhs[rhs_i];                                            \
        out[i] = FUNC;                                                         \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els, size_t num_dims, const size_t *dims_and_strides,         \
      const IN_TYPENAME *lhs, const IN_TYPENAME *rhs, OUT_TYPENAME *out) {     \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els, num_dims, dims_and_strides, lhs, rhs, out);                   \
  }

BINARY_OP(float, float, add_f32, x + y);
BINARY_OP(float, float, sub_f32, x - y);
BINARY_OP(float, float, mul_f32, x *y);
BINARY_OP(float, float, div_f32, x / y);
BINARY_OP(double, double, add_f64, x + y);
BINARY_OP(double, double, sub_f64, x - y);
BINARY_OP(double, double, mul_f64, x *y);
BINARY_OP(double, double, div_f64, x / y);
BINARY_OP(bool, bool, add_bool, x + y);
BINARY_OP(bool, bool, sub_bool, x - y);
BINARY_OP(bool, bool, mul_bool, x *y);
BINARY_OP(bool, bool, div_bool, x &&y);
BINARY_OP(uint8_t, uint8_t, add_u8, x + y);
BINARY_OP(uint8_t, uint8_t, sub_u8, sub_with_clamp(x, y));
BINARY_OP(uint8_t, uint8_t, mul_u8, x *y);
BINARY_OP(uint8_t, uint8_t, div_u8, x / y);
BINARY_OP(uint32_t, uint32_t, add_u32, x + y);
BINARY_OP(uint32_t, uint32_t, sub_u32, sub_with_clamp(x, y));
BINARY_OP(uint32_t, uint32_t, mul_u32, x *y);
BINARY_OP(uint32_t, uint32_t, div_u32, x / y);
BINARY_OP(int8_t, int8_t, add_i8, x + y);
BINARY_OP(int8_t, int8_t, sub_i8, x - y);
BINARY_OP(int8_t, int8_t, mul_i8, x *y);
BINARY_OP(int8_t, int8_t, div_i8, x / y);
BINARY_OP(int32_t, int32_t, add_i32, x + y);
BINARY_OP(int32_t, int32_t, sub_i32, x - y);
BINARY_OP(int32_t, int32_t, mul_i32, x *y);
BINARY_OP(int32_t, int32_t, div_i32, x / y);
BINARY_OP(int64_t, int64_t, add_i64, x + y);
BINARY_OP(int64_t, int64_t, sub_i64, x - y);
BINARY_OP(int64_t, int64_t, mul_i64, x *y);
BINARY_OP(int64_t, int64_t, div_i64, x / y);

BINARY_OP(float, bool, logical_and_f32, (x != 0.0f && y != 0.0f));
BINARY_OP(float, bool, logical_or_f32, (x != 0.0f || y != 0.0f));
BINARY_OP(float, bool, logical_xor_f32, (x != 0.0f) != (y != 0.0f));
BINARY_OP(double, bool, logical_and_f64, (x != 0.0 && y != 0.0));
BINARY_OP(double, bool, logical_or_f64, (x != 0.0 || y != 0.0));
BINARY_OP(double, bool, logical_xor_f64, (x != 0.0) != (y != 0.0));
BINARY_OP(bool, bool, logical_and_bool, x &&y);
BINARY_OP(bool, bool, logical_or_bool, x || y);
BINARY_OP(bool, bool, logical_xor_bool, x != y);
BINARY_OP(uint8_t, bool, logical_and_u8, (x != 0u && y != 0u));
BINARY_OP(uint8_t, bool, logical_or_u8, (x != 0u || y != 0u));
BINARY_OP(uint8_t, bool, logical_xor_u8, (x != 0u) != (y != 0u));
BINARY_OP(uint32_t, bool, logical_and_u32, (x != 0u && y != 0u));
BINARY_OP(uint32_t, bool, logical_or_u32, (x != 0u || y != 0u));
BINARY_OP(uint32_t, bool, logical_xor_u32, (x != 0u) != (y != 0u));
BINARY_OP(int8_t, bool, logical_and_i8, (x != 0 && y != 0));
BINARY_OP(int8_t, bool, logical_or_i8, (x != 0 || y != 0));
BINARY_OP(int8_t, bool, logical_xor_i8, (x != 0) != (y != 0));
BINARY_OP(int32_t, bool, logical_and_i32, (x != 0 && y != 0));
BINARY_OP(int32_t, bool, logical_or_i32, (x != 0 || y != 0));
BINARY_OP(int32_t, bool, logical_xor_i32, (x != 0) != (y != 0));
BINARY_OP(int64_t, bool, logical_and_i64, (x != 0 && y != 0));
BINARY_OP(int64_t, bool, logical_or_i64, (x != 0 || y != 0));
BINARY_OP(int64_t, bool, logical_xor_i64, (x != 0) != (y != 0));

BINARY_OP(float, bool, eq_f32, x == y);
BINARY_OP(float, bool, ne_f32, x != y);
BINARY_OP(float, bool, lt_f32, x < y);
BINARY_OP(float, bool, le_f32, x <= y);
BINARY_OP(float, bool, gt_f32, x > y);
BINARY_OP(float, bool, ge_f32, x >= y);
BINARY_OP(double, bool, eq_f64, x == y);
BINARY_OP(double, bool, ne_f64, x != y);
BINARY_OP(double, bool, lt_f64, x < y);
BINARY_OP(double, bool, le_f64, x <= y);
BINARY_OP(double, bool, gt_f64, x > y);
BINARY_OP(double, bool, ge_f64, x >= y);
BINARY_OP(bool, bool, eq_bool, x == y);
BINARY_OP(bool, bool, ne_bool, x != y);
BINARY_OP(bool, bool, lt_bool, x < y);
BINARY_OP(bool, bool, le_bool, x <= y);
BINARY_OP(bool, bool, gt_bool, x > y);
BINARY_OP(bool, bool, ge_bool, x >= y);
BINARY_OP(uint8_t, bool, eq_u8, x == y);
BINARY_OP(uint8_t, bool, ne_u8, x != y);
BINARY_OP(uint8_t, bool, lt_u8, x < y);
BINARY_OP(uint8_t, bool, le_u8, x <= y);
BINARY_OP(uint8_t, bool, gt_u8, x > y);
BINARY_OP(uint8_t, bool, ge_u8, x >= y);
BINARY_OP(uint32_t, bool, eq_u32, x == y);
BINARY_OP(uint32_t, bool, ne_u32, x != y);
BINARY_OP(uint32_t, bool, lt_u32, x < y);
BINARY_OP(uint32_t, bool, le_u32, x <= y);
BINARY_OP(uint32_t, bool, gt_u32, x > y);
BINARY_OP(uint32_t, bool, ge_u32, x >= y);
BINARY_OP(int8_t, bool, eq_i8, x == y);
BINARY_OP(int8_t, bool, ne_i8, x != y);
BINARY_OP(int8_t, bool, lt_i8, x < y);
BINARY_OP(int8_t, bool, le_i8, x <= y);
BINARY_OP(int8_t, bool, gt_i8, x > y);
BINARY_OP(int8_t, bool, ge_i8, x >= y);
BINARY_OP(int32_t, bool, eq_i32, x == y);
BINARY_OP(int32_t, bool, ne_i32, x != y);
BINARY_OP(int32_t, bool, lt_i32, x < y);
BINARY_OP(int32_t, bool, le_i32, x <= y);
BINARY_OP(int32_t, bool, gt_i32, x > y);
BINARY_OP(int32_t, bool, ge_i32, x >= y);
BINARY_OP(int64_t, bool, eq_i64, x == y);
BINARY_OP(int64_t, bool, ne_i64, x != y);
BINARY_OP(int64_t, bool, lt_i64, x < y);
BINARY_OP(int64_t, bool, le_i64, x <= y);
BINARY_OP(int64_t, bool, gt_i64, x > y);
BINARY_OP(int64_t, bool, ge_i64, x >= y);

// __half
BINARY_OP(__half, __half, add_f16, x + y);
BINARY_OP(__half, __half, sub_f16, x - y);
BINARY_OP(__half, __half, mul_f16, x *y);
BINARY_OP(__half, __half, div_f16, x / y);

BINARY_OP(__half, bool, logical_and_f16,
          (x != __half(0.0f) && y != __half(0.0f)));
BINARY_OP(__half, bool, logical_or_f16,
          (x != __half(0.0f) || y != __half(0.0f)));
BINARY_OP(__half, bool, logical_xor_f16,
          (x != __half(0.0f)) != (y != __half(0.0f)));

BINARY_OP(__half, bool, eq_f16, x == y);
BINARY_OP(__half, bool, ne_f16, x != y);
BINARY_OP(__half, bool, lt_f16, x < y);
BINARY_OP(__half, bool, le_f16, x <= y);
BINARY_OP(__half, bool, gt_f16, x > y);
BINARY_OP(__half, bool, ge_f16, x >= y);

// __nv_bfloat16
BINARY_OP(__nv_bfloat16, __nv_bfloat16, add_bf16, x + y);
BINARY_OP(__nv_bfloat16, __nv_bfloat16, sub_bf16, x - y);
BINARY_OP(__nv_bfloat16, __nv_bfloat16, mul_bf16, x *y);
BINARY_OP(__nv_bfloat16, __nv_bfloat16, div_bf16, x / y);

BINARY_OP(__nv_bfloat16, bool, logical_and_bf16,
          (x != __nv_bfloat16(0.0f) && y != __nv_bfloat16(0.0f)));
BINARY_OP(__nv_bfloat16, bool, logical_or_bf16,
          (x != __nv_bfloat16(0.0f) || y != __nv_bfloat16(0.0f)));
BINARY_OP(__nv_bfloat16, bool, logical_xor_bf16,
          (x != __nv_bfloat16(0.0f)) != (y != __nv_bfloat16(0.0f)));

BINARY_OP(__nv_bfloat16, bool, eq_bf16, x == y);
BINARY_OP(__nv_bfloat16, bool, ne_bf16, x != y);
BINARY_OP(__nv_bfloat16, bool, lt_bf16, x < y);
BINARY_OP(__nv_bfloat16, bool, le_bf16, x <= y);
BINARY_OP(__nv_bfloat16, bool, gt_bf16, x > y);
BINARY_OP(__nv_bfloat16, bool, ge_bf16, x >= y);
