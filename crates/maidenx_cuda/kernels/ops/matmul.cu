#include "../cuda_utils.cuh"
#include <stdint.h>
#include <stdio.h>

#define MATMUL_OP(TYPENAME, FN_NAME)                                           \
  __global__ void cuda_##FN_NAME##_kernel(                                     \
      const size_t num_els, const size_t *metadata, const TYPENAME *A,         \
      const TYPENAME *B, TYPENAME *C) {                                        \
    if (!metadata || !A || !B || !C)                                           \
      return;                                                                  \
                                                                               \
    size_t out_ndim = metadata[0];                                             \
    size_t a_ndim = metadata[1];                                               \
    size_t b_ndim = metadata[2];                                               \
                                                                               \
    const size_t *out_shape = metadata + 3;                                    \
    const size_t *a_shape = out_shape + out_ndim;                              \
    const size_t *b_shape = a_shape + a_ndim;                                  \
    const size_t *a_strides = b_shape + b_ndim;                                \
    const size_t *b_strides = a_strides + a_ndim;                              \
                                                                               \
    size_t a_offset = *(b_strides + b_ndim);                                   \
    size_t b_offset = *(b_strides + b_ndim + 1);                               \
                                                                               \
    size_t M = out_shape[out_ndim - 2];                                        \
    size_t N = out_shape[out_ndim - 1];                                        \
    size_t K = a_shape[a_ndim - 1];                                            \
                                                                               \
    bool a_cont = is_contiguous(a_ndim, a_shape, a_strides);                   \
    bool b_cont = is_contiguous(b_ndim, b_shape, b_strides);                   \
                                                                               \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;    \
         idx += blockDim.x * gridDim.x) {                                      \
      size_t mn = idx % (M * N);                                               \
      size_t batch_idx = idx / (M * N);                                        \
                                                                               \
      size_t m = mn / N;                                                       \
      size_t n = mn % N;                                                       \
                                                                               \
      TYPENAME acc = 0;                                                        \
      if (a_cont && b_cont) {                                                  \
        size_t a_base = a_offset + batch_idx * (M * K) + m * K;                \
        size_t b_base = b_offset + batch_idx * (K * N);                        \
        for (size_t k = 0; k < K; k++) {                                       \
          acc += A[a_base + k] * B[b_base + k * N + n];                        \
        }                                                                      \
      } else {                                                                 \
        for (size_t k = 0; k < K; k++) {                                       \
          size_t a_idx = a_offset + batch_idx * a_strides[0] +                 \
                         m * a_strides[a_ndim - 2] +                           \
                         k * a_strides[a_ndim - 1];                            \
          size_t b_idx = b_offset + batch_idx * b_strides[0] +                 \
                         k * b_strides[b_ndim - 2] +                           \
                         n * b_strides[b_ndim - 1];                            \
          acc += A[a_idx] * B[b_idx];                                          \
        }                                                                      \
      }                                                                        \
      C[idx] = acc;                                                            \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(size_t num_els, const size_t *metadata,       \
                                 const TYPENAME *A, const TYPENAME *B,         \
                                 TYPENAME *C) {                                \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(num_els, metadata, A, B,  \
                                                     C);                       \
  }

#define MATMUL_BACKWARD_OP(TYPENAME, FN_NAME)                                  \
  __global__ void cuda_##FN_NAME##_grad_a_kernel(                              \
      const size_t num_els, const size_t *metadata,                            \
      const TYPENAME *grad_output, const TYPENAME *B, TYPENAME *grad_A) {      \
    if (!metadata || !grad_output || !B || !grad_A)                            \
      return;                                                                  \
                                                                               \
    size_t out_ndim = metadata[0];                                             \
    size_t a_ndim = metadata[1];                                               \
    size_t b_ndim = metadata[2];                                               \
                                                                               \
    const size_t *out_shape = metadata + 3;                                    \
    const size_t *a_shape = out_shape + out_ndim;                              \
    const size_t *b_shape = out_shape + out_ndim + a_ndim; /* Skip A shape */  \
    const size_t *a_strides = b_shape + b_ndim;                                \
    const size_t *b_strides = a_strides + a_ndim;                              \
                                                                               \
    size_t a_offset = *(b_strides + b_ndim);                                   \
    size_t b_offset = *(b_strides + b_ndim + 1);                               \
                                                                               \
    size_t M = out_shape[out_ndim > 0 ? out_ndim - 2 : 0];                     \
    size_t N = out_shape[out_ndim > 0 ? out_ndim - 1 : 0];                     \
    size_t K = b_shape[b_ndim - 2]; /* This is the common dimension */         \
                                                                               \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;    \
         idx += blockDim.x * gridDim.x) {                                      \
      size_t mk = idx % (M * K);                                               \
      size_t batch_idx = idx / (M * K);                                        \
                                                                               \
      size_t m = mk / K;                                                       \
      size_t k = mk % K;                                                       \
                                                                               \
      TYPENAME acc = 0;                                                        \
      if (out_ndim == 0) { /* scalar grad_output */                            \
        acc = grad_output[0] * B[b_offset + k];                                \
      } else {                                                                 \
        size_t grad_base = batch_idx * (M * N) + m * N;                        \
        size_t b_base = b_offset + batch_idx * (K * N);                        \
        for (size_t n = 0; n < N; n++) {                                       \
          acc += grad_output[grad_base + n] * B[b_base + k * N + n];           \
        }                                                                      \
      }                                                                        \
      grad_A[idx] = acc;                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  __global__ void cuda_##FN_NAME##_grad_b_kernel(                              \
      const size_t num_els, const size_t *metadata,                            \
      const TYPENAME *grad_output, const TYPENAME *A, TYPENAME *grad_B) {      \
    if (!metadata || !grad_output || !A || !grad_B)                            \
      return;                                                                  \
                                                                               \
    size_t out_ndim = metadata[0];                                             \
    size_t a_ndim = metadata[1];                                               \
    size_t b_ndim = metadata[2];                                               \
                                                                               \
    const size_t *out_shape = metadata + 3;                                    \
    const size_t *a_shape = out_shape + out_ndim;                              \
    const size_t *b_shape = a_shape + a_ndim;                                  \
    const size_t *a_strides = b_shape + b_ndim;                                \
    const size_t *b_strides = a_strides + a_ndim;                              \
                                                                               \
    size_t a_offset = *(b_strides + b_ndim);                                   \
    size_t b_offset = *(b_strides + b_ndim + 1);                               \
                                                                               \
    size_t M = out_shape[out_ndim > 0 ? out_ndim - 2 : 0];                     \
    size_t N = out_shape[out_ndim > 0 ? out_ndim - 1 : 0];                     \
    size_t K = a_shape[a_ndim - 1];                                            \
                                                                               \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;    \
         idx += blockDim.x * gridDim.x) {                                      \
      size_t kn = idx % (K * N);                                               \
      size_t batch_idx = idx / (K * N);                                        \
                                                                               \
      size_t k = kn / N;                                                       \
      size_t n = kn % N;                                                       \
                                                                               \
      TYPENAME acc = 0;                                                        \
      if (out_ndim == 0) { /* scalar grad_output */                            \
        acc = grad_output[0] * A[a_offset + k];                                \
      } else {                                                                 \
        size_t grad_base = batch_idx * (M * N);                                \
        size_t a_base = a_offset + batch_idx * (M * K);                        \
        for (size_t m = 0; m < M; m++) {                                       \
          acc += A[a_base + m * K + k] * grad_output[grad_base + m * N + n];   \
        }                                                                      \
      }                                                                        \
      grad_B[idx] = acc;                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els_a, size_t num_els_b, const size_t *metadata,              \
      const TYPENAME *grad_output, const TYPENAME *A, const TYPENAME *B,       \
      TYPENAME *grad_A, TYPENAME *grad_B) {                                    \
    if (grad_A != nullptr) {                                                   \
      dim3 block_dim(256);                                                     \
      dim3 grid_dim((num_els_a + block_dim.x - 1) / block_dim.x);              \
      cuda_##FN_NAME##_grad_a_kernel<<<grid_dim, block_dim>>>(                 \
          num_els_a, metadata, grad_output, B, grad_A);                        \
    }                                                                          \
                                                                               \
    if (grad_B != nullptr) {                                                   \
      dim3 block_dim(256);                                                     \
      dim3 grid_dim((num_els_b + block_dim.x - 1) / block_dim.x);              \
      cuda_##FN_NAME##_grad_b_kernel<<<grid_dim, block_dim>>>(                 \
          num_els_b, metadata, grad_output, A, grad_B);                        \
    }                                                                          \
  }

MATMUL_OP(float, matmul_f32);
MATMUL_BACKWARD_OP(float, matmul_backward_f32);
MATMUL_OP(double, matmul_f64);
MATMUL_BACKWARD_OP(double, matmul_backward_f64);
MATMUL_OP(uint8_t, matmul_u8);
MATMUL_BACKWARD_OP(uint8_t, matmul_backward_u8);
MATMUL_OP(uint32_t, matmul_u32);
MATMUL_BACKWARD_OP(uint32_t, matmul_backward_u32);
MATMUL_OP(int8_t, matmul_i8);
MATMUL_BACKWARD_OP(int8_t, matmul_backward_i8);
MATMUL_OP(int32_t, matmul_i32);
MATMUL_BACKWARD_OP(int32_t, matmul_backward_i32);
MATMUL_OP(int64_t, matmul_i64);
MATMUL_BACKWARD_OP(int64_t, matmul_backward_i64);

// __half
MATMUL_OP(__half, matmul_f16);
MATMUL_BACKWARD_OP(__half, matmul_backward_f16);

// __nv_bfloat16
MATMUL_OP(__nv_bfloat16, matmul_bf16);
MATMUL_BACKWARD_OP(__nv_bfloat16, matmul_backward_bf16);
