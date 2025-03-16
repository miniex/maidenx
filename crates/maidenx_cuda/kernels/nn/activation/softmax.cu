#include "../../cuda_utils.cuh"
#include <cfloat>
#include <stdint.h>

// Forward declaration of GPU functions
template <typename T> __device__ T get_neg_infinity();
template <typename T> __device__ T get_exp(T x);

// Specializations for different types
template <> __device__ float get_neg_infinity<float>() { return -FLT_MAX; }
template <> __device__ double get_neg_infinity<double>() { return -DBL_MAX; }
template <> __device__ __half get_neg_infinity<__half>() {
  return __float2half(-FLT_MAX);
}
template <> __device__ __nv_bfloat16 get_neg_infinity<__nv_bfloat16>() {
  return __float2bfloat16(-FLT_MAX);
}

template <> __device__ float get_exp<float>(float x) { return expf(x); }
template <> __device__ double get_exp<double>(double x) { return exp(x); }
template <> __device__ __half get_exp<__half>(__half x) { return hexp(x); }
template <> __device__ __nv_bfloat16 get_exp<__nv_bfloat16>(__nv_bfloat16 x) {
  return __float2bfloat16(expf(__bfloat162float(x)));
}

// Helper function to compute product of dimensions
__device__ size_t product_of_dimensions(const size_t *dims, size_t start,
                                        size_t end) {
  size_t result = 1;
  for (size_t i = start; i < end; ++i) {
    result *= dims[i];
  }
  return result;
}

#define SOFTMAX_OP(TYPENAME, FN_NAME)                                          \
  extern "C" __global__ void cuda_softmax_##FN_NAME##_kernel(                  \
      const size_t num_els, const size_t num_dims, const size_t dim,           \
      const size_t *metadata, const TYPENAME *input, TYPENAME *output) {       \
                                                                               \
    /* Default to last dimension if out of bounds */                           \
    const size_t actual_dim = (dim >= num_dims) ? (num_dims - 1) : dim;        \
                                                                               \
    const size_t *dims = metadata;                                             \
    const size_t *strides = metadata + num_dims;                               \
    const size_t offset = metadata[2 * num_dims];                              \
                                                                               \
    /* Calculate sizes for slicing */                                          \
    const size_t pre_dim_size = product_of_dimensions(dims, 0, actual_dim);    \
    const size_t dim_size = dims[actual_dim];                                  \
    const size_t post_dim_size =                                               \
        product_of_dimensions(dims, actual_dim + 1, num_dims);                 \
                                                                               \
    /* Check if the input is contiguous */                                     \
    bool is_contiguous = true;                                                 \
    {                                                                          \
      size_t acc = 1;                                                          \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        if (strides[d] != acc) {                                               \
          is_contiguous = false;                                               \
          break;                                                               \
        }                                                                      \
        acc *= dims[d];                                                        \
      }                                                                        \
    }                                                                          \
                                                                               \
    const size_t total_slices = pre_dim_size * post_dim_size;                  \
                                                                               \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                   \
         idx < total_slices; idx += blockDim.x * gridDim.x) {                  \
                                                                               \
      const size_t pre_idx = idx / post_dim_size;                              \
      const size_t post_idx = idx % post_dim_size;                             \
                                                                               \
      /* Find max value in this slice for numerical stability */               \
      TYPENAME max_val = get_neg_infinity<TYPENAME>();                         \
                                                                               \
      for (size_t i = 0; i < dim_size; ++i) {                                  \
        size_t in_idx;                                                         \
        if (is_contiguous) {                                                   \
          in_idx = offset + pre_idx * (dim_size * post_dim_size) +             \
                   i * post_dim_size + post_idx;                               \
        } else {                                                               \
          /* Calculate index for non-contiguous tensor */                      \
          in_idx = offset;                                                     \
          size_t remaining = pre_idx;                                          \
          for (size_t d = 0; d < actual_dim; ++d) {                            \
            size_t coord = remaining % dims[d];                                \
            remaining /= dims[d];                                              \
            in_idx += coord * strides[d];                                      \
          }                                                                    \
          in_idx += i * strides[actual_dim];                                   \
                                                                               \
          remaining = post_idx;                                                \
          for (int d = num_dims - 1; d > actual_dim; --d) {                    \
            size_t coord = remaining % dims[d];                                \
            remaining /= dims[d];                                              \
            in_idx += coord * strides[d];                                      \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (input[in_idx] > max_val) {                                         \
          max_val = input[in_idx];                                             \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Compute sum of exponentials for this slice */                         \
      TYPENAME sum =                                                           \
          get_exp<TYPENAME>(get_neg_infinity<TYPENAME>() - max_val);           \
                                                                               \
      for (size_t i = 0; i < dim_size; ++i) {                                  \
        size_t in_idx;                                                         \
        if (is_contiguous) {                                                   \
          in_idx = offset + pre_idx * (dim_size * post_dim_size) +             \
                   i * post_dim_size + post_idx;                               \
        } else {                                                               \
          /* Calculate index for non-contiguous tensor */                      \
          in_idx = offset;                                                     \
          size_t remaining = pre_idx;                                          \
          for (size_t d = 0; d < actual_dim; ++d) {                            \
            size_t coord = remaining % dims[d];                                \
            remaining /= dims[d];                                              \
            in_idx += coord * strides[d];                                      \
          }                                                                    \
          in_idx += i * strides[actual_dim];                                   \
                                                                               \
          remaining = post_idx;                                                \
          for (int d = num_dims - 1; d > actual_dim; --d) {                    \
            size_t coord = remaining % dims[d];                                \
            remaining /= dims[d];                                              \
            in_idx += coord * strides[d];                                      \
          }                                                                    \
        }                                                                      \
                                                                               \
        sum += get_exp<TYPENAME>(input[in_idx] - max_val);                     \
      }                                                                        \
                                                                               \
      /* Calculate softmax for each element in this slice */                   \
      for (size_t i = 0; i < dim_size; ++i) {                                  \
        size_t in_idx;                                                         \
        if (is_contiguous) {                                                   \
          in_idx = offset + pre_idx * (dim_size * post_dim_size) +             \
                   i * post_dim_size + post_idx;                               \
        } else {                                                               \
          /* Calculate index for non-contiguous tensor */                      \
          in_idx = offset;                                                     \
          size_t remaining = pre_idx;                                          \
          for (size_t d = 0; d < actual_dim; ++d) {                            \
            size_t coord = remaining % dims[d];                                \
            remaining /= dims[d];                                              \
            in_idx += coord * strides[d];                                      \
          }                                                                    \
          in_idx += i * strides[actual_dim];                                   \
                                                                               \
          remaining = post_idx;                                                \
          for (int d = num_dims - 1; d > actual_dim; --d) {                    \
            size_t coord = remaining % dims[d];                                \
            remaining /= dims[d];                                              \
            in_idx += coord * strides[d];                                      \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* Calculate output index (assuming contiguous output) */              \
        size_t out_idx = pre_idx * (dim_size * post_dim_size) +                \
                         i * post_dim_size + post_idx;                         \
                                                                               \
        output[out_idx] = get_exp<TYPENAME>(input[in_idx] - max_val) / sum;    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_softmax_##FN_NAME(                                      \
      size_t num_els, size_t num_dims, size_t dim, const size_t *metadata,     \
      const TYPENAME *input, TYPENAME *output) {                               \
                                                                               \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_softmax_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                  \
        num_els, num_dims, dim, metadata, input, output);                      \
  }

// Generate implementations for all supported types
SOFTMAX_OP(float, f32)
SOFTMAX_OP(double, f64)
SOFTMAX_OP(__half, f16)
SOFTMAX_OP(__nv_bfloat16, bf16)
