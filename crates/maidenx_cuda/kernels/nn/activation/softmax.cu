#include "../../cuda_utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Compute the product of all elements in dims[start...end-1]
__device__ size_t product_of_dimensions(const size_t *dims, size_t start,
                                        size_t end) {
  size_t result = 1;
  for (size_t i = start; i < end; ++i) {
    result *= dims[i];
  }
  return result;
}

// Process one slice along the specified dimension
#define PROCESS_SOFTMAX_SLICE(TYPENAME, FLOAT_TYPE, CONVERT_IN, CONVERT_OUT,   \
                              EXP_FUNC)                                        \
  const size_t *dims = metadata;                                               \
  const size_t *strides = metadata + num_dims;                                 \
  const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;             \
                                                                               \
  /* Calculate sizes before, at, and after the specified dimension */          \
  size_t pre_dim_size = product_of_dimensions(dims, 0, dim);                   \
  size_t dim_size = dims[dim];                                                 \
  size_t post_dim_size = product_of_dimensions(dims, dim + 1, num_dims);       \
                                                                               \
  /* Each thread processes one slice along the specified dimension */          \
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                     \
       idx < pre_dim_size * post_dim_size; idx += blockDim.x * gridDim.x) {    \
                                                                               \
    /* Calculate coordinates in the pre_dim and post_dim space */              \
    size_t pre_idx = idx / post_dim_size;                                      \
    size_t post_idx = idx % post_dim_size;                                     \
                                                                               \
    /* Find max for numerical stability */                                     \
    FLOAT_TYPE max_val = -INFINITY;                                            \
    for (size_t i = 0; i < dim_size; ++i) {                                    \
      /* Calculate the flat index for this element */                          \
      size_t flat_idx;                                                         \
      if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {     \
        /* Contiguous case: we can compute the index directly */               \
        flat_idx = offset + pre_idx * dim_size * post_dim_size +               \
                   i * post_dim_size + post_idx;                               \
      } else {                                                                 \
        /* Non-contiguous case: compute index using strides */                 \
        flat_idx = offset;                                                     \
        size_t remaining = pre_idx;                                            \
        for (size_t d = 0; d < dim; ++d) {                                     \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          flat_idx += coord * strides[d];                                      \
        }                                                                      \
        flat_idx += i * strides[dim];                                          \
                                                                               \
        remaining = post_idx;                                                  \
        for (size_t d = num_dims - 1; d > dim; --d) {                          \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          flat_idx += coord * strides[d];                                      \
        }                                                                      \
      }                                                                        \
                                                                               \
      FLOAT_TYPE val = CONVERT_IN(input[flat_idx]);                            \
      max_val = max(max_val, val);                                             \
    }                                                                          \
                                                                               \
    /* Compute sum of exponentials */                                          \
    FLOAT_TYPE sum = 0;                                                        \
    for (size_t i = 0; i < dim_size; ++i) {                                    \
      /* Calculate the flat index for this element */                          \
      size_t flat_idx;                                                         \
      if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {     \
        flat_idx = offset + pre_idx * dim_size * post_dim_size +               \
                   i * post_dim_size + post_idx;                               \
      } else {                                                                 \
        flat_idx = offset;                                                     \
        size_t remaining = pre_idx;                                            \
        for (size_t d = 0; d < dim; ++d) {                                     \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          flat_idx += coord * strides[d];                                      \
        }                                                                      \
        flat_idx += i * strides[dim];                                          \
                                                                               \
        remaining = post_idx;                                                  \
        for (size_t d = num_dims - 1; d > dim; --d) {                          \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          flat_idx += coord * strides[d];                                      \
        }                                                                      \
      }                                                                        \
                                                                               \
      sum += EXP_FUNC(CONVERT_IN(input[flat_idx]) - max_val);                  \
    }                                                                          \
                                                                               \
    /* Compute softmax for each element in this slice */                       \
    for (size_t i = 0; i < dim_size; ++i) {                                    \
      /* Calculate the input flat index for this element */                    \
      size_t in_flat_idx;                                                      \
      if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {     \
        in_flat_idx = offset + pre_idx * dim_size * post_dim_size +            \
                      i * post_dim_size + post_idx;                            \
      } else {                                                                 \
        in_flat_idx = offset;                                                  \
        size_t remaining = pre_idx;                                            \
        for (size_t d = 0; d < dim; ++d) {                                     \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          in_flat_idx += coord * strides[d];                                   \
        }                                                                      \
        in_flat_idx += i * strides[dim];                                       \
                                                                               \
        remaining = post_idx;                                                  \
        for (size_t d = num_dims - 1; d > dim; --d) {                          \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          in_flat_idx += coord * strides[d];                                   \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Calculate the output flat index */                                    \
      size_t out_flat_idx;                                                     \
      if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {     \
        out_flat_idx =                                                         \
            pre_idx * dim_size * post_dim_size + i * post_dim_size + post_idx; \
      } else {                                                                 \
        /* For non-contiguous output, we need to calculate the index manually  \
         */                                                                    \
        /* We assume output is laid out in standard row-major order */         \
        out_flat_idx = 0;                                                      \
        size_t remaining = pre_idx;                                            \
        for (size_t d = 0; d < dim; ++d) {                                     \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          out_flat_idx = out_flat_idx * dims[d] + coord;                       \
        }                                                                      \
        out_flat_idx = out_flat_idx * dim_size + i;                            \
                                                                               \
        remaining = post_idx;                                                  \
        for (size_t d = num_dims - 1; d > dim; --d) {                          \
          size_t coord = remaining % dims[d];                                  \
          remaining /= dims[d];                                                \
          out_flat_idx = out_flat_idx * dims[d] + coord;                       \
        }                                                                      \
      }                                                                        \
                                                                               \
      FLOAT_TYPE val = CONVERT_IN(input[in_flat_idx]);                         \
      output[out_flat_idx] = CONVERT_OUT(EXP_FUNC(val - max_val) / sum);       \
    }                                                                          \
  }

// Macro to define kernel and host functions for a given type
#define DEFINE_SOFTMAX_FUNCTIONS(TYPE, SUFFIX, FLOAT_TYPE, CONVERT_IN,         \
                                 CONVERT_OUT, EXP_FUNC)                        \
  extern "C" __global__ void cuda_softmax_##SUFFIX##_kernel(                   \
      const size_t num_els, const size_t num_dims, const size_t dim,           \
      const size_t *metadata, const TYPE *input, TYPE *output) {               \
    PROCESS_SOFTMAX_SLICE(TYPE, FLOAT_TYPE, CONVERT_IN, CONVERT_OUT, EXP_FUNC) \
  }                                                                            \
                                                                               \
  extern "C" void cuda_softmax_##SUFFIX(size_t num_els, size_t num_dims,       \
                                        size_t dim, const size_t *metadata,    \
                                        const TYPE *input, TYPE *output) {     \
    if (dim >= num_dims) {                                                     \
      dim = num_dims - 1; /* Default to last dimension if out of bounds */     \
    }                                                                          \
                                                                               \
    /* Each thread processes one slice, so we need num_els / dim_size threads  \
     */                                                                        \
    size_t dim_size = metadata[dim];                                           \
    size_t num_slices = num_els / dim_size;                                    \
                                                                               \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_slices + block_dim.x - 1) / block_dim.x);               \
    cuda_softmax_##SUFFIX##_kernel<<<grid_dim, block_dim>>>(                   \
        num_els, num_dims, dim, metadata, input, output);                      \
  }

// Conversion identity function for float/double
__device__ float identity_float(float x) { return x; }
__device__ double identity_double(double x) { return x; }

// Define functions for different types
DEFINE_SOFTMAX_FUNCTIONS(float, f32, float, identity_float, identity_float,
                         expf)
DEFINE_SOFTMAX_FUNCTIONS(double, f64, double, identity_double, identity_double,
                         exp)
DEFINE_SOFTMAX_FUNCTIONS(__half, f16, float, __half2float, __float2half, expf)
DEFINE_SOFTMAX_FUNCTIONS(__nv_bfloat16, bf16, float, __bfloat162float,
                         __float2bfloat16, expf)
