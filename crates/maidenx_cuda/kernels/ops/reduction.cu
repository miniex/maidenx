#include "../compatibility.cuh"
#include "../cuda_utils.cuh"
#include <stdint.h>

#define MAX_DIMS 10

#define SUM_OP(TYPENAME, FN_NAME)                                              \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims, const size_t num_sum_dims,  \
      const size_t *metadata, const TYPENAME *inp, TYPENAME *out) {            \
    const size_t *dims = metadata;                                             \
    const size_t *strides = metadata + num_dims;                               \
    const size_t *sum_dims_l = metadata + 2 * num_dims;                        \
    const size_t *sum_dims_s = metadata + 2 * num_dims + num_sum_dims;         \
    const size_t offset = *(metadata + 2 * num_dims + 2 * num_sum_dims);       \
                                                                               \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[offset + i]);                           \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned strided_i =                                                   \
            offset + get_strided_index(i, num_dims, dims, strides);            \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[strided_i]);                            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" void cuda_##FN_NAME(size_t num_els, size_t num_dims,              \
                                 size_t num_red_dims, const size_t *metadata,  \
                                 const TYPENAME *inp, TYPENAME *out) {         \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els, num_dims, num_red_dims, metadata, inp, out);                  \
  }

#define SUM_TO_SHAPE_OP(TYPENAME, FN_NAME)                                     \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims, const size_t *metadata,     \
      const TYPENAME *inp, TYPENAME *out) {                                    \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t *output_dims = metadata + 2 * num_dims;                       \
    const size_t offset = *(metadata + 3 * num_dims);                          \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    size_t reduction_factors[MAX_DIMS];                                        \
    for (size_t d = 0; d < num_dims; d++) {                                    \
      reduction_factors[d] = input_dims[d] / output_dims[d];                   \
    }                                                                          \
                                                                               \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;  \
         i += blockDim.x * gridDim.x) {                                        \
                                                                               \
      size_t coords[MAX_DIMS];                                                 \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        coords[d] = tmp_i % input_dims[d];                                     \
        tmp_i /= input_dims[d];                                                \
      }                                                                        \
                                                                               \
      size_t dst_idx = 0;                                                      \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t out_coord = coords[d] / reduction_factors[d];                   \
        dst_idx = dst_idx * output_dims[d] + out_coord;                        \
      }                                                                        \
                                                                               \
      size_t src_idx = offset;                                                 \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        src_idx += coords[d] * input_strides[d];                               \
      }                                                                        \
                                                                               \
      atomicAdd(out + dst_idx, inp[src_idx]);                                  \
    }                                                                          \
  }                                                                            \
  extern "C" void cuda_##FN_NAME(size_t num_els, size_t num_dims,              \
                                 const size_t *metadata, const TYPENAME *inp,  \
                                 TYPENAME *out) {                              \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(num_els, num_dims,        \
                                                     metadata, inp, out);      \
  }

#define MEAN_OP(TYPENAME, FN_NAME)                                             \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims, const size_t num_mean_dims, \
      const size_t *metadata, const TYPENAME *inp, TYPENAME *out) {            \
    const size_t *dims = metadata;                                             \
    const size_t *strides = metadata + num_dims;                               \
    const size_t *mean_dims_l = metadata + 2 * num_dims;                       \
    const size_t *mean_dims_s = metadata + 2 * num_dims + num_mean_dims;       \
    const size_t offset = *(metadata + 2 * num_dims + 2 * num_mean_dims);      \
                                                                               \
    /* Calculate reduction factor */                                           \
    size_t reduction_factor = 1;                                               \
    for (size_t i = 0; i < num_mean_dims; i++) {                               \
      reduction_factor *= mean_dims_l[i];                                      \
    }                                                                          \
    TYPENAME factor = static_cast<TYPENAME>(reduction_factor);                 \
                                                                               \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_mean_dims; ++nd) {                  \
          size_t stride = mean_dims_s[nd];                                     \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / mean_dims_l[nd]) * stride + post;                 \
        }                                                                      \
        atomicAdd(out + dst_index, inp[offset + i] / factor);                  \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;             \
           i < num_els; i += blockDim.x * gridDim.x) {                         \
        unsigned strided_i =                                                   \
            offset + get_strided_index(i, num_dims, dims, strides);            \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_mean_dims; ++nd) {                  \
          size_t stride = mean_dims_s[nd];                                     \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / mean_dims_l[nd]) * stride + post;                 \
        }                                                                      \
        atomicAdd(out + dst_index, inp[strided_i] / factor);                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" void cuda_##FN_NAME(size_t num_els, size_t num_dims,              \
                                 size_t num_red_dims, const size_t *metadata,  \
                                 const TYPENAME *inp, TYPENAME *out) {         \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els, num_dims, num_red_dims, metadata, inp, out);                  \
  }

#define FOLD_OP(TYPENAME, FN_NAME)                                             \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els, const size_t num_dims, const size_t *metadata,     \
      const TYPENAME *inp, TYPENAME *out) {                                    \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t fold_dim = *(metadata + 2 * num_dims);                        \
    const size_t window_dim = *(metadata + 2 * num_dims + 1);                  \
    const size_t fold_size = *(metadata + 2 * num_dims + 2);                   \
    const size_t step = *(metadata + 2 * num_dims + 3);                        \
    const size_t window_size = *(metadata + 2 * num_dims + 4);                 \
    const size_t offset = *(metadata + 2 * num_dims + 5);                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;  \
         i += blockDim.x * gridDim.x) {                                        \
                                                                               \
      /* Calculate coordinates in input tensor */                              \
      size_t coords[MAX_DIMS];                                                 \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        coords[d] = tmp_i % input_dims[d];                                     \
        tmp_i /= input_dims[d];                                                \
      }                                                                        \
                                                                               \
      /* Calculate source index using input strides */                         \
      size_t src_idx = offset;                                                 \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        src_idx += coords[d] * input_strides[d];                               \
      }                                                                        \
                                                                               \
      /* Extract window index and position in window */                        \
      const size_t window_idx = coords[fold_dim];                              \
      const size_t pos_in_window = coords[window_dim];                         \
                                                                               \
      /* Calculate position in the original folded dimension */                \
      const size_t orig_pos = window_idx * step + pos_in_window;               \
                                                                               \
      /* Skip if outside the bounds of the folded dimension */                 \
      if (orig_pos >= fold_size) {                                             \
        continue;                                                              \
      }                                                                        \
                                                                               \
      /* Calculate destination index in output */                              \
      size_t dst_idx = 0;                                                      \
      size_t dst_dim_idx = 0;                                                  \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        if (d == window_dim) {                                                 \
          continue; /* Skip window dimension */                                \
        } else if (d == fold_dim) {                                            \
          dst_idx = dst_idx * fold_size + orig_pos;                            \
        } else {                                                               \
          dst_idx = dst_idx * input_dims[d] + coords[d];                       \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Add value to output */                                                \
      atomicAdd(out + dst_idx, inp[src_idx]);                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(size_t num_els, size_t num_dims,              \
                                 const size_t *metadata, const TYPENAME *inp,  \
                                 TYPENAME *out) {                              \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(num_els, num_dims,        \
                                                     metadata, inp, out);      \
  }

SUM_OP(float, sum_f32);
SUM_OP(double, sum_f64);
SUM_OP(uint8_t, sum_u8);
SUM_OP(uint32_t, sum_u32);
SUM_OP(int8_t, sum_i8);
SUM_OP(int32_t, sum_i32);
SUM_OP(int64_t, sum_i64);

SUM_TO_SHAPE_OP(float, sum_to_shape_f32);
SUM_TO_SHAPE_OP(double, sum_to_shape_f64);
SUM_TO_SHAPE_OP(uint8_t, sum_to_shape_u8);
SUM_TO_SHAPE_OP(uint32_t, sum_to_shape_u32);
SUM_TO_SHAPE_OP(int8_t, sum_to_shape_i8);
SUM_TO_SHAPE_OP(int32_t, sum_to_shape_i32);
SUM_TO_SHAPE_OP(int64_t, sum_to_shape_i64);

MEAN_OP(float, mean_f32);
MEAN_OP(double, mean_f64);

FOLD_OP(float, fold_f32);
FOLD_OP(double, fold_f64);
FOLD_OP(uint8_t, fold_u8);
FOLD_OP(uint32_t, fold_u32);
FOLD_OP(int8_t, fold_i8);
FOLD_OP(int32_t, fold_i32);
FOLD_OP(int64_t, fold_i64);

// __half
SUM_OP(__half, sum_f16);
SUM_TO_SHAPE_OP(__half, sum_to_shape_f16);
MEAN_OP(__half, mean_f16);
FOLD_OP(__half, fold_f16);

// __nv_bfloat16
SUM_OP(__nv_bfloat16, sum_bf16);
SUM_TO_SHAPE_OP(__nv_bfloat16, sum_to_shape_bf16);
MEAN_OP(__nv_bfloat16, mean_bf16);
FOLD_OP(__nv_bfloat16, fold_bf16);
