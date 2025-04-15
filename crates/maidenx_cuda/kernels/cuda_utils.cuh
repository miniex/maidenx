// Reference: candle-rs (https://github.com/huggingface/candle)

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <stdint.h>

__device__ __forceinline__ bool is_contiguous(const size_t num_dims,
                                              const size_t *dims,
                                              const size_t *strides) {
  size_t acc = 1;
  for (unsigned int d = 0; d < num_dims; d++) {
    unsigned int dim_idx = num_dims - 1 - d;
    if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
      return false;
    }
    acc *= dims[dim_idx];
  }
  return true;
}

__device__ __forceinline__ unsigned int
get_strided_index(size_t idx, size_t num_dims, const size_t *dims,
                  const size_t *strides) {
  size_t strided_i = 0;
  for (int d = num_dims - 1; d >= 0; d--) {
    size_t dim_idx_value = idx % dims[d];
    strided_i += (strides[d] == 0 ? 0 : dim_idx_value * strides[d]);
    idx /= dims[d];
  }
  return strided_i;
}

__device__ __forceinline__ unsigned int restrided(const unsigned int strided_i,
                                                  const size_t num_dims,
                                                  const size_t *dims,
                                                  const size_t *strides,
                                                  const size_t *new_strides) {
  unsigned int idx = 0;
  for (int d = 0; d < num_dims; d++) {
    idx += (strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d]) *
           new_strides[d];
  }
  return idx;
}

// __half

// __nv_bfloat16
