#include "../compatibility.cuh"
#include "../cuda_utils.cuh"
#include <stdint.h>

#define MAX_DIMS 10

#define PAD_WITH_CONSTANT_OP(TYPENAME, FN_NAME)                                \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els_in, const size_t num_els_out,                       \
      const size_t num_dims, const size_t *metadata, const TYPENAME *inp,      \
      TYPENAME *out, const TYPENAME pad_value) {                               \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;     \
    const size_t *output_dims = metadata + 2 * num_dims + 1;                   \
    const size_t *paddings = metadata + 3 * num_dims + 1;                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    /* Initialize output with pad value */                                     \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
      out[i] = pad_value;                                                      \
    }                                                                          \
    /* No need for __syncthreads() here as we're using global memory */        \
                                                                               \
    bool is_input_contiguous =                                                 \
        is_contiguous(num_dims, input_dims, input_strides);                    \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
                                                                               \
      /* Calculate coordinates in output tensor */                             \
      size_t out_coords[MAX_DIMS];                                             \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        out_coords[d] = tmp_i % output_dims[d];                                \
        tmp_i /= output_dims[d];                                               \
      }                                                                        \
                                                                               \
      /* Calculate corresponding coordinates in input tensor */                \
      bool in_bounds = true;                                                   \
      size_t in_coords[MAX_DIMS];                                              \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t pad_before = paddings[d * 2];                                   \
        int pos = (int)out_coords[d] - (int)pad_before;                        \
                                                                               \
        /* Check if this coordinate is within input bounds */                  \
        if (pos < 0 || pos >= (int)input_dims[d]) {                            \
          in_bounds = false;                                                   \
          break;                                                               \
        }                                                                      \
                                                                               \
        /* Adjust to input coordinates */                                      \
        in_coords[d] = (size_t)pos;                                            \
      }                                                                        \
                                                                               \
      /* If within bounds, copy from input to output */                        \
      if (in_bounds) {                                                         \
        /* Calculate index for input with strides */                           \
        size_t in_idx;                                                         \
        if (is_input_contiguous) {                                             \
          /* Calculate linear index for contiguous input */                    \
          in_idx = 0;                                                          \
          size_t stride = 1;                                                   \
          for (int d = num_dims - 1; d >= 0; --d) {                            \
            in_idx += in_coords[d] * stride;                                   \
            stride *= input_dims[d];                                           \
          }                                                                    \
        } else {                                                               \
          /* Calculate strided index */                                        \
          in_idx = 0;                                                          \
          for (size_t d = 0; d < num_dims; d++) {                              \
            in_idx += in_coords[d] * input_strides[d];                         \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* Copy value from input to output, including offset */                \
        if (in_idx < num_els_in) {                                             \
          out[i] = inp[input_offset + in_idx];                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(size_t num_els_in, size_t num_els_out,        \
                                 size_t num_dims, const size_t *metadata,      \
                                 const TYPENAME *inp, TYPENAME *out,           \
                                 TYPENAME pad_value) {                         \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els_out + block_dim.x - 1) / block_dim.x);              \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els_in, num_els_out, num_dims, metadata, inp, out, pad_value);     \
  }

#define PAD_WITH_REFLECTION_OP(TYPENAME, FN_NAME)                              \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els_in, const size_t num_els_out,                       \
      const size_t num_dims, const size_t *metadata, const TYPENAME *inp,      \
      TYPENAME *out) {                                                         \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;     \
    const size_t *output_dims = metadata + 2 * num_dims + 1;                   \
    const size_t *paddings = metadata + 3 * num_dims + 1;                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    bool is_input_contiguous =                                                 \
        is_contiguous(num_dims, input_dims, input_strides);                    \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
                                                                               \
      /* Calculate coordinates in output tensor */                             \
      size_t out_coords[MAX_DIMS];                                             \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        out_coords[d] = tmp_i % output_dims[d];                                \
        tmp_i /= output_dims[d];                                               \
      }                                                                        \
                                                                               \
      /* Calculate corresponding coordinates in input tensor with reflection   \
       */                                                                      \
      size_t in_coords[MAX_DIMS];                                              \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t pad_before = paddings[d * 2];                                   \
        size_t dim_size = input_dims[d];                                       \
                                                                               \
        /* Get position relative to the padded area */                         \
        int pos =                                                              \
            static_cast<int>(out_coords[d]) - static_cast<int>(pad_before);    \
                                                                               \
        /* Apply correct reflection padding */                                 \
        /* For an input [1,2,3,4] with pad=2, we want: [3,2,1,2,3,4,3,2] */    \
                                                                               \
        /* Basic reflection algorithm */                                       \
        if (pos < 0) {                                                         \
          /* Reflection for positions before array start */                    \
          pos = -pos; /* First reflection at 0 */                              \
        } else if (pos >= static_cast<int>(dim_size)) {                        \
          /* Reflection for positions after array end */                       \
          pos = 2 * static_cast<int>(dim_size) - pos -                         \
                2; /* Reflect at (dim_size-1) */                               \
        }                                                                      \
                                                                               \
        /* Handle multiple reflections if needed */                            \
        while (pos < 0 || pos >= static_cast<int>(dim_size)) {                 \
          if (pos < 0) {                                                       \
            pos = -pos; /* Reflect at 0 */                                     \
          } else if (pos >= static_cast<int>(dim_size)) {                      \
            pos = 2 * static_cast<int>(dim_size) - pos -                       \
                  2; /* Reflect at (dim_size-1) */                             \
          }                                                                    \
        }                                                                      \
                                                                               \
        in_coords[d] = static_cast<size_t>(pos);                               \
      }                                                                        \
                                                                               \
      /* Calculate index for input with strides */                             \
      size_t in_idx;                                                           \
      if (is_input_contiguous) {                                               \
        /* Calculate linear index for contiguous input */                      \
        in_idx = 0;                                                            \
        size_t stride = 1;                                                     \
        for (int d = num_dims - 1; d >= 0; --d) {                              \
          in_idx += in_coords[d] * stride;                                     \
          stride *= input_dims[d];                                             \
        }                                                                      \
      } else {                                                                 \
        /* Calculate strided index */                                          \
        in_idx = 0;                                                            \
        for (size_t d = 0; d < num_dims; d++) {                                \
          in_idx += in_coords[d] * input_strides[d];                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Copy value from input to output, including offset */                  \
      if (in_idx < num_els_in) {                                               \
        out[i] = inp[input_offset + in_idx];                                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(size_t num_els_in, size_t num_els_out,        \
                                 size_t num_dims, const size_t *metadata,      \
                                 const TYPENAME *inp, TYPENAME *out) {         \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els_out + block_dim.x - 1) / block_dim.x);              \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els_in, num_els_out, num_dims, metadata, inp, out);                \
  }

#define PAD_WITH_REPLICATION_OP(TYPENAME, FN_NAME)                             \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els_in, const size_t num_els_out,                       \
      const size_t num_dims, const size_t *metadata, const TYPENAME *inp,      \
      TYPENAME *out) {                                                         \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;     \
    const size_t *output_dims = metadata + 2 * num_dims + 1;                   \
    const size_t *paddings = metadata + 3 * num_dims + 1;                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    bool is_input_contiguous =                                                 \
        is_contiguous(num_dims, input_dims, input_strides);                    \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
                                                                               \
      /* Calculate coordinates in output tensor */                             \
      size_t out_coords[MAX_DIMS];                                             \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        out_coords[d] = tmp_i % output_dims[d];                                \
        tmp_i /= output_dims[d];                                               \
      }                                                                        \
                                                                               \
      /* Calculate corresponding coordinates in input tensor with replication  \
       */                                                                      \
      size_t in_coords[MAX_DIMS];                                              \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t pad_before = paddings[d * 2];                                   \
        long long pos = static_cast<long long>(out_coords[d]) -                \
                        static_cast<long long>(pad_before);                    \
                                                                               \
        /* Apply replication (clamp to valid range) */                         \
        pos = max(0LL, min(pos, static_cast<long long>(input_dims[d] - 1)));   \
                                                                               \
        in_coords[d] = static_cast<size_t>(pos);                               \
      }                                                                        \
                                                                               \
      /* Calculate index for input with strides */                             \
      size_t in_idx;                                                           \
      if (is_input_contiguous) {                                               \
        /* Calculate linear index for contiguous input */                      \
        in_idx = 0;                                                            \
        size_t stride = 1;                                                     \
        for (int d = num_dims - 1; d >= 0; --d) {                              \
          in_idx += in_coords[d] * stride;                                     \
          stride *= input_dims[d];                                             \
        }                                                                      \
      } else {                                                                 \
        /* Calculate strided index */                                          \
        in_idx = 0;                                                            \
        for (size_t d = 0; d < num_dims; d++) {                                \
          in_idx += in_coords[d] * input_strides[d];                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Copy value from input to output, including offset */                  \
      if (in_idx < num_els_in) {                                               \
        out[i] = inp[input_offset + in_idx];                                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(size_t num_els_in, size_t num_els_out,        \
                                 size_t num_dims, const size_t *metadata,      \
                                 const TYPENAME *inp, TYPENAME *out) {         \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els_out + block_dim.x - 1) / block_dim.x);              \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els_in, num_els_out, num_dims, metadata, inp, out);                \
  }

// Backward operations
#define PAD_WITH_CONSTANT_BACKWARD_OP(TYPENAME, FN_NAME)                       \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els_in, const size_t num_els_out,                       \
      const size_t num_dims, const size_t *metadata, const TYPENAME *grad_out, \
      TYPENAME *grad_in) {                                                     \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;     \
    const size_t *output_dims = metadata + 2 * num_dims + 1;                   \
    const size_t *paddings = metadata + 3 * num_dims + 1;                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    bool is_input_contiguous =                                                 \
        is_contiguous(num_dims, input_dims, input_strides);                    \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
                                                                               \
      /* Calculate coordinates in output tensor */                             \
      size_t out_coords[MAX_DIMS];                                             \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        out_coords[d] = tmp_i % output_dims[d];                                \
        tmp_i /= output_dims[d];                                               \
      }                                                                        \
                                                                               \
      /* Calculate corresponding coordinates in input tensor */                \
      bool in_bounds = true;                                                   \
      size_t in_coords[MAX_DIMS];                                              \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t pad_before = paddings[d * 2];                                   \
        int pos = (int)out_coords[d] - (int)pad_before;                        \
                                                                               \
        /* Check if this coordinate is within input bounds */                  \
        if (pos < 0 || pos >= (int)input_dims[d]) {                            \
          in_bounds = false;                                                   \
          break;                                                               \
        }                                                                      \
                                                                               \
        /* Adjust to input coordinates */                                      \
        in_coords[d] = (size_t)pos;                                            \
      }                                                                        \
                                                                               \
      /* If within bounds, accumulate gradient */                              \
      if (in_bounds) {                                                         \
        /* Calculate index for input with strides */                           \
        size_t in_idx;                                                         \
        if (is_input_contiguous) {                                             \
          /* Calculate linear index for contiguous input */                    \
          in_idx = 0;                                                          \
          size_t stride = 1;                                                   \
          for (int d = num_dims - 1; d >= 0; --d) {                            \
            in_idx += in_coords[d] * stride;                                   \
            stride *= input_dims[d];                                           \
          }                                                                    \
        } else {                                                               \
          /* Calculate strided index */                                        \
          in_idx = 0;                                                          \
          for (size_t d = 0; d < num_dims; d++) {                              \
            in_idx += in_coords[d] * input_strides[d];                         \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* Accumulate gradient */                                              \
        if (in_idx < num_els_in) {                                             \
          atomicAdd(&grad_in[in_idx], grad_out[i]);                            \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els_in, size_t num_els_out, size_t num_dims,                  \
      const size_t *metadata, const TYPENAME *grad_out, TYPENAME *grad_in) {   \
    /* Initialize grad_in to zeros */                                          \
    cudaMemset(grad_in, 0, num_els_in * sizeof(TYPENAME));                     \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els_out + block_dim.x - 1) / block_dim.x);              \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els_in, num_els_out, num_dims, metadata, grad_out, grad_in);       \
  }

#define PAD_WITH_REFLECTION_BACKWARD_OP(TYPENAME, FN_NAME)                     \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els_in, const size_t num_els_out,                       \
      const size_t num_dims, const size_t *metadata, const TYPENAME *grad_out, \
      TYPENAME *grad_in) {                                                     \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;     \
    const size_t *output_dims = metadata + 2 * num_dims + 1;                   \
    const size_t *paddings = metadata + 3 * num_dims + 1;                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    bool is_input_contiguous =                                                 \
        is_contiguous(num_dims, input_dims, input_strides);                    \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
                                                                               \
      /* Calculate coordinates in output tensor */                             \
      size_t out_coords[MAX_DIMS];                                             \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        out_coords[d] = tmp_i % output_dims[d];                                \
        tmp_i /= output_dims[d];                                               \
      }                                                                        \
                                                                               \
      /* Calculate corresponding coordinates in input tensor with reflection   \
       */                                                                      \
      size_t in_coords[MAX_DIMS];                                              \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t pad_before = paddings[d * 2];                                   \
        size_t dim_size = input_dims[d];                                       \
                                                                               \
        /* Get position relative to the padded area */                         \
        int pos =                                                              \
            static_cast<int>(out_coords[d]) - static_cast<int>(pad_before);    \
                                                                               \
        /* Apply correct reflection padding */                                 \
        /* For an input [1,2,3,4] with pad=2, we want: [3,2,1,2,3,4,3,2] */    \
                                                                               \
        /* Basic reflection algorithm */                                       \
        if (pos < 0) {                                                         \
          /* Reflection for positions before array start */                    \
          pos = -pos; /* First reflection at 0 */                              \
        } else if (pos >= static_cast<int>(dim_size)) {                        \
          /* Reflection for positions after array end */                       \
          pos = 2 * static_cast<int>(dim_size) - pos -                         \
                2; /* Reflect at (dim_size-1) */                               \
        }                                                                      \
                                                                               \
        /* Handle multiple reflections if needed */                            \
        while (pos < 0 || pos >= static_cast<int>(dim_size)) {                 \
          if (pos < 0) {                                                       \
            pos = -pos; /* Reflect at 0 */                                     \
          } else if (pos >= static_cast<int>(dim_size)) {                      \
            pos = 2 * static_cast<int>(dim_size) - pos -                       \
                  2; /* Reflect at (dim_size-1) */                             \
          }                                                                    \
        }                                                                      \
                                                                               \
        in_coords[d] = static_cast<size_t>(pos);                               \
      }                                                                        \
                                                                               \
      /* Calculate index for input with strides */                             \
      size_t in_idx;                                                           \
      if (is_input_contiguous) {                                               \
        /* Calculate linear index for contiguous input */                      \
        in_idx = 0;                                                            \
        size_t stride = 1;                                                     \
        for (int d = num_dims - 1; d >= 0; --d) {                              \
          in_idx += in_coords[d] * stride;                                     \
          stride *= input_dims[d];                                             \
        }                                                                      \
      } else {                                                                 \
        /* Calculate strided index */                                          \
        in_idx = 0;                                                            \
        for (size_t d = 0; d < num_dims; d++) {                                \
          in_idx += in_coords[d] * input_strides[d];                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Accumulate gradient */                                                \
      if (in_idx < num_els_in) {                                               \
        atomicAdd(&grad_in[in_idx], grad_out[i]);                              \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els_in, size_t num_els_out, size_t num_dims,                  \
      const size_t *metadata, const TYPENAME *grad_out, TYPENAME *grad_in) {   \
    /* Initialize grad_in to zeros */                                          \
    cudaMemset(grad_in, 0, num_els_in * sizeof(TYPENAME));                     \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els_out + block_dim.x - 1) / block_dim.x);              \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els_in, num_els_out, num_dims, metadata, grad_out, grad_in);       \
  }

#define PAD_WITH_REPLICATION_BACKWARD_OP(TYPENAME, FN_NAME)                    \
  extern "C" __global__ void cuda_##FN_NAME##_kernel(                          \
      const size_t num_els_in, const size_t num_els_out,                       \
      const size_t num_dims, const size_t *metadata, const TYPENAME *grad_out, \
      TYPENAME *grad_in) {                                                     \
    const size_t *input_dims = metadata;                                       \
    const size_t *input_strides = metadata + num_dims;                         \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;     \
    const size_t *output_dims = metadata + 2 * num_dims + 1;                   \
    const size_t *paddings = metadata + 3 * num_dims + 1;                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    bool is_input_contiguous =                                                 \
        is_contiguous(num_dims, input_dims, input_strides);                    \
                                                                               \
    /* Process elements in parallel */                                         \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;               \
         i < num_els_out; i += blockDim.x * gridDim.x) {                       \
                                                                               \
      /* Calculate coordinates in output tensor */                             \
      size_t out_coords[MAX_DIMS];                                             \
      size_t tmp_i = i;                                                        \
      for (int d = num_dims - 1; d >= 0; --d) {                                \
        out_coords[d] = tmp_i % output_dims[d];                                \
        tmp_i /= output_dims[d];                                               \
      }                                                                        \
                                                                               \
      /* Calculate corresponding coordinates in input tensor with replication  \
       */                                                                      \
      size_t in_coords[MAX_DIMS];                                              \
                                                                               \
      for (size_t d = 0; d < num_dims; d++) {                                  \
        size_t pad_before = paddings[d * 2];                                   \
        long long pos = static_cast<long long>(out_coords[d]) -                \
                        static_cast<long long>(pad_before);                    \
                                                                               \
        /* Apply replication (clamp to valid range) */                         \
        pos = max(0LL, min(pos, static_cast<long long>(input_dims[d] - 1)));   \
                                                                               \
        in_coords[d] = static_cast<size_t>(pos);                               \
      }                                                                        \
                                                                               \
      /* Calculate index for input with strides */                             \
      size_t in_idx;                                                           \
      if (is_input_contiguous) {                                               \
        /* Calculate linear index for contiguous input */                      \
        in_idx = 0;                                                            \
        size_t stride = 1;                                                     \
        for (int d = num_dims - 1; d >= 0; --d) {                              \
          in_idx += in_coords[d] * stride;                                     \
          stride *= input_dims[d];                                             \
        }                                                                      \
      } else {                                                                 \
        /* Calculate strided index */                                          \
        in_idx = 0;                                                            \
        for (size_t d = 0; d < num_dims; d++) {                                \
          in_idx += in_coords[d] * input_strides[d];                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* Accumulate gradient */                                                \
      if (in_idx < num_els_in) {                                               \
        atomicAdd(&grad_in[in_idx], grad_out[i]);                              \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_##FN_NAME(                                              \
      size_t num_els_in, size_t num_els_out, size_t num_dims,                  \
      const size_t *metadata, const TYPENAME *grad_out, TYPENAME *grad_in) {   \
    /* Initialize grad_in to zeros */                                          \
    cudaMemset(grad_in, 0, num_els_in * sizeof(TYPENAME));                     \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els_out + block_dim.x - 1) / block_dim.x);              \
    cuda_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(                          \
        num_els_in, num_els_out, num_dims, metadata, grad_out, grad_in);       \
  }

PAD_WITH_CONSTANT_OP(float, pad_with_constant_f32);
PAD_WITH_CONSTANT_OP(double, pad_with_constant_f64);
PAD_WITH_CONSTANT_OP(uint8_t, pad_with_constant_u8);
PAD_WITH_CONSTANT_OP(uint16_t, pad_with_constant_u16);
PAD_WITH_CONSTANT_OP(uint32_t, pad_with_constant_u32);
PAD_WITH_CONSTANT_OP(uint64_t, pad_with_constant_u64);
PAD_WITH_CONSTANT_OP(int8_t, pad_with_constant_i8);
PAD_WITH_CONSTANT_OP(int16_t, pad_with_constant_i16);
PAD_WITH_CONSTANT_OP(int32_t, pad_with_constant_i32);
PAD_WITH_CONSTANT_OP(int64_t, pad_with_constant_i64);

PAD_WITH_REFLECTION_OP(float, pad_with_reflection_f32);
PAD_WITH_REFLECTION_OP(double, pad_with_reflection_f64);
PAD_WITH_REFLECTION_OP(uint8_t, pad_with_reflection_u8);
PAD_WITH_REFLECTION_OP(uint16_t, pad_with_reflection_u16);
PAD_WITH_REFLECTION_OP(uint32_t, pad_with_reflection_u32);
PAD_WITH_REFLECTION_OP(uint64_t, pad_with_reflection_u64);
PAD_WITH_REFLECTION_OP(int8_t, pad_with_reflection_i8);
PAD_WITH_REFLECTION_OP(int16_t, pad_with_reflection_i16);
PAD_WITH_REFLECTION_OP(int32_t, pad_with_reflection_i32);
PAD_WITH_REFLECTION_OP(int64_t, pad_with_reflection_i64);

PAD_WITH_REPLICATION_OP(float, pad_with_replication_f32);
PAD_WITH_REPLICATION_OP(double, pad_with_replication_f64);
PAD_WITH_REPLICATION_OP(uint8_t, pad_with_replication_u8);
PAD_WITH_REPLICATION_OP(uint16_t, pad_with_replication_u16);
PAD_WITH_REPLICATION_OP(uint32_t, pad_with_replication_u32);
PAD_WITH_REPLICATION_OP(uint64_t, pad_with_replication_u64);
PAD_WITH_REPLICATION_OP(int8_t, pad_with_replication_i8);
PAD_WITH_REPLICATION_OP(int16_t, pad_with_replication_i16);
PAD_WITH_REPLICATION_OP(int32_t, pad_with_replication_i32);
PAD_WITH_REPLICATION_OP(int64_t, pad_with_replication_i64);

PAD_WITH_CONSTANT_BACKWARD_OP(float, pad_with_constant_backward_f32);
PAD_WITH_CONSTANT_BACKWARD_OP(double, pad_with_constant_backward_f64);
PAD_WITH_CONSTANT_BACKWARD_OP(uint8_t, pad_with_constant_backward_u8);
PAD_WITH_CONSTANT_BACKWARD_OP(uint16_t, pad_with_constant_backward_u16);
PAD_WITH_CONSTANT_BACKWARD_OP(uint32_t, pad_with_constant_backward_u32);
PAD_WITH_CONSTANT_BACKWARD_OP(uint64_t, pad_with_constant_backward_u64);
PAD_WITH_CONSTANT_BACKWARD_OP(int8_t, pad_with_constant_backward_i8);
PAD_WITH_CONSTANT_BACKWARD_OP(int16_t, pad_with_constant_backward_i16);
PAD_WITH_CONSTANT_BACKWARD_OP(int32_t, pad_with_constant_backward_i32);
PAD_WITH_CONSTANT_BACKWARD_OP(int64_t, pad_with_constant_backward_i64);

PAD_WITH_REFLECTION_BACKWARD_OP(float, pad_with_reflection_backward_f32);
PAD_WITH_REFLECTION_BACKWARD_OP(double, pad_with_reflection_backward_f64);
PAD_WITH_REFLECTION_BACKWARD_OP(uint8_t, pad_with_reflection_backward_u8);
PAD_WITH_REFLECTION_BACKWARD_OP(uint16_t, pad_with_reflection_backward_u16);
PAD_WITH_REFLECTION_BACKWARD_OP(uint32_t, pad_with_reflection_backward_u32);
PAD_WITH_REFLECTION_BACKWARD_OP(uint64_t, pad_with_reflection_backward_u64);
PAD_WITH_REFLECTION_BACKWARD_OP(int8_t, pad_with_reflection_backward_i8);
PAD_WITH_REFLECTION_BACKWARD_OP(int16_t, pad_with_reflection_backward_i16);
PAD_WITH_REFLECTION_BACKWARD_OP(int32_t, pad_with_reflection_backward_i32);
PAD_WITH_REFLECTION_BACKWARD_OP(int64_t, pad_with_reflection_backward_i64);

PAD_WITH_REPLICATION_BACKWARD_OP(float, pad_with_replication_backward_f32);
PAD_WITH_REPLICATION_BACKWARD_OP(double, pad_with_replication_backward_f64);
PAD_WITH_REPLICATION_BACKWARD_OP(uint8_t, pad_with_replication_backward_u8);
PAD_WITH_REPLICATION_BACKWARD_OP(uint16_t, pad_with_replication_backward_u16);
PAD_WITH_REPLICATION_BACKWARD_OP(uint32_t, pad_with_replication_backward_u32);
PAD_WITH_REPLICATION_BACKWARD_OP(uint64_t, pad_with_replication_backward_u64);
PAD_WITH_REPLICATION_BACKWARD_OP(int8_t, pad_with_replication_backward_i8);
PAD_WITH_REPLICATION_BACKWARD_OP(int16_t, pad_with_replication_backward_i16);
PAD_WITH_REPLICATION_BACKWARD_OP(int32_t, pad_with_replication_backward_i32);
PAD_WITH_REPLICATION_BACKWARD_OP(int64_t, pad_with_replication_backward_i64);

// __half operations
PAD_WITH_CONSTANT_OP(__half, pad_with_constant_f16);
PAD_WITH_REFLECTION_OP(__half, pad_with_reflection_f16);
PAD_WITH_REPLICATION_OP(__half, pad_with_replication_f16);
PAD_WITH_CONSTANT_BACKWARD_OP(__half, pad_with_constant_backward_f16);
PAD_WITH_REFLECTION_BACKWARD_OP(__half, pad_with_reflection_backward_f16);
PAD_WITH_REPLICATION_BACKWARD_OP(__half, pad_with_replication_backward_f16);

// __nv_bfloat16 operations
PAD_WITH_CONSTANT_OP(__nv_bfloat16, pad_with_constant_bf16);
PAD_WITH_REFLECTION_OP(__nv_bfloat16, pad_with_reflection_bf16);
PAD_WITH_REPLICATION_OP(__nv_bfloat16, pad_with_replication_bf16);
PAD_WITH_CONSTANT_BACKWARD_OP(__nv_bfloat16, pad_with_constant_backward_bf16);
PAD_WITH_REFLECTION_BACKWARD_OP(__nv_bfloat16,
                                pad_with_reflection_backward_bf16);
PAD_WITH_REPLICATION_BACKWARD_OP(__nv_bfloat16,
                                 pad_with_replication_backward_bf16);
