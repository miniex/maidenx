#include "../cuda_utils.cuh"
#include <stdint.h>

template <typename T> __device__ T get_zero() { return T(0); }

#define CONV2D_OP(TYPENAME, FN_NAME)                                           \
  extern "C" __global__ void cuda_conv2d_im2col_##FN_NAME##_kernel(            \
      const size_t num_els, const size_t *dims_and_strides,                    \
      const TYPENAME *input, TYPENAME *col) {                                  \
    const size_t *dims = dims_and_strides;                                     \
    const size_t *strides = dims_and_strides + 8;                              \
                                                                               \
    size_t batch_size = dims[0];                                               \
    size_t channels = dims[1];                                                 \
    size_t height = dims[2];                                                   \
    size_t width = dims[3];                                                    \
    size_t kernel_h = dims[4];                                                 \
    size_t kernel_w = dims[5];                                                 \
    size_t out_h = dims[6];                                                    \
    size_t out_w = dims[7];                                                    \
                                                                               \
    size_t pad_h = strides[0];                                                 \
    size_t pad_w = strides[1];                                                 \
    size_t stride_h = strides[2];                                              \
    size_t stride_w = strides[3];                                              \
                                                                               \
    size_t kernel_size = kernel_h * kernel_w;                                  \
    size_t output_size = out_h * out_w;                                        \
                                                                               \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;    \
         idx += blockDim.x * gridDim.x) {                                      \
      size_t w_out = idx % out_w;                                              \
      size_t h_out = (idx / out_w) % out_h;                                    \
      size_t c = (idx / (out_h * out_w)) % channels;                           \
      size_t b = idx / (channels * out_h * out_w);                             \
                                                                               \
      int h_in = (int)(h_out * stride_h) - (int)pad_h;                         \
      int w_in = (int)(w_out * stride_w) - (int)pad_w;                         \
                                                                               \
      size_t col_offset = (c * kernel_size) * output_size;                     \
      size_t input_offset = ((b * channels + c) * height) * width;             \
                                                                               \
      for (size_t kh = 0; kh < kernel_h; ++kh) {                               \
        for (size_t kw = 0; kw < kernel_w; ++kw) {                             \
          int h = h_in + kh;                                                   \
          int w = w_in + kw;                                                   \
                                                                               \
          size_t col_idx = ((b * channels * kernel_size + c * kernel_size +    \
                             kh * kernel_w + kw) *                             \
                                out_h +                                        \
                            h_out) *                                           \
                               out_w +                                         \
                           w_out;                                              \
                                                                               \
          col[col_idx] = (h >= 0 && h < height && w >= 0 && w < width)         \
                             ? input[input_offset + h * width + w]             \
                             : get_zero<TYPENAME>();                           \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void cuda_conv2d_col2im_##FN_NAME##_kernel(            \
      const size_t num_els, const size_t *dims_and_strides,                    \
      const TYPENAME *col, TYPENAME *output) {                                 \
    const size_t *dims = dims_and_strides;                                     \
    const size_t *strides = dims_and_strides + 8;                              \
                                                                               \
    size_t batch_size = dims[0];                                               \
    size_t channels = dims[1];                                                 \
    size_t height = dims[2];                                                   \
    size_t width = dims[3];                                                    \
    size_t kernel_h = dims[4];                                                 \
    size_t kernel_w = dims[5];                                                 \
    size_t in_h = dims[6];                                                     \
    size_t in_w = dims[7];                                                     \
                                                                               \
    size_t pad_h = strides[0];                                                 \
    size_t pad_w = strides[1];                                                 \
    size_t stride_h = strides[2];                                              \
    size_t stride_w = strides[3];                                              \
                                                                               \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;    \
         idx += blockDim.x * gridDim.x) {                                      \
      size_t w = idx % width;                                                  \
      size_t h = (idx / width) % height;                                       \
      size_t c = (idx / (height * width)) % channels;                          \
      size_t b = idx / (channels * height * width);                            \
                                                                               \
      TYPENAME sum = 0;                                                        \
                                                                               \
      size_t w_pad = w + pad_w;                                                \
      size_t h_pad = h + pad_h;                                                \
                                                                               \
      size_t h_start =                                                         \
          (h_pad < kernel_h) ? 0 : (h_pad - kernel_h) / stride_h + 1;          \
      size_t w_start =                                                         \
          (w_pad < kernel_w) ? 0 : (w_pad - kernel_w) / stride_w + 1;          \
      size_t h_end = min(h_pad / stride_h + 1, in_h);                          \
      size_t w_end = min(w_pad / stride_w + 1, in_w);                          \
                                                                               \
      for (size_t h_out = h_start; h_out < h_end; ++h_out) {                   \
        for (size_t w_out = w_start; w_out < w_end; ++w_out) {                 \
          size_t kh = h_pad - h_out * stride_h;                                \
          size_t kw = w_pad - w_out * stride_w;                                \
                                                                               \
          if (kh < kernel_h && kw < kernel_w) {                                \
            size_t col_idx = ((b * channels * kernel_h * kernel_w +            \
                               c * kernel_h * kernel_w + kh * kernel_w + kw) * \
                                  in_h +                                       \
                              h_out) *                                         \
                                 in_w +                                        \
                             w_out;                                            \
            sum += col[col_idx];                                               \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      output[idx] = sum;                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_conv2d_im2col_##FN_NAME(                                \
      size_t num_els, const size_t *dims_and_strides, const TYPENAME *input,   \
      TYPENAME *col) {                                                         \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_conv2d_im2col_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(            \
        num_els, dims_and_strides, input, col);                                \
  }                                                                            \
                                                                               \
  extern "C" void cuda_conv2d_col2im_##FN_NAME(                                \
      size_t num_els, const size_t *dims_and_strides, const TYPENAME *col,     \
      TYPENAME *output) {                                                      \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_els + block_dim.x - 1) / block_dim.x);                  \
    cuda_conv2d_col2im_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(            \
        num_els, dims_and_strides, col, output);                               \
  }

CONV2D_OP(float, f32)
CONV2D_OP(double, f64)
CONV2D_OP(uint8_t, u8)
CONV2D_OP(uint32_t, u32)
CONV2D_OP(int32_t, i32)
CONV2D_OP(int64_t, i64)

// __half
CONV2D_OP(__half, f16)

// __nv_bfloat16
CONV2D_OP(__nv_bfloat16, bf16)
