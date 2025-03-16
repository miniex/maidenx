#include "../compatibility.cuh"
#include "../cuda_utils.cuh"
#include <stdint.h>

template <typename T> __device__ T get_zero() { return T(0); }

#define CONV2D_OP(TYPENAME, FN_NAME)                                           \
  extern "C" __global__ void cuda_conv2d_im2col_##FN_NAME##_kernel(            \
      const size_t num_col_elements, /* = B*C*kernel_h*kernel_w*out_h*out_w */ \
      const size_t *dims_and_strides,                                          \
      const TYPENAME *input, /* shape: [B, C, H, W] */                         \
      TYPENAME *col /* shape: [B, C, kernel_h, kernel_w, out_h, out_w] */      \
  ) {                                                                          \
    const size_t *dims = dims_and_strides;                                     \
    const size_t *strides = dims_and_strides + 8;                              \
                                                                               \
    size_t B = dims[0];                                                        \
    size_t C = dims[1];                                                        \
    size_t H = dims[2];                                                        \
    size_t W = dims[3];                                                        \
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
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (idx >= num_col_elements)                                               \
      return;                                                                  \
                                                                               \
    size_t ow = idx % out_w;                                                   \
    size_t tmp = idx / out_w;                                                  \
                                                                               \
    size_t oh = tmp % out_h;                                                   \
    tmp = tmp / out_h;                                                         \
                                                                               \
    size_t kw = tmp % kernel_w;                                                \
    tmp = tmp / kernel_w;                                                      \
                                                                               \
    size_t kh = tmp % kernel_h;                                                \
    tmp = tmp / kernel_h;                                                      \
                                                                               \
    size_t c = tmp % C;                                                        \
    size_t b = tmp / C;                                                        \
                                                                               \
    int h_in = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h) +     \
               static_cast<int>(kh);                                           \
    int w_in = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w) +     \
               static_cast<int>(kw);                                           \
                                                                               \
    TYPENAME val = get_zero<TYPENAME>();                                       \
    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {            \
      size_t input_offset =                                                    \
          ((b * C + c) * H + (size_t)h_in) * W + (size_t)w_in;                 \
      val = input[input_offset];                                               \
    }                                                                          \
                                                                               \
    col[idx] = val;                                                            \
  }                                                                            \
                                                                               \
  extern "C" void cuda_conv2d_im2col_##FN_NAME(                                \
      size_t num_col_elements, const size_t *dims_and_strides,                 \
      const TYPENAME *input, TYPENAME *col) {                                  \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_col_elements + block_dim.x - 1) / block_dim.x);         \
    cuda_conv2d_im2col_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(            \
        num_col_elements, dims_and_strides, input, col);                       \
  }                                                                            \
                                                                               \
  extern "C" __global__ void cuda_conv2d_col2im_##FN_NAME##_kernel(            \
      const size_t num_col_elements, /* = B*C*kernel_h*kernel_w*out_h*out_w */ \
      const size_t *dims_and_strides,                                          \
      const TYPENAME                                                           \
          *col,        /* shape: [B, C, kernel_h, kernel_w, out_h, out_w] */   \
      TYPENAME *output /* shape: [B, C, H, W] */                               \
  ) {                                                                          \
    const size_t *dims = dims_and_strides;                                     \
    const size_t *strides = dims_and_strides + 8;                              \
                                                                               \
    size_t B = dims[0];                                                        \
    size_t C = dims[1];                                                        \
    size_t H = dims[2];                                                        \
    size_t W = dims[3];                                                        \
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
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (idx >= num_col_elements)                                               \
      return;                                                                  \
                                                                               \
    size_t ow = idx % out_w;                                                   \
    size_t tmp = idx / out_w;                                                  \
                                                                               \
    size_t oh = tmp % out_h;                                                   \
    tmp = tmp / out_h;                                                         \
                                                                               \
    size_t kw = tmp % kernel_w;                                                \
    tmp = tmp / kernel_w;                                                      \
                                                                               \
    size_t kh = tmp % kernel_h;                                                \
    tmp = tmp / kernel_h;                                                      \
                                                                               \
    size_t c = tmp % C;                                                        \
    size_t b = tmp / C;                                                        \
                                                                               \
    int h_in = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h) +     \
               static_cast<int>(kh);                                           \
    int w_in = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w) +     \
               static_cast<int>(kw);                                           \
                                                                               \
    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {            \
      size_t out_offset = ((b * C + c) * H + (size_t)h_in) * W + (size_t)w_in; \
      TYPENAME val = col[idx];                                                 \
      atomicAdd(&output[out_offset], val);                                     \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void cuda_conv2d_col2im_##FN_NAME(                                \
      size_t num_col_elements, const size_t *dims_and_strides,                 \
      const TYPENAME *col, TYPENAME *output) {                                 \
    dim3 block_dim(256);                                                       \
    dim3 grid_dim((num_col_elements + block_dim.x - 1) / block_dim.x);         \
    cuda_conv2d_col2im_##FN_NAME##_kernel<<<grid_dim, block_dim>>>(            \
        num_col_elements, dims_and_strides, col, output);                      \
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
