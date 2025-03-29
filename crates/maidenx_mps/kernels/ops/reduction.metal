#include "../atomics.metal"
#include "../metal_utils.metal"
#include <metal_stdlib>
#include <metal_math>
#include <metal_atomic>

using namespace metal;

#define MAX_DIMS 10

// Helper kernel to initialize output arrays
template <typename T>
kernel void fill_kernel(device T *data [[buffer(0)]],
                        constant T& value [[buffer(1)]],
                        constant size_t& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
  if (id < size) {
    data[id] = value;
  }
}

#define SUM_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)                             \
  kernel void metal_##FN_NAME##_kernel(                                        \
      device TYPENAME *out [[buffer(0)]],                                      \
      const device TYPENAME *inp [[buffer(1)]],                                \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t& num_sum_dims [[buffer(4)]],                             \
      constant size_t *metadata [[buffer(5)]],                                 \
      uint id [[thread_position_in_grid]])                                     \
  {                                                                            \
    if (id >= num_els) return;                                                 \
                                                                               \
    const constant size_t *dims = metadata;                                    \
    const constant size_t *strides = metadata + num_dims;                      \
    const constant size_t *sum_dims_l = metadata + 2 * num_dims;               \
    const constant size_t *sum_dims_s = metadata + 2 * num_dims + num_sum_dims;\
    const size_t offset = *(metadata + 2 * num_dims + 2 * num_sum_dims);       \
                                                                               \
    size_t src_idx;                                                            \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      src_idx = offset + id;                                                   \
    } else {                                                                   \
      src_idx = offset + get_strided_index(id, num_dims, dims, strides);       \
    }                                                                          \
                                                                               \
    size_t dst_index = id;                                                     \
    for (uint nd = 0; nd < num_sum_dims; ++nd) {                               \
      size_t stride = sum_dims_s[nd];                                          \
      size_t pre = dst_index / stride;                                         \
      size_t post = dst_index % stride;                                        \
      dst_index = (pre / sum_dims_l[nd]) * stride + post;                      \
    }                                                                          \
                                                                               \
    (void)ATOMIC_ADD_FUNC(out + dst_index, inp[src_idx]);                      \
  }

#define SUM_TO_SHAPE_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)                    \
  kernel void metal_##FN_NAME##_kernel(                                        \
      device TYPENAME *out [[buffer(0)]],                                      \
      const device TYPENAME *inp [[buffer(1)]],                                \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t *metadata [[buffer(4)]],                                 \
      uint id [[thread_position_in_grid]])                                     \
  {                                                                            \
    if (id >= num_els) return;                                                 \
                                                                               \
    const constant size_t *input_dims = metadata;                              \
    const constant size_t *input_strides = metadata + num_dims;                \
    const constant size_t *output_dims = metadata + 2 * num_dims;              \
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
    size_t coords[MAX_DIMS];                                                   \
    size_t tmp_i = id;                                                         \
    for (int d = num_dims - 1; d >= 0; --d) {                                  \
      coords[d] = tmp_i % input_dims[d];                                       \
      tmp_i /= input_dims[d];                                                  \
    }                                                                          \
                                                                               \
    size_t dst_idx = 0;                                                        \
    for (size_t d = 0; d < num_dims; d++) {                                    \
      size_t out_coord = coords[d] / reduction_factors[d];                     \
      dst_idx = dst_idx * output_dims[d] + out_coord;                          \
    }                                                                          \
                                                                               \
    size_t src_idx = offset;                                                   \
    for (size_t d = 0; d < num_dims; d++) {                                    \
      src_idx += coords[d] * input_strides[d];                                 \
    }                                                                          \
                                                                               \
    (void)ATOMIC_ADD_FUNC(out + dst_idx, inp[src_idx]);                        \
  }

#define MEAN_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)                            \
  kernel void metal_##FN_NAME##_kernel(                                        \
      device TYPENAME *out [[buffer(0)]],                                      \
      const device TYPENAME *inp [[buffer(1)]],                                \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t& num_mean_dims [[buffer(4)]],                            \
      constant size_t *metadata [[buffer(5)]],                                 \
      uint id [[thread_position_in_grid]])                                     \
  {                                                                            \
    if (id >= num_els) return;                                                 \
                                                                               \
    const constant size_t *dims = metadata;                                    \
    const constant size_t *strides = metadata + num_dims;                      \
    const constant size_t *mean_dims_l = metadata + 2 * num_dims;              \
    const constant size_t *mean_dims_s = metadata + 2 * num_dims + num_mean_dims;   \
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
      size_t dst_index = id;                                                   \
      for (uint nd = 0; nd < num_mean_dims; ++nd) {                            \
        size_t stride = mean_dims_s[nd];                                       \
        size_t pre = dst_index / stride;                                       \
        size_t post = dst_index % stride;                                      \
        dst_index = (pre / mean_dims_l[nd]) * stride + post;                   \
      }                                                                        \
      (void)ATOMIC_ADD_FUNC(out + dst_index, inp[offset + id] / factor);       \
    } else {                                                                   \
      uint strided_i =                                                         \
          offset + get_strided_index(id, num_dims, dims, strides);             \
      size_t dst_index = id;                                                   \
      for (uint nd = 0; nd < num_mean_dims; ++nd) {                            \
        size_t stride = mean_dims_s[nd];                                       \
        size_t pre = dst_index / stride;                                       \
        size_t post = dst_index % stride;                                      \
        dst_index = (pre / mean_dims_l[nd]) * stride + post;                   \
      }                                                                        \
      (void)ATOMIC_ADD_FUNC(out + dst_index, inp[strided_i] / factor);         \
    }                                                                          \
  }

#define FOLD_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)                            \
  kernel void metal_##FN_NAME##_kernel(                                        \
      device TYPENAME *out [[buffer(0)]],                                      \
      const device TYPENAME *inp [[buffer(1)]],                                \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t *metadata [[buffer(4)]],                                 \
      uint id [[thread_position_in_grid]])                                     \
  {                                                                            \
    if (id >= num_els) return;                                                 \
                                                                               \
    const constant size_t *input_dims = metadata;                              \
    const constant size_t *input_strides = metadata + num_dims;                \
    const size_t fold_dim = *(metadata + 2 * num_dims);                        \
    const size_t window_dim = *(metadata + 2 * num_dims + 1);                  \
    const size_t fold_size = *(metadata + 2 * num_dims + 2);                   \
    const size_t step = *(metadata + 2 * num_dims + 3);                        \
    /*const size_t window_size = *(metadata + 2 * num_dims + 4);*/             \
    const size_t offset = *(metadata + 2 * num_dims + 5);                      \
                                                                               \
    if (num_dims > MAX_DIMS)                                                   \
      return;                                                                  \
                                                                               \
    /* Calculate coordinates in input tensor */                                \
    size_t coords[MAX_DIMS];                                                   \
    size_t tmp_i = id;                                                         \
    for (int d = num_dims - 1; d >= 0; --d) {                                  \
      coords[d] = tmp_i % input_dims[d];                                       \
      tmp_i /= input_dims[d];                                                  \
    }                                                                          \
                                                                               \
    /* Calculate source index using input strides */                           \
    size_t src_idx = offset;                                                   \
    for (size_t d = 0; d < num_dims; d++) {                                    \
      src_idx += coords[d] * input_strides[d];                                 \
    }                                                                          \
                                                                               \
    /* Extract window index and position in window */                          \
    const size_t window_idx = coords[fold_dim];                                \
    const size_t pos_in_window = coords[window_dim];                           \
                                                                               \
    /* Calculate position in the original folded dimension */                  \
    const size_t orig_pos = window_idx * step + pos_in_window;                 \
                                                                               \
    /* Skip if outside the bounds of the folded dimension */                   \
    if (orig_pos >= fold_size) {                                               \
      return;                                                                  \
    }                                                                          \
                                                                               \
    /* Calculate destination index in output */                                \
    size_t dst_idx = 0;                                                        \
                                                                               \
    for (size_t d = 0; d < num_dims; d++) {                                    \
      if (d == window_dim) {                                                   \
        continue; /* Skip window dimension */                                  \
      } else if (d == fold_dim) {                                              \
        dst_idx = dst_idx * fold_size + orig_pos;                              \
      } else {                                                                 \
        dst_idx = dst_idx * input_dims[d] + coords[d];                         \
      }                                                                        \
    }                                                                          \
                                                                               \
    /* Add value to output */                                                  \
    (void)ATOMIC_ADD_FUNC(out + dst_idx, inp[src_idx]);                        \
  }

#define MAX_OP(TYPENAME, FN_NAME, MIN_VALUE, ATOMIC_MAX_FUNC)                  \
  kernel void metal_##FN_NAME##_kernel(                                        \
      device TYPENAME *out [[buffer(0)]],                                      \
      const device TYPENAME *inp [[buffer(1)]],                                \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t& num_max_dims [[buffer(4)]],                             \
      constant size_t *metadata [[buffer(5)]],                                 \
      uint id [[thread_position_in_grid]])                                     \
  {                                                                            \
    if (id >= num_els) return;                                                 \
                                                                               \
    if (id < num_els) {                                                        \
        out[id] = MIN_VALUE;                                                   \
    }                                                                          \
                                                                               \
    const constant size_t *dims = metadata;                                    \
    const constant size_t *strides = metadata + num_dims;                      \
    const constant size_t *max_dims_l = metadata + 2 * num_dims;               \
    const constant size_t *max_dims_s = metadata + 2 * num_dims + num_max_dims;\
    const size_t offset = *(metadata + 2 * num_dims + 2 * num_max_dims);       \
                                                                               \
    size_t src_idx;                                                            \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      src_idx = id;                                                            \
    } else {                                                                   \
      src_idx = get_strided_index(id, num_dims, dims, strides);                \
    }                                                                          \
    size_t value_idx = (src_idx + offset) % num_els;                           \
                                                                               \
    /* Calculate destination index */                                          \
    size_t dst_idx = id;                                                       \
    for (uint nd = 0; nd < num_max_dims; ++nd) {                               \
      size_t stride = max_dims_s[nd];                                          \
      size_t pre = dst_idx / stride;                                           \
      size_t post = dst_idx % stride;                                          \
      dst_idx = (pre / max_dims_l[nd]) * stride + post;                        \
    }                                                                          \
                                                                               \
    (void)ATOMIC_MAX_FUNC(out + dst_idx, inp[value_idx]);                      \
  }

#define MIN_OP(TYPENAME, FN_NAME, MAX_VALUE, ATOMIC_MIN_FUNC)                  \
  kernel void metal_##FN_NAME##_kernel(                                        \
      device TYPENAME *out [[buffer(0)]],                                      \
      const device TYPENAME *inp [[buffer(1)]],                                \
      constant size_t& num_els [[buffer(2)]],                                  \
      constant size_t& num_dims [[buffer(3)]],                                 \
      constant size_t& num_min_dims [[buffer(4)]],                             \
      constant size_t *metadata [[buffer(5)]],                                 \
      uint id [[thread_position_in_grid]])                                     \
  {                                                                            \
    if (id >= num_els) return;                                                 \
                                                                               \
    if (id < num_els) {                                                        \
        out[id] = MAX_VALUE;                                                   \
    }                                                                          \
                                                                               \
    const constant size_t *dims = metadata;                                    \
    const constant size_t *strides = metadata + num_dims;                      \
    const constant size_t *min_dims_l = metadata + 2 * num_dims;               \
    const constant size_t *min_dims_s = metadata + 2 * num_dims + num_min_dims;\
    const size_t offset = *(metadata + 2 * num_dims + 2 * num_min_dims);       \
                                                                               \
    bool is_cont = is_contiguous(num_dims, dims, strides);                     \
                                                                               \
    if (is_cont) {                                                             \
      size_t src_idx = id;                                                     \
      size_t src_value_idx = (src_idx + offset) % num_els;                     \
                                                                               \
      /* Calculate destination index */                                        \
      size_t dst_idx = id;                                                     \
      for (uint nd = 0; nd < num_min_dims; ++nd) {                             \
        size_t stride = min_dims_s[nd];                                        \
        size_t pre = dst_idx / stride;                                         \
        size_t post = dst_idx % stride;                                        \
        dst_idx = (pre / min_dims_l[nd]) * stride + post;                      \
      }                                                                        \
                                                                               \
      /* Update min value atomically */                                        \
      (void)ATOMIC_MIN_FUNC(out + dst_idx, inp[src_value_idx]);                \
    } else {                                                                   \
      size_t src_idx = get_strided_index(id, num_dims, dims, strides);         \
      size_t src_value_idx = (src_idx + offset) % num_els;                     \
                                                                               \
      /* Calculate destination index */                                        \
      size_t dst_idx = id;                                                     \
      for (uint nd = 0; nd < num_min_dims; ++nd) {                             \
        size_t stride = min_dims_s[nd];                                        \
        size_t pre = dst_idx / stride;                                         \
        size_t post = dst_idx % stride;                                        \
        dst_idx = (pre / min_dims_l[nd]) * stride + post;                      \
      }                                                                        \
                                                                               \
      /* Update min value atomically */                                        \
      (void)ATOMIC_MIN_FUNC(out + dst_idx, inp[src_value_idx]);                \
    }                                                                          \
  }

// Instantiate operations for different types
// Float 32
SUM_OP(float, sum_f32, atomic_add_float)
SUM_TO_SHAPE_OP(float, sum_to_shape_f32, atomic_add_float)
MEAN_OP(float, mean_f32, atomic_add_float)
FOLD_OP(float, fold_f32, atomic_add_float)
MAX_OP(float, max_f32, -INFINITY, atomic_max_float)
MIN_OP(float, min_f32, INFINITY, atomic_min_float)

// Half-precision operations
SUM_OP(half, sum_f16, atomic_add_half)
SUM_TO_SHAPE_OP(half, sum_to_shape_f16, atomic_add_half)
MEAN_OP(half, mean_f16, atomic_add_half)
FOLD_OP(half, fold_f16, atomic_add_half)
MAX_OP(half, max_f16, half(-INFINITY), atomic_max_half)
MIN_OP(half, min_f16, half(INFINITY), atomic_min_half)

// Brain float operations
SUM_OP(bfloat, sum_bf16, atomic_add_bfloat)
SUM_TO_SHAPE_OP(bfloat, sum_to_shape_bf16, atomic_add_bfloat)
MEAN_OP(bfloat, mean_bf16, atomic_add_bfloat)
FOLD_OP(bfloat, fold_bf16, atomic_add_bfloat)
MAX_OP(bfloat, max_bf16, bfloat(-65504.0f), [](device bfloat* addr, bfloat val) { 
    return atomic_max_bfloat(addr, val);
})
MIN_OP(bfloat, min_bf16, bfloat(65504.0f), [](device bfloat* addr, bfloat val) {
    return atomic_min_bfloat(addr, val);
})

// uint8_t operations
SUM_OP(uint8_t, sum_u8, atomic_add_uint8)
SUM_TO_SHAPE_OP(uint8_t, sum_to_shape_u8, atomic_add_uint8)
FOLD_OP(uint8_t, fold_u8, atomic_add_uint8)
MAX_OP(uint8_t, max_u8, 0, atomic_max_uint8)
MIN_OP(uint8_t, min_u8, 255, atomic_min_uint8)

// uint16_t operations
SUM_OP(uint16_t, sum_u16, atomic_add_uint16)
SUM_TO_SHAPE_OP(uint16_t, sum_to_shape_u16, atomic_add_uint16)
FOLD_OP(uint16_t, fold_u16, atomic_add_uint16)
MAX_OP(uint16_t, max_u16, 0, atomic_max_uint16)
MIN_OP(uint16_t, min_u16, 65535, atomic_min_uint16)


// uint32_t operations
SUM_OP(uint32_t, sum_u32, atomic_add_uint32)
SUM_TO_SHAPE_OP(uint32_t, sum_to_shape_u32, atomic_add_uint32)
FOLD_OP(uint32_t, fold_u32, atomic_add_uint32)
MAX_OP(uint32_t, max_u32, 0, atomic_max_uint32)
MIN_OP(uint32_t, min_u32, 4294967295u, atomic_min_uint32)

// int8_t operations
SUM_OP(int8_t, sum_i8, atomic_add_int8)
SUM_TO_SHAPE_OP(int8_t, sum_to_shape_i8, atomic_add_int8)
FOLD_OP(int8_t, fold_i8, atomic_add_int8)
MAX_OP(int8_t, max_i8, -128, atomic_max_int8)
MIN_OP(int8_t, min_i8, 127, atomic_min_int8)

// int16_t operations
SUM_OP(int16_t, sum_i16, atomic_add_int16)
SUM_TO_SHAPE_OP(int16_t, sum_to_shape_i16, atomic_add_int16)
FOLD_OP(int16_t, fold_i16, atomic_add_int16)
MAX_OP(int16_t, max_i16, -32768, atomic_max_int16)
MIN_OP(int16_t, min_i16, 32767, atomic_min_int16)

// int32_t operations
SUM_OP(int32_t, sum_i32, atomic_add_int32)
SUM_TO_SHAPE_OP(int32_t, sum_to_shape_i32, atomic_add_int32)
FOLD_OP(int32_t, fold_i32, atomic_add_int32)
MAX_OP(int32_t, max_i32, -2147483648, atomic_max_int32)
MIN_OP(int32_t, min_i32, 2147483647, atomic_min_int32)
