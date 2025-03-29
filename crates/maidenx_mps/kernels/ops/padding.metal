#include "../metal_utils.metal"
#include "../atomics.metal"
#include <metal_stdlib>
#include <metal_math>
#include <metal_atomic>

using namespace metal;

#define MAX_DIMS 10

// Helper kernel to initialize output arrays with a constant value
template <typename T>
kernel void fill_kernel(device T *data [[buffer(0)]],
                        constant T& value [[buffer(1)]],
                        constant size_t& size [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
  if (id < size) {
    data[id] = value;
  }
}

#define PAD_WITH_CONSTANT_OP(TYPENAME, FN_NAME)                               \
  kernel void metal_##FN_NAME##_kernel(                                       \
      device TYPENAME *out [[buffer(0)]],                                     \
      const device TYPENAME *inp [[buffer(1)]],                               \
      constant size_t& num_els_in [[buffer(2)]],                              \
      constant size_t& num_els_out [[buffer(3)]],                             \
      constant size_t& num_dims [[buffer(4)]],                                \
      constant size_t *metadata [[buffer(5)]],                                \
      constant TYPENAME& pad_value [[buffer(6)]],                             \
      uint id [[thread_position_in_grid]]) {                                  \
    if (id >= num_els_out) return;                                            \
                                                                              \
    const constant size_t *input_dims = metadata;                             \
    const constant size_t *input_strides = metadata + num_dims;               \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;    \
    const constant size_t *output_dims = metadata + 2 * num_dims + 1;         \
    const constant size_t *paddings = metadata + 3 * num_dims + 1;            \
                                                                              \
    if (num_dims > MAX_DIMS)                                                  \
      return;                                                                 \
                                                                              \
    /* Initialize output with pad value */                                    \
    out[id] = pad_value;                                                      \
                                                                              \
    bool is_input_contiguous =                                                \
        is_contiguous(num_dims, input_dims, input_strides);                   \
                                                                              \
    /* Calculate coordinates in output tensor */                              \
    size_t out_coords[MAX_DIMS];                                              \
    size_t tmp_i = id;                                                        \
    for (int d = num_dims - 1; d >= 0; --d) {                                 \
      out_coords[d] = tmp_i % output_dims[d];                                 \
      tmp_i /= output_dims[d];                                                \
    }                                                                         \
                                                                              \
    /* Calculate corresponding coordinates in input tensor */                 \
    bool in_bounds = true;                                                    \
    size_t in_coords[MAX_DIMS];                                               \
                                                                              \
    for (size_t d = 0; d < num_dims; d++) {                                   \
      size_t pad_before = paddings[d * 2];                                    \
      int pos = (int)out_coords[d] - (int)pad_before;                         \
                                                                              \
      /* Check if this coordinate is within input bounds */                   \
      if (pos < 0 || pos >= (int)input_dims[d]) {                             \
        in_bounds = false;                                                    \
        break;                                                                \
      }                                                                       \
                                                                              \
      /* Adjust to input coordinates */                                       \
      in_coords[d] = (size_t)pos;                                             \
    }                                                                         \
                                                                              \
    /* If within bounds, copy from input to output */                         \
    if (in_bounds) {                                                          \
      /* Calculate index for input with strides */                            \
      size_t in_idx;                                                          \
      if (is_input_contiguous) {                                              \
        /* Calculate linear index for contiguous input */                     \
        in_idx = 0;                                                           \
        size_t stride = 1;                                                    \
        for (int d = num_dims - 1; d >= 0; --d) {                             \
          in_idx += in_coords[d] * stride;                                    \
          stride *= input_dims[d];                                            \
        }                                                                     \
      } else {                                                                \
        /* Calculate strided index */                                         \
        in_idx = 0;                                                           \
        for (size_t d = 0; d < num_dims; d++) {                               \
          in_idx += in_coords[d] * input_strides[d];                          \
        }                                                                     \
      }                                                                       \
                                                                              \
      /* Copy value from input to output, including offset */                 \
      if (in_idx < num_els_in) {                                              \
        out[id] = inp[input_offset + in_idx];                                 \
      }                                                                       \
    }                                                                         \
  }

#define PAD_WITH_REFLECTION_OP(TYPENAME, FN_NAME)                             \
  kernel void metal_##FN_NAME##_kernel(                                       \
      device TYPENAME *out [[buffer(0)]],                                     \
      const device TYPENAME *inp [[buffer(1)]],                               \
      constant size_t& num_els_in [[buffer(2)]],                              \
      constant size_t& num_els_out [[buffer(3)]],                             \
      constant size_t& num_dims [[buffer(4)]],                                \
      constant size_t *metadata [[buffer(5)]],                                \
      uint id [[thread_position_in_grid]]) {                                  \
    if (id >= num_els_out) return;                                            \
                                                                              \
    const constant size_t *input_dims = metadata;                             \
    const constant size_t *input_strides = metadata + num_dims;               \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;    \
    const constant size_t *output_dims = metadata + 2 * num_dims + 1;         \
    const constant size_t *paddings = metadata + 3 * num_dims + 1;            \
                                                                              \
    if (num_dims > MAX_DIMS)                                                  \
      return;                                                                 \
                                                                              \
    bool is_input_contiguous =                                                \
        is_contiguous(num_dims, input_dims, input_strides);                   \
                                                                              \
    /* Calculate coordinates in output tensor */                              \
    size_t out_coords[MAX_DIMS];                                              \
    size_t tmp_i = id;                                                        \
    for (int d = num_dims - 1; d >= 0; --d) {                                 \
      out_coords[d] = tmp_i % output_dims[d];                                 \
      tmp_i /= output_dims[d];                                                \
    }                                                                         \
                                                                              \
    /* Calculate corresponding coordinates in input tensor with reflection */ \
    size_t in_coords[MAX_DIMS];                                               \
                                                                              \
    for (size_t d = 0; d < num_dims; d++) {                                   \
      size_t pad_before = paddings[d * 2];                                    \
      size_t dim_size = input_dims[d];                                        \
                                                                              \
      /* Get position relative to the padded area */                          \
      int pos = static_cast<int>(out_coords[d]) -                             \
                static_cast<int>(pad_before);                                 \
                                                                              \
      /* Apply correct reflection padding */                                  \
      /* For an input [1,2,3,4] with pad=2, we want: [3,2,1,2,3,4,3,2] */     \
                                                                              \
      /* Basic reflection algorithm */                                        \
      if (pos < 0) {                                                          \
        /* Reflection for positions before array start */                     \
        pos = -pos; /* First reflection at 0 */                               \
      } else if (pos >= static_cast<int>(dim_size)) {                         \
        /* Reflection for positions after array end */                        \
        pos = 2 * static_cast<int>(dim_size) - pos -                          \
              2; /* Reflect at (dim_size-1) */                                \
      }                                                                       \
                                                                              \
      /* Handle multiple reflections if needed */                             \
      while (pos < 0 || pos >= static_cast<int>(dim_size)) {                  \
        if (pos < 0) {                                                        \
          pos = -pos; /* Reflect at 0 */                                      \
        } else if (pos >= static_cast<int>(dim_size)) {                       \
          pos = 2 * static_cast<int>(dim_size) - pos -                        \
                2; /* Reflect at (dim_size-1) */                              \
        }                                                                     \
      }                                                                       \
                                                                              \
      in_coords[d] = static_cast<size_t>(pos);                                \
    }                                                                         \
                                                                              \
    /* Calculate index for input with strides */                              \
    size_t in_idx;                                                            \
    if (is_input_contiguous) {                                                \
      /* Calculate linear index for contiguous input */                       \
      in_idx = 0;                                                             \
      size_t stride = 1;                                                      \
      for (int d = num_dims - 1; d >= 0; --d) {                               \
        in_idx += in_coords[d] * stride;                                      \
        stride *= input_dims[d];                                              \
      }                                                                       \
    } else {                                                                  \
      /* Calculate strided index */                                           \
      in_idx = 0;                                                             \
      for (size_t d = 0; d < num_dims; d++) {                                 \
        in_idx += in_coords[d] * input_strides[d];                            \
      }                                                                       \
    }                                                                         \
                                                                              \
    /* Copy value from input to output, including offset */                   \
    if (in_idx < num_els_in) {                                                \
      out[id] = inp[input_offset + in_idx];                                   \
    }                                                                         \
  }

#define PAD_WITH_REPLICATION_OP(TYPENAME, FN_NAME)                           \
  kernel void metal_##FN_NAME##_kernel(                                      \
      device TYPENAME *out [[buffer(0)]],                                    \
      const device TYPENAME *inp [[buffer(1)]],                              \
      constant size_t& num_els_in [[buffer(2)]],                             \
      constant size_t& num_els_out [[buffer(3)]],                            \
      constant size_t& num_dims [[buffer(4)]],                               \
      constant size_t *metadata [[buffer(5)]],                               \
      uint id [[thread_position_in_grid]]) {                                 \
    if (id >= num_els_out) return;                                           \
                                                                             \
    const constant size_t *input_dims = metadata;                            \
    const constant size_t *input_strides = metadata + num_dims;              \
    const size_t input_offset = metadata ? *(metadata + 2 * num_dims) : 0;   \
    const constant size_t *output_dims = metadata + 2 * num_dims + 1;        \
    const constant size_t *paddings = metadata + 3 * num_dims + 1;           \
                                                                             \
    if (num_dims > MAX_DIMS)                                                 \
      return;                                                                \
                                                                             \
    bool is_input_contiguous =                                               \
        is_contiguous(num_dims, input_dims, input_strides);                  \
                                                                             \
    /* Calculate coordinates in output tensor */                             \
    size_t out_coords[MAX_DIMS];                                             \
    size_t tmp_i = id;                                                       \
    for (int d = num_dims - 1; d >= 0; --d) {                                \
      out_coords[d] = tmp_i % output_dims[d];                                \
      tmp_i /= output_dims[d];                                               \
    }                                                                        \
                                                                             \
    /* Calculate corresponding coordinates in input tensor with replication */\
    size_t in_coords[MAX_DIMS];                                              \
                                                                             \
    for (size_t d = 0; d < num_dims; d++) {                                  \
      size_t pad_before = paddings[d * 2];                                   \
      long pos = static_cast<long>(out_coords[d]) -                          \
                 static_cast<long>(pad_before);                              \
                                                                             \
      /* Apply replication (clamp to valid range) */                         \
      pos = max(0L, min(pos, static_cast<long>(input_dims[d] - 1)));         \
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
      out[id] = inp[input_offset + in_idx];                                  \
    }                                                                        \
  }

// Backward operations - fixed to remove unused input_offset
#define PAD_WITH_CONSTANT_BACKWARD_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)     \
  kernel void metal_##FN_NAME##_kernel(                                       \
      device TYPENAME *grad_in [[buffer(0)]],                                 \
      const device TYPENAME *grad_out [[buffer(1)]],                          \
      constant size_t& num_els_in [[buffer(2)]],                              \
      constant size_t& num_els_out [[buffer(3)]],                             \
      constant size_t& num_dims [[buffer(4)]],                                \
      constant size_t *metadata [[buffer(5)]],                                \
      uint id [[thread_position_in_grid]]) {                                  \
    if (id >= num_els_out) return;                                            \
                                                                              \
    const constant size_t *input_dims = metadata;                             \
    const constant size_t *input_strides = metadata + num_dims;               \
    /* Removed unused input_offset variable */                                \
    const constant size_t *output_dims = metadata + 2 * num_dims + 1;         \
    const constant size_t *paddings = metadata + 3 * num_dims + 1;            \
                                                                              \
    if (num_dims > MAX_DIMS)                                                  \
      return;                                                                 \
                                                                              \
    bool is_input_contiguous =                                                \
        is_contiguous(num_dims, input_dims, input_strides);                   \
                                                                              \
    /* Calculate coordinates in output tensor */                              \
    size_t out_coords[MAX_DIMS];                                              \
    size_t tmp_i = id;                                                        \
    for (int d = num_dims - 1; d >= 0; --d) {                                 \
      out_coords[d] = tmp_i % output_dims[d];                                 \
      tmp_i /= output_dims[d];                                                \
    }                                                                         \
                                                                              \
    /* Calculate corresponding coordinates in input tensor */                 \
    bool in_bounds = true;                                                    \
    size_t in_coords[MAX_DIMS];                                               \
                                                                              \
    for (size_t d = 0; d < num_dims; d++) {                                   \
      size_t pad_before = paddings[d * 2];                                    \
      int pos = (int)out_coords[d] - (int)pad_before;                         \
                                                                              \
      /* Check if this coordinate is within input bounds */                   \
      if (pos < 0 || pos >= (int)input_dims[d]) {                             \
        in_bounds = false;                                                    \
        break;                                                                \
      }                                                                       \
                                                                              \
      /* Adjust to input coordinates */                                       \
      in_coords[d] = (size_t)pos;                                             \
    }                                                                         \
                                                                              \
    /* If within bounds, accumulate gradient */                               \
    if (in_bounds) {                                                          \
      /* Calculate index for input with strides */                            \
      size_t in_idx;                                                          \
      if (is_input_contiguous) {                                              \
        /* Calculate linear index for contiguous input */                     \
        in_idx = 0;                                                           \
        size_t stride = 1;                                                    \
        for (int d = num_dims - 1; d >= 0; --d) {                             \
          in_idx += in_coords[d] * stride;                                    \
          stride *= input_dims[d];                                            \
        }                                                                     \
      } else {                                                                \
        /* Calculate strided index */                                         \
        in_idx = 0;                                                           \
        for (size_t d = 0; d < num_dims; d++) {                               \
          in_idx += in_coords[d] * input_strides[d];                          \
        }                                                                     \
      }                                                                       \
                                                                              \
      /* Accumulate gradient */                                               \
      if (in_idx < num_els_in) {                                              \
        (void)ATOMIC_ADD_FUNC(&grad_in[in_idx], grad_out[id]);                \
      }                                                                       \
    }                                                                         \
  }

#define PAD_WITH_REFLECTION_BACKWARD_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)   \
  kernel void metal_##FN_NAME##_kernel(                                       \
      device TYPENAME *grad_in [[buffer(0)]],                                 \
      const device TYPENAME *grad_out [[buffer(1)]],                          \
      constant size_t& num_els_in [[buffer(2)]],                              \
      constant size_t& num_els_out [[buffer(3)]],                             \
      constant size_t& num_dims [[buffer(4)]],                                \
      constant size_t *metadata [[buffer(5)]],                                \
      uint id [[thread_position_in_grid]]) {                                  \
    if (id >= num_els_out) return;                                            \
                                                                              \
    const constant size_t *input_dims = metadata;                             \
    const constant size_t *input_strides = metadata + num_dims;               \
    /* Removed unused input_offset variable */                                \
    const constant size_t *output_dims = metadata + 2 * num_dims + 1;         \
    const constant size_t *paddings = metadata + 3 * num_dims + 1;            \
                                                                              \
    if (num_dims > MAX_DIMS)                                                  \
      return;                                                                 \
                                                                              \
    bool is_input_contiguous =                                                \
        is_contiguous(num_dims, input_dims, input_strides);                   \
                                                                              \
    /* Calculate coordinates in output tensor */                              \
    size_t out_coords[MAX_DIMS];                                              \
    size_t tmp_i = id;                                                        \
    for (int d = num_dims - 1; d >= 0; --d) {                                 \
      out_coords[d] = tmp_i % output_dims[d];                                 \
      tmp_i /= output_dims[d];                                                \
    }                                                                         \
                                                                              \
    /* Calculate corresponding coordinates in input tensor with reflection */ \
    size_t in_coords[MAX_DIMS];                                               \
                                                                              \
    for (size_t d = 0; d < num_dims; d++) {                                   \
      size_t pad_before = paddings[d * 2];                                    \
      size_t dim_size = input_dims[d];                                        \
                                                                              \
      /* Get position relative to the padded area */                          \
      int pos = static_cast<int>(out_coords[d]) -                             \
                static_cast<int>(pad_before);                                 \
                                                                              \
      /* Apply correct reflection padding */                                  \
      /* For an input [1,2,3,4] with pad=2, we want: [3,2,1,2,3,4,3,2] */     \
                                                                              \
      /* Basic reflection algorithm */                                        \
      if (pos < 0) {                                                          \
        /* Reflection for positions before array start */                     \
        pos = -pos; /* First reflection at 0 */                               \
      } else if (pos >= static_cast<int>(dim_size)) {                         \
        /* Reflection for positions after array end */                        \
        pos = 2 * static_cast<int>(dim_size) - pos -                          \
              2; /* Reflect at (dim_size-1) */                                \
      }                                                                       \
                                                                              \
      /* Handle multiple reflections if needed */                             \
      while (pos < 0 || pos >= static_cast<int>(dim_size)) {                  \
        if (pos < 0) {                                                        \
          pos = -pos; /* Reflect at 0 */                                      \
        } else if (pos >= static_cast<int>(dim_size)) {                       \
          pos = 2 * static_cast<int>(dim_size) - pos -                        \
                2; /* Reflect at (dim_size-1) */                              \
        }                                                                     \
      }                                                                       \
                                                                              \
      in_coords[d] = static_cast<size_t>(pos);                                \
    }                                                                         \
                                                                              \
    /* Calculate index for input with strides */                              \
    size_t in_idx;                                                            \
    if (is_input_contiguous) {                                                \
      /* Calculate linear index for contiguous input */                       \
      in_idx = 0;                                                             \
      size_t stride = 1;                                                      \
      for (int d = num_dims - 1; d >= 0; --d) {                               \
        in_idx += in_coords[d] * stride;                                      \
        stride *= input_dims[d];                                              \
      }                                                                       \
    } else {                                                                  \
      /* Calculate strided index */                                           \
      in_idx = 0;                                                             \
      for (size_t d = 0; d < num_dims; d++) {                                 \
        in_idx += in_coords[d] * input_strides[d];                            \
      }                                                                       \
    }                                                                         \
                                                                              \
    /* Accumulate gradient */                                                 \
    if (in_idx < num_els_in) {                                                \
      (void)ATOMIC_ADD_FUNC(&grad_in[in_idx], grad_out[id]);                  \
    }                                                                         \
  }

#define PAD_WITH_REPLICATION_BACKWARD_OP(TYPENAME, FN_NAME, ATOMIC_ADD_FUNC)  \
  kernel void metal_##FN_NAME##_kernel(                                       \
      device TYPENAME *grad_in [[buffer(0)]],                                 \
      const device TYPENAME *grad_out [[buffer(1)]],                          \
      constant size_t& num_els_in [[buffer(2)]],                              \
      constant size_t& num_els_out [[buffer(3)]],                             \
      constant size_t& num_dims [[buffer(4)]],                                \
      constant size_t *metadata [[buffer(5)]],                                \
      uint id [[thread_position_in_grid]]) {                                  \
    if (id >= num_els_out) return;                                            \
                                                                              \
    const constant size_t *input_dims = metadata;                             \
    const constant size_t *input_strides = metadata + num_dims;               \
    /* Removed unused input_offset variable */                                \
    const constant size_t *output_dims = metadata + 2 * num_dims + 1;         \
    const constant size_t *paddings = metadata + 3 * num_dims + 1;            \
                                                                              \
    if (num_dims > MAX_DIMS)                                                  \
      return;                                                                 \
                                                                              \
    bool is_input_contiguous =                                                \
        is_contiguous(num_dims, input_dims, input_strides);                   \
                                                                              \
    /* Calculate coordinates in output tensor */                              \
    size_t out_coords[MAX_DIMS];                                              \
    size_t tmp_i = id;                                                        \
    for (int d = num_dims - 1; d >= 0; --d) {                                 \
      out_coords[d] = tmp_i % output_dims[d];                                 \
      tmp_i /= output_dims[d];                                                \
    }                                                                         \
                                                                              \
    /* Calculate corresponding coordinates in input tensor with replication */ \
    size_t in_coords[MAX_DIMS];                                               \
                                                                              \
    for (size_t d = 0; d < num_dims; d++) {                                   \
      size_t pad_before = paddings[d * 2];                                    \
      long pos = static_cast<long>(out_coords[d]) -                           \
                 static_cast<long>(pad_before);                               \
                                                                              \
      /* Apply replication (clamp to valid range) */                          \
      pos = max(0L, min(pos, static_cast<long>(input_dims[d] - 1)));          \
                                                                              \
      in_coords[d] = static_cast<size_t>(pos);                                \
    }                                                                         \
                                                                              \
    /* Calculate index for input with strides */                              \
    size_t in_idx;                                                            \
    if (is_input_contiguous) {                                                \
      /* Calculate linear index for contiguous input */                       \
      in_idx = 0;                                                             \
      size_t stride = 1;                                                      \
      for (int d = num_dims - 1; d >= 0; --d) {                               \
        in_idx += in_coords[d] * stride;                                      \
        stride *= input_dims[d];                                              \
      }                                                                       \
    } else {                                                                  \
      /* Calculate strided index */                                           \
      in_idx = 0;                                                             \
      for (size_t d = 0; d < num_dims; d++) {                                 \
        in_idx += in_coords[d] * input_strides[d];                            \
      }                                                                       \
    }                                                                         \
                                                                              \
    /* Accumulate gradient */                                                 \
    if (in_idx < num_els_in) {                                                \
      (void)ATOMIC_ADD_FUNC(&grad_in[in_idx], grad_out[id]);                  \
    }                                                                         \
  }

// Float 32
PAD_WITH_CONSTANT_OP(float, pad_with_constant_f32)
PAD_WITH_REFLECTION_OP(float, pad_with_reflection_f32)
PAD_WITH_REPLICATION_OP(float, pad_with_replication_f32)
PAD_WITH_CONSTANT_BACKWARD_OP(float, pad_with_constant_backward_f32, atomic_add_float)
PAD_WITH_REFLECTION_BACKWARD_OP(float, pad_with_reflection_backward_f32, atomic_add_float)
PAD_WITH_REPLICATION_BACKWARD_OP(float, pad_with_replication_backward_f32, atomic_add_float)

// Half-precision operations
PAD_WITH_CONSTANT_OP(half, pad_with_constant_f16)
PAD_WITH_REFLECTION_OP(half, pad_with_reflection_f16)
PAD_WITH_REPLICATION_OP(half, pad_with_replication_f16)
PAD_WITH_CONSTANT_BACKWARD_OP(half, pad_with_constant_backward_f16, atomic_add_half)
PAD_WITH_REFLECTION_BACKWARD_OP(half, pad_with_reflection_backward_f16, atomic_add_half)
PAD_WITH_REPLICATION_BACKWARD_OP(half, pad_with_replication_backward_f16, atomic_add_half)

// BFloat16 operations
PAD_WITH_CONSTANT_OP(bfloat, pad_with_constant_bf16)
PAD_WITH_REFLECTION_OP(bfloat, pad_with_reflection_bf16)
PAD_WITH_REPLICATION_OP(bfloat, pad_with_replication_bf16)
PAD_WITH_CONSTANT_BACKWARD_OP(bfloat, pad_with_constant_backward_bf16, atomic_add_bfloat)
PAD_WITH_REFLECTION_BACKWARD_OP(bfloat, pad_with_reflection_backward_bf16, atomic_add_bfloat)
PAD_WITH_REPLICATION_BACKWARD_OP(bfloat, pad_with_replication_backward_bf16, atomic_add_bfloat)

// uint8_t operations
PAD_WITH_CONSTANT_OP(uint8_t, pad_with_constant_u8)
PAD_WITH_REFLECTION_OP(uint8_t, pad_with_reflection_u8)
PAD_WITH_REPLICATION_OP(uint8_t, pad_with_replication_u8)
PAD_WITH_CONSTANT_BACKWARD_OP(uint8_t, pad_with_constant_backward_u8, atomic_add_uint8)
PAD_WITH_REFLECTION_BACKWARD_OP(uint8_t, pad_with_reflection_backward_u8, atomic_add_uint8)
PAD_WITH_REPLICATION_BACKWARD_OP(uint8_t, pad_with_replication_backward_u8, atomic_add_uint8)

// uint16_t operations
PAD_WITH_CONSTANT_OP(uint16_t, pad_with_constant_u16)
PAD_WITH_REFLECTION_OP(uint16_t, pad_with_reflection_u16)
PAD_WITH_REPLICATION_OP(uint16_t, pad_with_replication_u16)
PAD_WITH_CONSTANT_BACKWARD_OP(uint16_t, pad_with_constant_backward_u16, atomic_add_uint16)
PAD_WITH_REFLECTION_BACKWARD_OP(uint16_t, pad_with_reflection_backward_u16, atomic_add_uint16)
PAD_WITH_REPLICATION_BACKWARD_OP(uint16_t, pad_with_replication_backward_u16, atomic_add_uint16)

// uint32_t operations
PAD_WITH_CONSTANT_OP(uint32_t, pad_with_constant_u32)
PAD_WITH_REFLECTION_OP(uint32_t, pad_with_reflection_u32)
PAD_WITH_REPLICATION_OP(uint32_t, pad_with_replication_u32)
PAD_WITH_CONSTANT_BACKWARD_OP(uint32_t, pad_with_constant_backward_u32, atomic_add_uint32)
PAD_WITH_REFLECTION_BACKWARD_OP(uint32_t, pad_with_reflection_backward_u32, atomic_add_uint32)
PAD_WITH_REPLICATION_BACKWARD_OP(uint32_t, pad_with_replication_backward_u32, atomic_add_uint32)

// int8_t operations
PAD_WITH_CONSTANT_OP(int8_t, pad_with_constant_i8)
PAD_WITH_REFLECTION_OP(int8_t, pad_with_reflection_i8)
PAD_WITH_REPLICATION_OP(int8_t, pad_with_replication_i8)
PAD_WITH_CONSTANT_BACKWARD_OP(int8_t, pad_with_constant_backward_i8, atomic_add_int8)
PAD_WITH_REFLECTION_BACKWARD_OP(int8_t, pad_with_reflection_backward_i8, atomic_add_int8)
PAD_WITH_REPLICATION_BACKWARD_OP(int8_t, pad_with_replication_backward_i8, atomic_add_int8)

// int16_t operations
PAD_WITH_CONSTANT_OP(int16_t, pad_with_constant_i16)
PAD_WITH_REFLECTION_OP(int16_t, pad_with_reflection_i16)
PAD_WITH_REPLICATION_OP(int16_t, pad_with_replication_i16)
PAD_WITH_CONSTANT_BACKWARD_OP(int16_t, pad_with_constant_backward_i16, atomic_add_int16)
PAD_WITH_REFLECTION_BACKWARD_OP(int16_t, pad_with_reflection_backward_i16, atomic_add_int16)
PAD_WITH_REPLICATION_BACKWARD_OP(int16_t, pad_with_replication_backward_i16, atomic_add_int16)

// int32_t operations
PAD_WITH_CONSTANT_OP(int32_t, pad_with_constant_i32)
PAD_WITH_REFLECTION_OP(int32_t, pad_with_reflection_i32)
PAD_WITH_REPLICATION_OP(int32_t, pad_with_replication_i32)
PAD_WITH_CONSTANT_BACKWARD_OP(int32_t, pad_with_constant_backward_i32, atomic_add_int32)
PAD_WITH_REFLECTION_BACKWARD_OP(int32_t, pad_with_reflection_backward_i32, atomic_add_int32)
PAD_WITH_REPLICATION_BACKWARD_OP(int32_t, pad_with_replication_backward_i32, atomic_add_int32)
