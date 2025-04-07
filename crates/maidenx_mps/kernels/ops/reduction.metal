#include "../metal_utils.metal"
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

#define MAX_DIMS 16

// =========================================================
// Utility helpers
// =========================================================

template <typename T>
inline T zero_val() { return (T)0; }

// Decode a linear index into multidimensional coordinates (constant dims)
inline void linear_to_coords(size_t lin_idx,
                             const constant size_t *dims,
                             size_t ndim,
                             thread size_t *coords /* out */) {
  for (int d = (int)ndim - 1; d >= 0; --d) {
    coords[d] = lin_idx % dims[d];
    lin_idx  /= dims[d];
  }
}

// Overload for dims in thread address space
inline void linear_to_coords(size_t lin_idx,
                             thread const size_t *dims,
                             size_t ndim,
                             thread size_t *coords /* out */) {
  for (int d = (int)ndim - 1; d >= 0; --d) {
    coords[d] = lin_idx % dims[d];
    lin_idx  /= dims[d];
  }
}

// =========================================================
// 1‑axis reductions (sum / mean / max / min)
// =========================================================

#define REDUCTION_SUM_KERNEL(IN_TYPENAME, FN_SUFFIX)                                                         \
kernel void metal_sum_##FN_SUFFIX##_kernel(                                                                 \
    device IN_TYPENAME *output               [[buffer(0)]],                                                  \
    const device IN_TYPENAME *input          [[buffer(1)]],                                                  \
    constant size_t           &num_els       [[buffer(2)]],                                                  \
    constant size_t           &num_dims      [[buffer(3)]],                                                  \
    constant size_t           &num_red_dims  [[buffer(4)]],                                                  \
    constant size_t           *metadata      [[buffer(5)]],                                                  \
    uint                       gid           [[thread_position_in_grid]],                                   \
    uint                       threads_total [[threads_per_grid]])                                           \
{                                                                                                            \
    if (num_red_dims != 1) return;                                                                           \
                                                                                                             \
    const constant size_t *dims    = metadata;                                                               \
    const constant size_t *strides = metadata + num_dims;                                                    \
    size_t red_size   = *(metadata + 2 * num_dims);                                                          \
    size_t red_stride = *(metadata + 2 * num_dims + 1);                                                      \
    size_t offset     = *(metadata + 2 * num_dims + 2);                                                      \
                                                                                                             \
    size_t out_size = num_els / red_size;                                                                    \
                                                                                                             \
    for (size_t out_idx = gid; out_idx < out_size; out_idx += threads_total) {                               \
        size_t tmp      = out_idx;                                                                           \
        size_t inp_base = offset;                                                                            \
        /* reconstruct coordinates for non‑reduction dims */                                                 \
        for (int d = (int)num_dims - 1; d >= 0; --d) {                                                       \
            if (strides[d] == red_stride) continue; /* reduction dim */                                      \
            size_t coord = tmp % dims[d];                                                                    \
            tmp         /= dims[d];                                                                          \
            inp_base    += coord * strides[d];                                                               \
        }                                                                                                    \
        IN_TYPENAME acc = zero_val<IN_TYPENAME>();                                                           \
        for (size_t k = 0; k < red_size; ++k) {                                                              \
            acc += input[inp_base + k * red_stride];                                                         \
        }                                                                                                    \
        output[out_idx] = acc;                                                                               \
    }                                                                                                        \
}

#define REDUCTION_MEAN_KERNEL(IN_TYPENAME, FN_SUFFIX)                                                        \
kernel void metal_mean_##FN_SUFFIX##_kernel(                                                                \
    device IN_TYPENAME *output               [[buffer(0)]],                                                  \
    const device IN_TYPENAME *input          [[buffer(1)]],                                                  \
    constant size_t           &num_els       [[buffer(2)]],                                                  \
    constant size_t           &num_dims      [[buffer(3)]],                                                  \
    constant size_t           &num_red_dims  [[buffer(4)]],                                                  \
    constant size_t           *metadata      [[buffer(5)]],                                                  \
    uint                       gid           [[thread_position_in_grid]],                                   \
    uint                       threads_total [[threads_per_grid]])                                           \
{                                                                                                            \
    if (num_red_dims != 1) return;                                                                           \
    const constant size_t *dims    = metadata;                                                               \
    const constant size_t *strides = metadata + num_dims;                                                    \
    size_t red_size   = *(metadata + 2 * num_dims);                                                          \
    size_t red_stride = *(metadata + 2 * num_dims + 1);                                                      \
    size_t offset     = *(metadata + 2 * num_dims + 2);                                                      \
    size_t out_size = num_els / red_size;                                                                    \
                                                                                                             \
    for (size_t out_idx = gid; out_idx < out_size; out_idx += threads_total) {                               \
        size_t tmp      = out_idx;                                                                           \
        size_t inp_base = offset;                                                                            \
        for (int d = (int)num_dims - 1; d >= 0; --d) {                                                       \
            if (strides[d] == red_stride) continue;                                                          \
            size_t coord = tmp % dims[d];                                                                    \
            tmp         /= dims[d];                                                                          \
            inp_base    += coord * strides[d];                                                               \
        }                                                                                                    \
        IN_TYPENAME acc = zero_val<IN_TYPENAME>();                                                           \
        for (size_t k = 0; k < red_size; ++k) {                                                              \
            acc += input[inp_base + k * red_stride];                                                         \
        }                                                                                                    \
        output[out_idx] = acc / (IN_TYPENAME)red_size;                                                       \
    }                                                                                                        \
}

#define REDUCTION_MAX_KERNEL(IN_TYPENAME, FN_SUFFIX)                                                         \
kernel void metal_max_##FN_SUFFIX##_kernel(                                                                 \
    device IN_TYPENAME *output               [[buffer(0)]],                                                  \
    const device IN_TYPENAME *input          [[buffer(1)]],                                                  \
    constant size_t           &num_els       [[buffer(2)]],                                                  \
    constant size_t           &num_dims      [[buffer(3)]],                                                  \
    constant size_t           &num_red_dims  [[buffer(4)]],                                                  \
    constant size_t           *metadata      [[buffer(5)]],                                                  \
    uint                       gid           [[thread_position_in_grid]],                                   \
    uint                       threads_total [[threads_per_grid]])                                           \
{                                                                                                            \
    if (num_red_dims != 1) return;                                                                           \
    const constant size_t *dims    = metadata;                                                               \
    const constant size_t *strides = metadata + num_dims;                                                    \
    size_t red_size   = *(metadata + 2 * num_dims);                                                          \
    size_t red_stride = *(metadata + 2 * num_dims + 1);                                                      \
    size_t offset     = *(metadata + 2 * num_dims + 2);                                                      \
    size_t out_size   = num_els / red_size;                                                                  \
                                                                                                             \
    for (size_t out_idx = gid; out_idx < out_size; out_idx += threads_total) {                               \
        size_t tmp      = out_idx;                                                                           \
        size_t inp_base = offset;                                                                            \
        for (int d = (int)num_dims - 1; d >= 0; --d) {                                                       \
            if (strides[d] == red_stride) continue;                                                          \
            size_t coord = tmp % dims[d];                                                                    \
            tmp         /= dims[d];                                                                          \
            inp_base    += coord * strides[d];                                                               \
        }                                                                                                    \
        IN_TYPENAME acc = input[inp_base];                                                                   \
        for (size_t k = 1; k < red_size; ++k) {                                                              \
            IN_TYPENAME v = input[inp_base + k * red_stride];                                                \
            if (v > acc) acc = v;                                                                            \
        }                                                                                                    \
        output[out_idx] = acc;                                                                               \
    }                                                                                                        \
}

#define REDUCTION_MIN_KERNEL(IN_TYPENAME, FN_SUFFIX)                                                         \
kernel void metal_min_##FN_SUFFIX##_kernel(                                                                 \
    device IN_TYPENAME *output               [[buffer(0)]],                                                  \
    const device IN_TYPENAME *input          [[buffer(1)]],                                                  \
    constant size_t           &num_els       [[buffer(2)]],                                                  \
    constant size_t           &num_dims      [[buffer(3)]],                                                  \
    constant size_t           &num_red_dims  [[buffer(4)]],                                                  \
    constant size_t           *metadata      [[buffer(5)]],                                                  \
    uint                       gid           [[thread_position_in_grid]],                                   \
    uint                       threads_total [[threads_per_grid]])                                           \
{                                                                                                            \
    if (num_red_dims != 1) return;                                                                           \
    const constant size_t *dims    = metadata;                                                               \
    const constant size_t *strides = metadata + num_dims;                                                    \
    size_t red_size   = *(metadata + 2 * num_dims);                                                          \
    size_t red_stride = *(metadata + 2 * num_dims + 1);                                                      \
    size_t offset     = *(metadata + 2 * num_dims + 2);                                                      \
    size_t out_size   = num_els / red_size;                                                                  \
                                                                                                             \
    for (size_t out_idx = gid; out_idx < out_size; out_idx += threads_total) {                               \
        size_t tmp      = out_idx;                                                                           \
        size_t inp_base = offset;                                                                            \
        for (int d = (int)num_dims - 1; d >= 0; --d) {                                                       \
            if (strides[d] == red_stride) continue;                                                          \
            size_t coord = tmp % dims[d];                                                                    \
            tmp         /= dims[d];                                                                          \
            inp_base    += coord * strides[d];                                                               \
        }                                                                                                    \
        IN_TYPENAME acc = input[inp_base];                                                                   \
        for (size_t k = 1; k < red_size; ++k) {                                                              \
            IN_TYPENAME v = input[inp_base + k * red_stride];                                                \
            if (v < acc) acc = v;                                                                            \
        }                                                                                                    \
        output[out_idx] = acc;                                                                               \
    }                                                                                                        \
}

// =========================================================
// sum_to_shape (broadcast‑aware reduction)
// =========================================================

#define SHAPE_SUM_TO_SHAPE_KERNEL(IN_TYPENAME, FN_SUFFIX)                                                    \
kernel void metal_sum_to_shape_##FN_SUFFIX##_kernel(                                                        \
    device IN_TYPENAME *output               [[buffer(0)]],                                                  \
    const device IN_TYPENAME *input          [[buffer(1)]],                                                  \
    constant size_t           &num_els       [[buffer(2)]],                                                  \
    constant size_t           &num_dims      [[buffer(3)]],                                                  \
    constant size_t           *metadata      [[buffer(4)]],                                                  \
    uint                       gid           [[thread_position_in_grid]],                                   \
    uint                       threads_total [[threads_per_grid]])                                           \
{                                                                                                            \
    const constant size_t *in_dims    = metadata;                                                            \
    const constant size_t *in_strides = metadata + num_dims;                                                 \
    const constant size_t *tgt_shape  = metadata + 2 * num_dims;                                             \
    size_t offset = *(metadata + 3 * num_dims);                                                              \
                                                                                                             \
    size_t out_size = 1;                                                                                     \
    size_t factors[MAX_DIMS];                                                                                \
    for (size_t d = 0; d < num_dims; ++d) {                                                                  \
        out_size     *= tgt_shape[d];                                                                        \
        factors[d]    = in_dims[d] / tgt_shape[d];                                                           \
    }                                                                                                        \
                                                                                                             \
    thread size_t coords[MAX_DIMS];                                                                          \
    for (size_t out_idx = gid; out_idx < out_size; out_idx += threads_total) {                               \
        linear_to_coords(out_idx, tgt_shape, num_dims, coords);                                              \
                                                                                                             \
        size_t base = offset;                                                                                \
        for (size_t d = 0; d < num_dims; ++d) {                                                              \
            base += coords[d] * factors[d] * in_strides[d];                                                  \
        }                                                                                                    \
                                                                                                             \
        size_t red_total = 1;                                                                                \
        for (size_t d = 0; d < num_dims; ++d) red_total *= factors[d];                                       \
                                                                                                             \
        IN_TYPENAME acc = zero_val<IN_TYPENAME>();                                                           \
        for (size_t r = 0; r < red_total; ++r) {                                                             \
            size_t tmp_r = r;                                                                                \
            size_t idx   = base;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; --d) {                                                   \
                size_t rc = tmp_r % factors[d];                                                              \
                tmp_r    /= factors[d];                                                                      \
                idx      += rc * in_strides[d];                                                              \
            }                                                                                                \
            acc += input[idx];                                                                               \
        }                                                                                                    \
        output[out_idx] = acc;                                                                               \
    }                                                                                                        \
}

// =========================================================
// fold – overlap‑add of sliding windows
// =========================================================

#define SHAPE_FOLD_KERNEL(IN_TYPENAME, FN_SUFFIX)                                                            \
kernel void metal_fold_##FN_SUFFIX##_kernel(                                                                \
    device IN_TYPENAME *output               [[buffer(0)]],                                                  \
    const device IN_TYPENAME *input          [[buffer(1)]],                                                  \
    constant size_t           &num_els_out   [[buffer(2)]],                                                  \
    constant size_t           &num_dims      [[buffer(3)]],                                                  \
    constant size_t           *metadata      [[buffer(4)]],                                                  \
    uint                       gid           [[thread_position_in_grid]],                                   \
    uint                       threads_total [[threads_per_grid]])                                           \
{                                                                                                            \
    const constant size_t *in_dims    = metadata;                                                            \
    const constant size_t *in_strides = metadata + num_dims;                                                 \
                                                                                                             \
    size_t fold_dim     = *(metadata + 2 * num_dims);                                                        \
    size_t window_dim   = *(metadata + 2 * num_dims + 1);                                                    \
    size_t fold_size    = *(metadata + 2 * num_dims + 2);                                                    \
    size_t step         = *(metadata + 2 * num_dims + 3);                                                    \
    size_t window_size  = *(metadata + 2 * num_dims + 4);                                                    \
    size_t offset       = *(metadata + 2 * num_dims + 5);                                                    \
                                                                                                             \
    /* Build output dims (num_dims - 1) */                                                                   \
    size_t out_dims[MAX_DIMS];                                                                               \
    size_t out_num_dims = 0;                                                                                 \
    for (size_t d = 0; d < num_dims; ++d) {                                                                  \
        if (d == window_dim) continue;                                                                       \
        if (d == fold_dim) {                                                                                 \
            out_dims[out_num_dims++] = fold_size;                                                            \
        } else {                                                                                             \
            out_dims[out_num_dims++] = in_dims[d];                                                           \
        }                                                                                                    \
    }                                                                                                        \
                                                                                                             \
    thread size_t coords[MAX_DIMS];                                                                          \
    for (size_t out_idx = gid; out_idx < num_els_out; out_idx += threads_total) {                            \
        linear_to_coords(out_idx, out_dims, out_num_dims, coords);                                           \
                                                                                                             \
        size_t base = offset;                                                                                \
        size_t coord_out_idx = 0;                                                                            \
        size_t x_coord = 0;                                                                                  \
        for (size_t d = 0; d < num_dims; ++d) {                                                              \
            if (d == window_dim) continue;                                                                   \
            size_t coord = coords[coord_out_idx];                                                            \
            if (d == fold_dim) {                                                                             \
                x_coord = coord;                                                                             \
            } else {                                                                                         \
                base += coord * in_strides[d];                                                               \
            }                                                                                                \
            coord_out_idx++;                                                                                 \
        }                                                                                                    \
                                                                                                             \
        size_t n_windows = in_dims[fold_dim];                                                                \
        IN_TYPENAME acc = zero_val<IN_TYPENAME>();                                                           \
        for (size_t w = 0; w < n_windows; ++w) {                                                             \
            size_t start = w * step;                                                                         \
            if (x_coord < start || x_coord >= start + window_size) continue;                                 \
            size_t k = x_coord - start;                                                                      \
            size_t idx = base + w * in_strides[fold_dim] + k * in_strides[window_dim];                       \
            acc += input[idx];                                                                               \
        }                                                                                                    \
        output[out_idx] = acc;                                                                               \
    }                                                                                                        \
}

// =========================================================
// Instantiation for supported dtypes
// =========================================================

// ----- float32 / float16 / bfloat16 -----
REDUCTION_SUM_KERNEL(float,   f32)
REDUCTION_MEAN_KERNEL(float,  f32)
REDUCTION_MAX_KERNEL(float,   f32)
REDUCTION_MIN_KERNEL(float,   f32)
SHAPE_SUM_TO_SHAPE_KERNEL(float, f32)
SHAPE_FOLD_KERNEL(float, f32)

REDUCTION_SUM_KERNEL(half,    f16)
REDUCTION_MEAN_KERNEL(half,   f16)
REDUCTION_MAX_KERNEL(half,    f16)
REDUCTION_MIN_KERNEL(half,    f16)
SHAPE_SUM_TO_SHAPE_KERNEL(half, f16)
SHAPE_FOLD_KERNEL(half, f16)

REDUCTION_SUM_KERNEL(bfloat,  bf16)
REDUCTION_MEAN_KERNEL(bfloat, bf16)
REDUCTION_MAX_KERNEL(bfloat,  bf16)
REDUCTION_MIN_KERNEL(bfloat,  bf16)
SHAPE_SUM_TO_SHAPE_KERNEL(bfloat, bf16)
SHAPE_FOLD_KERNEL(bfloat, bf16)

// ----- unsigned integers -----
REDUCTION_SUM_KERNEL(uint8_t,  u8)
REDUCTION_MEAN_KERNEL(uint8_t, u8)
REDUCTION_MAX_KERNEL(uint8_t,  u8)
REDUCTION_MIN_KERNEL(uint8_t,  u8)
SHAPE_SUM_TO_SHAPE_KERNEL(uint8_t, u8)
SHAPE_FOLD_KERNEL(uint8_t, u8)

REDUCTION_SUM_KERNEL(uint16_t,  u16)
REDUCTION_MEAN_KERNEL(uint16_t, u16)
REDUCTION_MAX_KERNEL(uint16_t,  u16)
REDUCTION_MIN_KERNEL(uint16_t,  u16)
SHAPE_SUM_TO_SHAPE_KERNEL(uint16_t, u16)
SHAPE_FOLD_KERNEL(uint16_t, u16)

REDUCTION_SUM_KERNEL(uint32_t,  u32)
REDUCTION_MEAN_KERNEL(uint32_t, u32)
REDUCTION_MAX_KERNEL(uint32_t,  u32)
REDUCTION_MIN_KERNEL(uint32_t,  u32)
SHAPE_SUM_TO_SHAPE_KERNEL(uint32_t, u32)
SHAPE_FOLD_KERNEL(uint32_t, u32)

// ----- signed integers -----
REDUCTION_SUM_KERNEL(int8_t,  i8)
REDUCTION_MEAN_KERNEL(int8_t, i8)
REDUCTION_MAX_KERNEL(int8_t,  i8)
REDUCTION_MIN_KERNEL(int8_t,  i8)
SHAPE_SUM_TO_SHAPE_KERNEL(int8_t, i8)
SHAPE_FOLD_KERNEL(int8_t, i8)

REDUCTION_SUM_KERNEL(int16_t,  i16)
REDUCTION_MEAN_KERNEL(int16_t, i16)
REDUCTION_MAX_KERNEL(int16_t,  i16)
REDUCTION_MIN_KERNEL(int16_t,  i16)
SHAPE_SUM_TO_SHAPE_KERNEL(int16_t, i16)
SHAPE_FOLD_KERNEL(int16_t, i16)

REDUCTION_SUM_KERNEL(int32_t,  i32)
REDUCTION_MEAN_KERNEL(int32_t, i32)
REDUCTION_MAX_KERNEL(int32_t,  i32)
REDUCTION_MIN_KERNEL(int32_t,  i32)
SHAPE_SUM_TO_SHAPE_KERNEL(int32_t, i32)
SHAPE_FOLD_KERNEL(int32_t, i32)

// Cleanup macro namespace
#undef MAX_DIMS
#undef zero_val
#undef linear_to_coords
#undef REDUCTION_SUM_KERNEL
#undef REDUCTION_MEAN_KERNEL
#undef REDUCTION_MAX_KERNEL
#undef REDUCTION_MIN_KERNEL
#undef SHAPE_SUM_TO_SHAPE_KERNEL
#undef SHAPE_FOLD_KERNEL

