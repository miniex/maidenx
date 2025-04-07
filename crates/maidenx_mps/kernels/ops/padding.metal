#include "../metal_utils.metal"
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

#define MAX_DIMS 16

// ============================= Helpers =============================

template <typename T> inline T zero_val() { return (T)0; }

// For atomic add, but if there's no real atomic ops needed, this fallback suffices:
template <typename T> inline void atomic_add(device T *ptr, T v) { *ptr += v; }

inline void lin2coords(size_t lin, const constant size_t *dims, size_t ndim, thread size_t *coords) {
    for (int d = (int)ndim - 1; d >= 0; --d) {
        coords[d] = lin % dims[d];
        lin /= dims[d];
    }
}

inline size_t coords2offset(const thread size_t *coords, const constant size_t *strides, size_t ndim) {
    size_t off = 0;
    for (size_t d = 0; d < ndim; ++d) {
        off += coords[d] * strides[d];
    }
    return off;
}

// metadata layout helpers
// [0..num_dims-1]: in_dims
// [num_dims..2*num_dims-1]: in_strides
// [2*num_dims]: in_offset
// [2*num_dims+1..3*num_dims]: out_dims
// [3*num_dims+1..3*num_dims+2*num_dims]: pads (pairs)
#define META_PTRS                                                                        \
    const constant size_t *in_dims    = metadata;                                        \
    const constant size_t *in_strides = metadata + num_dims;                             \
    size_t in_offset                  = *(metadata + 2 * num_dims);                      \
    const constant size_t *out_dims   = metadata + (2 * num_dims + 1);                   \
    const constant size_t *pads       = metadata + (3 * num_dims + 1);

#define PAD_BEFORE(d) pads[2*(d)]
#define PAD_AFTER(d)  pads[2*(d) + 1]

// ============================= Forward kernels =============================

#define PAD_CONSTANT_KERNEL(T, SUF)                                                       \
kernel void metal_pad_with_constant_##SUF##_kernel(                                      \
    device T *out [[buffer(0)]], const device T *inp [[buffer(1)]],                      \
    constant size_t &n_in [[buffer(2)]], constant size_t &n_out [[buffer(3)]],           \
    constant size_t &num_dims [[buffer(4)]], constant size_t *metadata [[buffer(5)]],    \
    constant T &pad_val [[buffer(6)]],                                                   \
    uint gid [[thread_position_in_grid]], uint nthreads [[threads_per_grid]])            \
{                                                                                         \
    META_PTRS;                                                                            \
    thread size_t c[MAX_DIMS];                                                            \
    for (size_t o = gid; o < n_out; o += nthreads) {                                      \
        lin2coords(o, out_dims, num_dims, c);                                            \
        /* Check if output index is inside the original input area */                    \
        bool inside = true;                                                              \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            size_t co = c[d];                                                            \
            if (co < PAD_BEFORE(d) || co >= PAD_BEFORE(d) + in_dims[d]) {                \
                inside = false;                                                          \
                break;                                                                   \
            }                                                                            \
        }                                                                                \
        if (!inside) {                                                                   \
            out[o] = pad_val;                                                            \
            continue;                                                                    \
        }                                                                                \
        /* Subtract the pad_before offsets to map back into input space */               \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            c[d] -= PAD_BEFORE(d);                                                       \
        }                                                                                \
        size_t in_off = in_offset + coords2offset(c, in_strides, num_dims);              \
        out[o] = inp[in_off];                                                            \
    }                                                                                    \
}

#define PAD_REFLECT_KERNEL(T, SUF)                                                       \
kernel void metal_pad_with_reflection_##SUF##_kernel(                                    \
    device T *out [[buffer(0)]], const device T *inp [[buffer(1)]],                      \
    constant size_t &n_in [[buffer(2)]], constant size_t &n_out [[buffer(3)]],           \
    constant size_t &num_dims [[buffer(4)]], constant size_t *metadata [[buffer(5)]],    \
    uint gid [[thread_position_in_grid]], uint nthreads [[threads_per_grid]])            \
{                                                                                         \
    META_PTRS;                                                                            \
    thread size_t c[MAX_DIMS];                                                            \
    for (size_t o = gid; o < n_out; o += nthreads) {                                      \
        lin2coords(o, out_dims, num_dims, c);                                            \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            size_t co = c[d];                                                            \
            size_t pb = PAD_BEFORE(d);                                                   \
            size_t dim = in_dims[d];                                                     \
            /* Reflection logic: left region / right region */                           \
            if (co < pb) {                                                               \
                co = pb - co;                                                            \
            } else if (co >= pb + dim) {                                                 \
                co = 2 * dim - (co - pb) - 2;                                            \
            } else {                                                                     \
                co -= pb;                                                                \
            }                                                                            \
            c[d] = co;                                                                   \
        }                                                                                \
        size_t in_off = in_offset + coords2offset(c, in_strides, num_dims);              \
        out[o] = inp[in_off];                                                            \
    }                                                                                    \
}

#define PAD_REPL_KERNEL(T, SUF)                                                          \
kernel void metal_pad_with_replication_##SUF##_kernel(                                   \
    device T *out [[buffer(0)]], const device T *inp [[buffer(1)]],                      \
    constant size_t &n_in [[buffer(2)]], constant size_t &n_out [[buffer(3)]],           \
    constant size_t &num_dims [[buffer(4)]], constant size_t *metadata [[buffer(5)]],    \
    uint gid [[thread_position_in_grid]], uint nthreads [[threads_per_grid]])            \
{                                                                                         \
    META_PTRS;                                                                            \
    thread size_t c[MAX_DIMS];                                                            \
    for (size_t o = gid; o < n_out; o += nthreads) {                                      \
        lin2coords(o, out_dims, num_dims, c);                                            \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            size_t co = c[d], pb = PAD_BEFORE(d), dim = in_dims[d];                      \
            if (co < pb) {                                                               \
                co = 0;                                                                  \
            } else if (co >= pb + dim) {                                                 \
                co = dim - 1;                                                            \
            } else {                                                                     \
                co -= pb;                                                                \
            }                                                                            \
            c[d] = co;                                                                   \
        }                                                                                \
        size_t in_off = in_offset + coords2offset(c, in_strides, num_dims);              \
        out[o] = inp[in_off];                                                            \
    }                                                                                    \
}

// ============================= Backward kernels =============================

#define PAD_CONSTANT_BACK_KERNEL(T, SUF)                                                 \
kernel void metal_pad_with_constant_backward_##SUF##_kernel(                             \
    device T *gin [[buffer(0)]], const device T *gout [[buffer(1)]],                     \
    constant size_t &n_in [[buffer(2)]], constant size_t &n_out [[buffer(3)]],           \
    constant size_t &num_dims [[buffer(4)]], constant size_t *metadata [[buffer(5)]],    \
    uint gid [[thread_position_in_grid]], uint nthreads [[threads_per_grid]])            \
{                                                                                         \
    META_PTRS;                                                                            \
    /* zero out gin in parallel */                                                        \
    for (size_t i = gid; i < n_in; i += nthreads) {                                       \
        gin[i] = zero_val<T>();                                                           \
    }                                                                                    \
    threadgroup_barrier(mem_flags::mem_device);                                           \
                                                                                          \
    thread size_t c[MAX_DIMS];                                                            \
    for (size_t o = gid; o < n_out; o += nthreads) {                                      \
        lin2coords(o, out_dims, num_dims, c);                                            \
        /* Check if inside original input region */                                       \
        bool inside = true;                                                              \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            size_t co = c[d];                                                            \
            if (co < PAD_BEFORE(d) || co >= PAD_BEFORE(d) + in_dims[d]) {                \
                inside = false;                                                          \
                break;                                                                   \
            }                                                                            \
        }                                                                                \
        if (!inside) continue;                                                           \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            c[d] -= PAD_BEFORE(d);                                                       \
        }                                                                                \
        size_t in_off = in_offset + coords2offset(c, in_strides, num_dims);              \
        gin[in_off] += gout[o];                                                          \
    }                                                                                    \
}

#define REFLECT_MAP(d)                                                                   \
{                                                                                         \
    size_t pb  = PAD_BEFORE(d);                                                           \
    size_t dim = in_dims[d];                                                              \
    size_t co  = c[d];                                                                    \
    if (co < pb) {                                                                        \
        co = pb - co;                                                                     \
    } else if (co >= pb + dim) {                                                          \
        co = 2 * dim - (co - pb) - 2;                                                     \
    } else {                                                                              \
        co -= pb;                                                                         \
    }                                                                                     \
    c[d] = co;                                                                            \
}

#define REPL_MAP(d)                                                                      \
{                                                                                         \
    size_t pb  = PAD_BEFORE(d);                                                           \
    size_t dim = in_dims[d];                                                              \
    size_t co  = c[d];                                                                    \
    if (co < pb) {                                                                        \
        co = 0;                                                                           \
    } else if (co >= pb + dim) {                                                          \
        co = dim - 1;                                                                     \
    } else {                                                                              \
        co -= pb;                                                                         \
    }                                                                                     \
    c[d] = co;                                                                            \
}

#define PATTERN_BACK_KERNEL(T, PAT, SUF, MAP_MACRO)                                      \
kernel void metal_pad_with_##PAT##_backward_##SUF##_kernel(                              \
    device T *gin [[buffer(0)]], const device T *gout [[buffer(1)]],                     \
    constant size_t &n_in [[buffer(2)]], constant size_t &n_out [[buffer(3)]],           \
    constant size_t &num_dims [[buffer(4)]], constant size_t *metadata [[buffer(5)]],    \
    uint gid [[thread_position_in_grid]])                                                \
{                                                                                         \
    /* single-thread approach: only one thread does the accumulation */                  \
    if (gid != 0) return;                                                                 \
    META_PTRS;                                                                            \
    for (size_t i = 0; i < n_in; ++i) {                                                   \
        gin[i] = zero_val<T>();                                                           \
    }                                                                                    \
    thread size_t c[MAX_DIMS];                                                            \
    for (size_t o = 0; o < n_out; ++o) {                                                 \
        lin2coords(o, out_dims, num_dims, c);                                            \
        for (size_t d = 0; d < num_dims; ++d) {                                          \
            MAP_MACRO(d);                                                                \
        }                                                                                \
        size_t in_off = in_offset + coords2offset(c, in_strides, num_dims);              \
        gin[in_off] += gout[o];                                                          \
    }                                                                                    \
}

// ============================= Instantiation =============================

// forward kernels
PAD_CONSTANT_KERNEL(float,f32)
PAD_REFLECT_KERNEL(float,f32)
PAD_REPL_KERNEL(float,f32)

PAD_CONSTANT_KERNEL(half,f16)
PAD_REFLECT_KERNEL(half,f16)
PAD_REPL_KERNEL(half,f16)

PAD_CONSTANT_KERNEL(bfloat,bf16)
PAD_REFLECT_KERNEL(bfloat,bf16)
PAD_REPL_KERNEL(bfloat,bf16)

PAD_CONSTANT_KERNEL(uint8_t,u8)
PAD_REFLECT_KERNEL(uint8_t,u8)
PAD_REPL_KERNEL(uint8_t,u8)

PAD_CONSTANT_KERNEL(uint16_t,u16)
PAD_REFLECT_KERNEL(uint16_t,u16)
PAD_REPL_KERNEL(uint16_t,u16)

PAD_CONSTANT_KERNEL(uint32_t,u32)
PAD_REFLECT_KERNEL(uint32_t,u32)
PAD_REPL_KERNEL(uint32_t,u32)

PAD_CONSTANT_KERNEL(int8_t,i8)
PAD_REFLECT_KERNEL(int8_t,i8)
PAD_REPL_KERNEL(int8_t,i8)

PAD_CONSTANT_KERNEL(int16_t,i16)
PAD_REFLECT_KERNEL(int16_t,i16)
PAD_REPL_KERNEL(int16_t,i16)

PAD_CONSTANT_KERNEL(int32_t,i32)
PAD_REFLECT_KERNEL(int32_t,i32)
PAD_REPL_KERNEL(int32_t,i32)

// constant backward
PAD_CONSTANT_BACK_KERNEL(float,f32)
PAD_CONSTANT_BACK_KERNEL(half,f16)
PAD_CONSTANT_BACK_KERNEL(bfloat,bf16)

// reflection / replication backward for float/half/bfloat
PATTERN_BACK_KERNEL(float,reflection,f32,REFLECT_MAP)
PATTERN_BACK_KERNEL(half,reflection,f16,REFLECT_MAP)
PATTERN_BACK_KERNEL(bfloat,reflection,bf16,REFLECT_MAP)
PATTERN_BACK_KERNEL(float,replication,f32,REPL_MAP)
PATTERN_BACK_KERNEL(half,replication,f16,REPL_MAP)
PATTERN_BACK_KERNEL(bfloat,replication,bf16,REPL_MAP)

// cleanup
#undef PAD_BEFORE
#undef PAD_AFTER
#undef META_PTRS
#undef PAD_CONSTANT_KERNEL
#undef PAD_REFLECT_KERNEL
#undef PAD_REPL_KERNEL
#undef PAD_CONSTANT_BACK_KERNEL
#undef PATTERN_BACK_KERNEL
#undef REFLECT_MAP
#undef REPL_MAP
#undef MAX_DIMS

