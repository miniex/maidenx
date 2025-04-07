// conv2d_im2col_col2im.metal – no‑atomic version
#include "../metal_utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T> T get_zero() { return T(0); }

/* ─────────────── im2col ─────────────── */
#define IM2COL_KERNEL(T, SUF)                                                  \
kernel void metal_conv2d_im2col_##SUF##_kernel(                                \
    const device T *input [[buffer(0)]],                                       \
    device T *col         [[buffer(1)]],                                       \
    constant size_t &num_col_el [[buffer(2)]],                                 \
    constant size_t *meta [[buffer(3)]],                                       \
    uint gid [[thread_position_in_grid]], uint nthreads [[threads_per_grid]])  \
{                                                                              \
    const constant size_t *d = meta;             /* dims 0..7  */              \
    const constant size_t *s = meta + 8;          /* strides 0..3 */           \
                                                                               \
    size_t C=d[1], H=d[2], W=d[3];                                              \
    size_t kh_max=d[4], kw_max=d[5], oh_max=d[6], ow_max=d[7];                  \
    size_t pad_h=s[0], pad_w=s[1], str_h=s[2], str_w=s[3];                      \
                                                                               \
    for(size_t idx=gid; idx<num_col_el; idx+=nthreads){                         \
        size_t ow = idx % ow_max;  size_t tmp = idx / ow_max;                   \
        size_t oh = tmp % oh_max;  tmp /= oh_max;                               \
        size_t kw = tmp % kw_max;  tmp /= kw_max;                               \
        size_t kh = tmp % kh_max;  tmp /= kh_max;                               \
        size_t c  = tmp % C;       size_t b  = tmp / C;                         \
                                                                               \
        int h_in = int(oh*str_h) - int(pad_h) + int(kh);                        \
        int w_in = int(ow*str_w) - int(pad_w) + int(kw);                        \
                                                                               \
        T v = get_zero<T>();                                                    \
        if(h_in>=0 && h_in<int(H) && w_in>=0 && w_in<int(W)){                   \
            size_t off = ((b*C+c)*H + size_t(h_in))*W + size_t(w_in);           \
            v = input[off];                                                     \
        }                                                                      \
        col[idx]=v;                                                             \
    }                                                                          \
}

/* ─────────────── col2im (serial add) ─────────────── */
#define COL2IM_KERNEL(T, SUF)                                                  \
kernel void metal_conv2d_col2im_##SUF##_kernel(                                \
    const device T *col [[buffer(0)]],                                         \
    device T *out      [[buffer(1)]],                                          \
    constant size_t &num_col_el [[buffer(2)]],                                 \
    constant size_t *meta [[buffer(3)]],                                       \
    uint gid [[thread_position_in_grid]])                                      \
{                                                                              \
    /* single thread (gid==0) accumulates to avoid atomic ops */               \
    if(gid!=0) return;                                                         \
    const constant size_t *d = meta;                                           \
    const constant size_t *s = meta + 8;                                       \
    size_t C=d[1], H=d[2], W=d[3];                                             \
    size_t kh_max=d[4], kw_max=d[5], oh_max=d[6], ow_max=d[7];                  \
    size_t pad_h=s[0], pad_w=s[1], str_h=s[2], str_w=s[3];                      \
                                                                               \
    /* zero‑fill output */                                                     \
    size_t out_size = d[0]*C*H*W;                                              \
    for(size_t i=0;i<out_size;++i) out[i]=get_zero<T>();                        \
                                                                               \
    thread size_t idx=0;                                                       \
    for(idx=0; idx<num_col_el; ++idx){                                         \
        size_t ow = idx % ow_max;  size_t tmp = idx / ow_max;                  \
        size_t oh = tmp % oh_max;  tmp /= oh_max;                              \
        size_t kw = tmp % kw_max;  tmp /= kw_max;                              \
        size_t kh = tmp % kh_max;  tmp /= kh_max;                              \
        size_t c  = tmp % C;       size_t b  = tmp / C;                        \
                                                                               \
        int h_in = int(oh*str_h) - int(pad_h) + int(kh);                       \
        int w_in = int(ow*str_w) - int(pad_w) + int(kw);                       \
        if(h_in>=0 && h_in<int(H) && w_in>=0 && w_in<int(W)){                  \
            size_t off = ((b*C+c)*H + size_t(h_in))*W + size_t(w_in);          \
            out[off] += col[idx];                                              \
        }                                                                      \
    }                                                                          \
}

/* ────── instantiate (float / half / bfloat) ────── */
IM2COL_KERNEL(float,f32)   COL2IM_KERNEL(float,f32)
IM2COL_KERNEL(half,f16)    COL2IM_KERNEL(half,f16)
IM2COL_KERNEL(bfloat,bf16) COL2IM_KERNEL(bfloat,bf16)
