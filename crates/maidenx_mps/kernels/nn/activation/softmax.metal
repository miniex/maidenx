#include "../../metal_utils.metal"
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

// Helper function to get negative infinity for different types
template <typename T> T get_neg_infinity();

// Specializations for different types
template <> float get_neg_infinity<float>() { return -FLT_MAX; }
template <> half get_neg_infinity<half>() { return -HALF_MAX; }
template <> bfloat get_neg_infinity<bfloat>() { return bfloat(-FLT_MAX); }

// Helper function to compute exponentials for different types
template <typename T> T get_exp(T x);

// Specializations for different types
template <> float get_exp<float>(float x) { return exp(x); }
template <> half get_exp<half>(half x) { return half(exp(float(x))); }
template <> bfloat get_exp<bfloat>(bfloat x) { return bfloat(exp(float(x))); }

// Helper function to compute product of dimensions
size_t product_of_dimensions(const constant size_t *dims, size_t start, size_t end) {
    size_t result = 1;
    for (size_t i = start; i < end; ++i) {
        result *= dims[i];
    }
    return result;
}

#define SOFTMAX_OP(TYPENAME, FN_NAME)                                         \
  kernel void metal_softmax_##FN_NAME##_kernel(                               \
      const device TYPENAME *input [[buffer(0)]],                             \
      device TYPENAME *output [[buffer(1)]],                                  \
      constant size_t& num_els [[buffer(2)]],                                 \
      constant size_t& num_dims [[buffer(3)]],                                \
      constant size_t& dim [[buffer(4)]],                                     \
      constant size_t *metadata [[buffer(5)]],                                \
      uint thread_index [[thread_position_in_grid]],                          \
      uint threads_per_grid [[threads_per_grid]]) {                           \
                                                                              \
    /* Default to last dimension if out of bounds */                          \
    const size_t actual_dim = (dim >= num_dims) ? (num_dims - 1) : dim;       \
                                                                              \
    const constant size_t *dims = metadata;                                   \
    const constant size_t *strides = metadata + num_dims;                     \
    const size_t offset = metadata[2 * num_dims];                             \
                                                                              \
    /* Calculate sizes for slicing */                                         \
    const size_t pre_dim_size = product_of_dimensions(dims, 0, actual_dim);   \
    const size_t dim_size = dims[actual_dim];                                 \
    const size_t post_dim_size =                                              \
        product_of_dimensions(dims, actual_dim + 1, num_dims);                \
                                                                              \
    /* Check if the input is contiguous */                                    \
    bool is_contiguous = true;                                                \
    {                                                                         \
        size_t acc = 1;                                                       \
        for (size_t d = num_dims; d-- > 0;) {                                 \
            if (strides[d] != acc) {                                          \
                is_contiguous = false;                                        \
                break;                                                        \
            }                                                                 \
            acc *= dims[d];                                                   \
        }                                                                     \
    }                                                                         \
                                                                              \
    const size_t total_slices = pre_dim_size * post_dim_size;                 \
                                                                              \
    for (size_t idx = thread_index; idx < total_slices;                       \
         idx += threads_per_grid) {                                           \
                                                                              \
        const size_t pre_idx = idx / post_dim_size;                           \
        const size_t post_idx = idx % post_dim_size;                          \
                                                                              \
        /* Find max value in this slice for numerical stability */            \
        TYPENAME max_val = get_neg_infinity<TYPENAME>();                      \
                                                                              \
        for (size_t i = 0; i < dim_size; ++i) {                               \
            size_t in_idx;                                                    \
            if (is_contiguous) {                                              \
                in_idx = offset + pre_idx * (dim_size * post_dim_size) +      \
                       i * post_dim_size + post_idx;                          \
            } else {                                                          \
                /* Calculate index for non-contiguous tensor */               \
                in_idx = offset;                                              \
                size_t remaining = pre_idx;                                   \
                for (size_t d = 0; d < actual_dim; ++d) {                     \
                    size_t coord = remaining % dims[d];                       \
                    remaining /= dims[d];                                     \
                    in_idx += coord * strides[d];                             \
                }                                                             \
                in_idx += i * strides[actual_dim];                            \
                                                                              \
                remaining = post_idx;                                         \
                for (size_t d = num_dims; d-- > actual_dim + 1;) {            \
                    size_t coord = remaining % dims[d];                       \
                    remaining /= dims[d];                                     \
                    in_idx += coord * strides[d];                             \
                }                                                             \
            }                                                                 \
                                                                              \
            if (input[in_idx] > max_val) {                                    \
                max_val = input[in_idx];                                      \
            }                                                                 \
        }                                                                     \
                                                                              \
        /* Compute sum of exponentials for this slice */                      \
        TYPENAME sum =                                                        \
            get_exp<TYPENAME>(get_neg_infinity<TYPENAME>() - max_val);        \
                                                                              \
        for (size_t i = 0; i < dim_size; ++i) {                               \
            size_t in_idx;                                                    \
            if (is_contiguous) {                                              \
                in_idx = offset + pre_idx * (dim_size * post_dim_size) +      \
                       i * post_dim_size + post_idx;                          \
            } else {                                                          \
                /* Calculate index for non-contiguous tensor */               \
                in_idx = offset;                                              \
                size_t remaining = pre_idx;                                   \
                for (size_t d = 0; d < actual_dim; ++d) {                     \
                    size_t coord = remaining % dims[d];                       \
                    remaining /= dims[d];                                     \
                    in_idx += coord * strides[d];                             \
                }                                                             \
                in_idx += i * strides[actual_dim];                            \
                                                                              \
                remaining = post_idx;                                         \
                for (size_t d = num_dims; d-- > actual_dim + 1;) {            \
                    size_t coord = remaining % dims[d];                       \
                    remaining /= dims[d];                                     \
                    in_idx += coord * strides[d];                             \
                }                                                             \
            }                                                                 \
                                                                              \
            sum += get_exp<TYPENAME>(input[in_idx] - max_val);                \
        }                                                                     \
                                                                              \
        /* Calculate softmax for each element in this slice */                \
        for (size_t i = 0; i < dim_size; ++i) {                               \
            size_t in_idx;                                                    \
            if (is_contiguous) {                                              \
                in_idx = offset + pre_idx * (dim_size * post_dim_size) +      \
                       i * post_dim_size + post_idx;                          \
            } else {                                                          \
                /* Calculate index for non-contiguous tensor */               \
                in_idx = offset;                                              \
                size_t remaining = pre_idx;                                   \
                for (size_t d = 0; d < actual_dim; ++d) {                     \
                    size_t coord = remaining % dims[d];                       \
                    remaining /= dims[d];                                     \
                    in_idx += coord * strides[d];                             \
                }                                                             \
                in_idx += i * strides[actual_dim];                            \
                                                                              \
                remaining = post_idx;                                         \
                for (size_t d = num_dims; d-- > actual_dim + 1;) {            \
                    size_t coord = remaining % dims[d];                       \
                    remaining /= dims[d];                                     \
                    in_idx += coord * strides[d];                             \
                }                                                             \
            }                                                                 \
                                                                              \
            /* Calculate output index (assuming contiguous output) */         \
            size_t out_idx = pre_idx * (dim_size * post_dim_size) +           \
                           i * post_dim_size + post_idx;                      \
                                                                              \
            output[out_idx] = get_exp<TYPENAME>(input[in_idx] - max_val) / sum;\
        }                                                                     \
    }                                                                         \
  }

SOFTMAX_OP(float, f32)
SOFTMAX_OP(half, f16)
SOFTMAX_OP(bfloat, bf16)
