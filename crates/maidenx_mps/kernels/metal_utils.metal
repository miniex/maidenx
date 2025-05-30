#include <metal_stdlib>
#include <metal_math>

using namespace metal;

inline bool is_contiguous(const size_t num_dims,
                         constant size_t *dims,
                         constant size_t *strides) {
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

inline unsigned int
get_strided_index(unsigned int idx, const size_t num_dims, constant size_t *dims,
                  constant size_t *strides) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        unsigned int current_dim = dims[dim_idx];
        unsigned int current_stride = strides[dim_idx];

        unsigned int idx_mod_dim = current_dim == 0 ? 0 : (idx % current_dim);

        if (current_stride != 0) {
            strided_i += idx_mod_dim * current_stride;
        }

        idx = current_dim == 0 ? 0 : (idx / current_dim);
    }
    return strided_i;
}

inline unsigned int restrided(const unsigned int strided_i,
                            const size_t num_dims,
                            constant size_t *dims,
                            constant size_t *strides,
                            constant size_t *new_strides) {
    unsigned int idx = 0;
    for (size_t d = 0; d < num_dims; d++) {
        idx += (strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d]) *
               new_strides[d];
    }
    return idx;
}
