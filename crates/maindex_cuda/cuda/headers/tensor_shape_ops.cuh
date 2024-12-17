#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simple 2D matrix transpose
void tensor_transpose_2d(float* output, const float* input,
                        size_t rows, size_t cols,
                        cudaStream_t stream = 0);

// N-dimensional tensor transpose along specified dimensions
void tensor_transpose_dim(float* output, const float* input,
                         const size_t* shape, // Array of dimension sizes
                         const int num_dims,  // Number of dimensions
                         const int dim0,      // First dimension to transpose
                         const int dim1,      // Second dimension to transpose
                         cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
