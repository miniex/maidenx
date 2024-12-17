#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Linear Layer Forward and Backward Declarations
void linear_forward(float *output, const float *input, const float *weights,
                    const float *bias, int batch_size, int input_dim,
                    int output_dim, cudaStream_t stream);
void linear_backward(const float *d_output, const float *input,
                     const float *weights, float *d_input, float *d_weights,
                     float *d_bias, int batch_size, int input_dim,
                     int output_dim);

// Bilinear Layer Forward and Backward Declarations

void bilinear_forward(float *output, const float *input1, const float *input2,
                      const float *weights, int batch_size, int dim1, int dim2,
                      int output_dim, cudaStream_t stream);
void bilinear_backward(const float *d_output, const float *input1,
                       const float *input2, const float *weights,
                       float *d_input1, float *d_input2, float *d_weights,
                       int batch_size, int dim1, int dim2, int output_dim);

#ifdef __cplusplus
}
#endif
