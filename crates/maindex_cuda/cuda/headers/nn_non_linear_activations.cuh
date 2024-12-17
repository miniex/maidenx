#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ReLU
void relu_forward(float *output, const float *input, size_t size,
                  cudaStream_t stream = 0);
void relu_backward(float *grad_input, const float *grad_output,
                   const float *input, size_t size, cudaStream_t stream = 0);

// Sigmoid
void sigmoid_forward(float *output, const float *input, size_t size,
                     cudaStream_t stream = 0);
void sigmoid_backward(float *grad_input, const float *grad_output,
                      const float *output, size_t size,
                      cudaStream_t stream = 0);

// Tanh
void tanh_forward(float *output, const float *input, size_t size,
                  cudaStream_t stream = 0);
void tanh_backward(float *grad_input, const float *grad_output,
                   const float *output, size_t size, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
