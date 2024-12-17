#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void tensor_add(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream = 0);
void tensor_sub(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream = 0);
void tensor_mul(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream = 0);
void tensor_div(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream = 0);

void tensor_mat_mul(float *output, const float *input1, const float *input2,
                    const int M, const int N, const int K,
                    cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
