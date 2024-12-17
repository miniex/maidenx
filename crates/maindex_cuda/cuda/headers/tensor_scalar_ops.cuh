#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void tensor_scalar_add(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream = 0);
void tensor_scalar_sub(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream = 0);
void tensor_scalar_mul(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream = 0);
void tensor_scalar_div(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
