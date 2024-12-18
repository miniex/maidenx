#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void tensor_mean(float *output, const float *input, size_t size,
                 cudaStream_t stream = 0);
void tensor_sum(float *output, const float *input, size_t size,
                cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
