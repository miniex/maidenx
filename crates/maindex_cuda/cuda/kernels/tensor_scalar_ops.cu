#include "tensor_scalar_ops.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",                   \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(err);                                                               \
    }                                                                          \
  }

// Tensor-scalar addition kernel
__global__ void tensor_scalar_add_kernel(float *output, const float *input,
                                         float scalar, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] + scalar;
  }
}

// Tensor-scalar subtraction kernel
__global__ void tensor_scalar_sub_kernel(float *output, const float *input,
                                         float scalar, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] - scalar;
  }
}

// Tensor-scalar multiplication kernel
__global__ void tensor_scalar_mul_kernel(float *output, const float *input,
                                         float scalar, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] * scalar;
  }
}

// Tensor-scalar division kernel
__global__ void tensor_scalar_div_kernel(float *output, const float *input,
                                         float scalar, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] / scalar;
  }
}

// Tensor pow kernel
__global__ void tensor_pow_kernel(float *output, const float *input,
                                  float exponent, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = powf(input[idx], exponent);
  }
}

// Tensor-scalar operations with CUDA streams
extern "C" {
void tensor_scalar_add(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_scalar_add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input,
                                                                  scalar, size);
}

void tensor_scalar_sub(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_scalar_sub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input,
                                                                  scalar, size);
}

void tensor_scalar_mul(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_scalar_mul_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input,
                                                                  scalar, size);
}

void tensor_scalar_div(float *output, const float *input, float scalar,
                       size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_scalar_div_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input,
                                                                  scalar, size);
}

void tensor_pow(float *output, const float *input, float exponent, size_t size,
                cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_pow_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input,
                                                           exponent, size);
}
}
