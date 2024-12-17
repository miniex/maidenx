#include "tensor_basic_ops.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

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

// Tensor addition kernel
__global__ void tensor_add_kernel(float *output, const float *input1,
                                  const float *input2, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] + input2[idx];
  }
}

// Tensor subtraction kernel
__global__ void tensor_sub_kernel(float *output, const float *input1,
                                  const float *input2, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] - input2[idx];
  }
}

// Tensor multiplication kernel
__global__ void tensor_mul_kernel(float *output, const float *input1,
                                  const float *input2, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] * input2[idx];
  }
}

// Tensor division kernel
__global__ void tensor_div_kernel(float *output, const float *input1,
                                  const float *input2, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] / input2[idx];
  }
}

// Matrix multiplication kernel
__global__ void tensor_mat_mul_kernel(float *output, const float *input1,
                                      const float *input2, const int M,
                                      const int N, const int K) {
  // Shared memory with padding to avoid bank conflicts
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // Load tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      tile_A[threadIdx.y][threadIdx.x] =
          input1[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (t * TILE_SIZE + threadIdx.y < K && col < N) {
      tile_B[threadIdx.y][threadIdx.x] =
          input2[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute the product of the tiles
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write the result to global memory
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

// Tensor operations with CUDA streams
extern "C" {
void tensor_add(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input1,
                                                           input2, size);
}

void tensor_sub(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_sub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input1,
                                                           input2, size);
}

void tensor_mul(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_mul_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input1,
                                                           input2, size);
}

void tensor_div(float *output, const float *input1, const float *input2,
                size_t size, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tensor_div_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, input1,
                                                           input2, size);
}

void tensor_mat_mul(float *output, const float *input1, const float *input2,
                    const int M, const int N, const int K,
                    cudaStream_t stream) {
  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 num_blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

  tensor_mat_mul_kernel<<<num_blocks, block_size, 0, stream>>>(output, input1,
                                                               input2, M, N, K);
}
}
