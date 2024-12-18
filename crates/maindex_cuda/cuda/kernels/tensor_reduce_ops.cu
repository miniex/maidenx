#include "tensor_reduce_ops.cuh"
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void tensor_mean_kernel(float *output, const float *input,
                                   size_t size) {
  __shared__ float shared_sum[BLOCK_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared_sum[tid] = 0.0f;
  if (idx < size) {
    shared_sum[tid] = input[idx];
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, shared_sum[0] / size);
  }
}

__global__ void tensor_sum_kernel(float *output, const float *input,
                                  size_t size) {
  __shared__ float shared_sum[BLOCK_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared_sum[tid] = 0.0f;
  if (idx < size) {
    shared_sum[tid] = input[idx];
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, shared_sum[0]);
  }
}

__device__ void compute_indices(int idx, const int *shape, int ndim, int dim,
                                int *coords) {
  // Compute indices excluding the reduction dimension
  int remaining = idx;
  for (int i = 0; i < ndim; ++i) {
    if (i == dim) {
      coords[i] = 0; // Set reduction dimension index to 0
    } else {
      coords[i] = remaining % shape[i];
      remaining /= shape[i];
    }
  }
}

__global__ void tensor_sum_with_dim_kernel(float *output, const float *input,
                                           const int *shape, int ndim, int dim,
                                           int output_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= output_size)
    return;

  // Output coordinates
  int coords[8] = {0};
  int remaining = tid;

  // Compute coordinates excluding reduction dimension
  for (int i = ndim - 1; i >= 0; --i) {
    if (i == dim) {
      coords[i] = 0; // Reduction dimension is 0 in output
    } else {
      coords[i] = remaining % shape[i];
      remaining /= shape[i];
    }
  }

  // Calculate input base index
  int input_idx = 0;
  int stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    input_idx += coords[i] * stride;
    stride *= shape[i];
  }

  // Sum along the reduction dimension
  float sum = 0.0f;
  stride = 1;
  for (int i = dim + 1; i < ndim; ++i) {
    stride *= shape[i];
  }
  for (int i = 0; i < shape[dim]; ++i) {
    sum += input[input_idx + i * stride];
  }

  output[tid] = sum;
}

extern "C" {
void tensor_mean(float *output, const float *input, size_t size,
                 cudaStream_t stream) {
  float zero = 0.0f;
  cudaMemcpyAsync(output, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);

  int block_size = BLOCK_SIZE;
  int num_blocks = (size + block_size - 1) / block_size;
  tensor_mean_kernel<<<num_blocks, block_size, 0, stream>>>(output, input,
                                                            size);
}

void tensor_sum(float *output, const float *input, size_t size,
                cudaStream_t stream) {
  float zero = 0.0f;
  cudaMemcpyAsync(output, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);

  int block_size = BLOCK_SIZE;
  int num_blocks = (size + block_size - 1) / block_size;
  tensor_sum_kernel<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

void tensor_sum_with_dim(float *output, const float *input, const int *shape,
                         int ndim, int dim, cudaStream_t stream) {
  // Calculate output size
  int output_size = 1;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_size *= shape[i];
    }
  }

  // Initialize output buffer to zeros
  cudaMemsetAsync(output, 0, output_size * sizeof(float), stream);

  // Launch kernel
  int block_size = 256;
  int num_blocks = (output_size + block_size - 1) / block_size;

  // Copy shape to device
  int *d_shape;
  cudaMalloc(&d_shape, ndim * sizeof(int));
  cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice,
                  stream);

  tensor_sum_with_dim_kernel<<<num_blocks, block_size, 0, stream>>>(
      output, input, d_shape, ndim, dim, output_size);

  // Free device memory
  cudaFree(d_shape);
}
}

