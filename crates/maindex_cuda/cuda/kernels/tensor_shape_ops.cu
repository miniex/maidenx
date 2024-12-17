#include "tensor_shape_ops.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",                   \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(err);                                                               \
    }                                                                          \
  }

// Kernel for 2D matrix transpose
__global__ void tensor_transpose_2d_kernel(float *output, const float *input,
                                           size_t rows, size_t cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < cols && idy < rows) {
    output[idx * rows + idy] = input[idy * cols + idx];
  }
}

// Helper function to calculate stride for each dimension
__device__ void calculate_strides(const size_t *shape, size_t *strides,
                                  int num_dims) {
  strides[num_dims - 1] = 1;
  for (int i = num_dims - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

// Kernel for n-dimensional tensor transpose
__global__ void tensor_transpose_dim_kernel(float *output, const float *input,
                                            const size_t *shape,
                                            const int num_dims, const int dim0,
                                            const int dim1) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate total size and strides
  size_t total_size = 1;
  size_t input_strides[32];
  size_t output_shape[32];
  size_t output_strides[32];

  // Copy shape and calculate total size
  for (int i = 0; i < num_dims; ++i) {
    total_size *= shape[i];
    output_shape[i] = shape[i];
  }

  if (tid >= total_size)
    return;

  // Swap dimensions in output shape
  size_t temp = output_shape[dim0];
  output_shape[dim0] = output_shape[dim1];
  output_shape[dim1] = temp;

  // Calculate strides for input and output
  calculate_strides(shape, input_strides, num_dims);
  calculate_strides(output_shape, output_strides, num_dims);

  // Calculate coordinates for input index
  size_t coords[32];
  size_t remaining = tid;
  for (int i = 0; i < num_dims; ++i) {
    coords[i] = remaining / input_strides[i];
    remaining %= input_strides[i];
  }

  // Swap coordinates for output
  temp = coords[dim0];
  coords[dim0] = coords[dim1];
  coords[dim1] = temp;

  // Calculate output index using output strides
  size_t output_idx = 0;
  for (int i = 0; i < num_dims; ++i) {
    output_idx += coords[i] * output_strides[i];
  }

  // Perform transpose
  output[output_idx] = input[tid];
}

extern "C" {

void tensor_transpose_2d(float *output, const float *input, size_t rows,
                         size_t cols, cudaStream_t stream) {
  dim3 block_size(16, 16);
  dim3 grid_size((cols + block_size.x - 1) / block_size.x,
                 (rows + block_size.y - 1) / block_size.y);

  tensor_transpose_2d_kernel<<<grid_size, block_size, 0, stream>>>(
      output, input, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void tensor_transpose_dim(float *output, const float *input,
                          const size_t *shape, const int num_dims,
                          const int dim0, const int dim1, cudaStream_t stream) {
  if (dim0 < 0 || dim0 >= num_dims || dim1 < 0 || dim1 >= num_dims) {
    fprintf(stderr, "Invalid dimensions for transpose\n");
    return;
  }

  // Calculate total size
  size_t total_size = 1;
  for (int i = 0; i < num_dims; ++i) {
    total_size *= shape[i];
  }

  // Allocate and copy shape array to device
  size_t *d_shape;
  CUDA_CHECK(cudaMalloc(&d_shape, num_dims * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpyAsync(d_shape, shape, num_dims * sizeof(size_t),
                             cudaMemcpyHostToDevice, stream));

  // Launch kernel
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  tensor_transpose_dim_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      output, input, d_shape, num_dims, dim0, dim1);
  CUDA_CHECK(cudaGetLastError());

  // Clean up
  CUDA_CHECK(cudaFreeAsync(d_shape, stream));
}
}
