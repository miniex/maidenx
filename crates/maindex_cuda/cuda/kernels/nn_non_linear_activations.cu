#include "nn_non_linear_activations.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",                   \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(err);                                                               \
    }                                                                          \
  }

// ReLU Forward Kernel
__global__ void relu_forward_kernel(float *output, const float *input,
                                    size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(input[idx], 0.0f);
  }
}

// ReLU Backward Kernel
__global__ void relu_backward_kernel(float *grad_input,
                                     const float *grad_output,
                                     const float *input, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
  }
}

// Sigmoid Forward Kernel
__global__ void sigmoid_forward_kernel(float *output, const float *input,
                                       size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

// Sigmoid Backward Kernel
__global__ void sigmoid_backward_kernel(float *grad_input,
                                        const float *grad_output,
                                        const float *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
  }
}

// Tanh Forward Kernel
__global__ void tanh_forward_kernel(float *output, const float *input,
                                    size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanhf(input[idx]);
  }
}

// Tanh Backward Kernel
__global__ void tanh_backward_kernel(float *grad_input,
                                     const float *grad_output,
                                     const float *output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
  }
}

// ReLU Forward
void relu_forward(float *output, const float *input, size_t size,
                  cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  relu_forward_kernel<<<blocks, threads, 0, stream>>>(output, input, size);
  CUDA_CHECK(cudaGetLastError());
}

// ReLU Backward
void relu_backward(float *grad_input, const float *grad_output,
                   const float *input, size_t size, cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  relu_backward_kernel<<<blocks, threads, 0, stream>>>(grad_input, grad_output,
                                                       input, size);
  CUDA_CHECK(cudaGetLastError());
}

// Sigmoid Forward
void sigmoid_forward(float *output, const float *input, size_t size,
                     cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  sigmoid_forward_kernel<<<blocks, threads, 0, stream>>>(output, input, size);
  CUDA_CHECK(cudaGetLastError());
}

// Sigmoid Backward
void sigmoid_backward(float *grad_input, const float *grad_output,
                      const float *output, size_t size, cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  sigmoid_backward_kernel<<<blocks, threads, 0, stream>>>(
      grad_input, grad_output, output, size);
  CUDA_CHECK(cudaGetLastError());
}

// Tanh Forward
void tanh_forward(float *output, const float *input, size_t size,
                  cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  tanh_forward_kernel<<<blocks, threads, 0, stream>>>(output, input, size);
  CUDA_CHECK(cudaGetLastError());
}

// Tanh Backward
void tanh_backward(float *grad_input, const float *grad_output,
                   const float *output, size_t size, cudaStream_t stream) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  tanh_backward_kernel<<<blocks, threads, 0, stream>>>(grad_input, grad_output,
                                                       output, size);
  CUDA_CHECK(cudaGetLastError());
}
