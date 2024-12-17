#include "nn_linear_layers.cuh"
#include <algorithm>
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

__global__ void linear_forward_kernel(float *output, const float *input,
                                      const float *weights, const float *bias,
                                      int batch_size, int input_dim,
                                      int output_dim) {
  int batch_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (out_idx < output_dim) {
    float result = 0.0f;

    if (bias != nullptr) {
      result = bias[out_idx];
    }

    const float *weight_row = weights + out_idx * input_dim;
    const float *input_row = input + batch_idx * input_dim;

    for (int i = 0; i < input_dim; i++) {
      result += input_row[i] * weight_row[i];
    }

    output[batch_idx * output_dim + out_idx] = result;
  }
}

__global__ void linear_backward_kernel(const float *d_output,
                                       const float *input, const float *weights,
                                       float *d_input, float *d_weights,
                                       float *d_bias, int batch_size,
                                       int input_dim, int output_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Handle d_bias
  if (d_bias != nullptr &&
      tid < output_dim) { // bias가 nullptr이 아닐 때만 수행
    float bias_grad = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
      bias_grad += d_output[b * output_dim + tid];
    }
    d_bias[tid] = bias_grad;
  }

  // Handle d_weights and d_input
  for (int b = 0; b < batch_size; ++b) {
    if (tid < output_dim * input_dim) {
      int out_idx = tid / input_dim;
      int in_idx = tid % input_dim;

      // d_weights update
      atomicAdd(&d_weights[tid], d_output[b * output_dim + out_idx] *
                                     input[b * input_dim + in_idx]);

      // d_input update
      atomicAdd(&d_input[b * input_dim + in_idx],
                d_output[b * output_dim + out_idx] *
                    weights[out_idx * input_dim + in_idx]);
    }
  }
}

__global__ void bilinear_forward_kernel(float *output, const float *input1,
                                        const float *input2,
                                        const float *weights, int batch_size,
                                        int dim1, int dim2, int output_dim) {
  int batch_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (out_idx < output_dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim1; i++) {
      float val1 = input1[batch_idx * dim1 + i];
      for (int j = 0; j < dim2; j++) {
        float val2 = input2[batch_idx * dim2 + j];
        sum += val1 * val2 * weights[(out_idx * dim1 + i) * dim2 + j];
      }
    }
    output[batch_idx * output_dim + out_idx] = sum;
  }
}

__global__ void
bilinear_backward_kernel(const float *d_output, const float *input1,
                         const float *input2, const float *weights,
                         float *d_input1, float *d_input2, float *d_weights,
                         int batch_size, int dim1, int dim2, int output_dim) {
  int batch_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (out_idx < output_dim) {
    float grad_out = d_output[batch_idx * output_dim + out_idx];
    const float *weights_slice = &weights[out_idx * dim1 * dim2];

    // Gradient for input1
    for (int i = 0; i < dim1; ++i) {
      float grad_input1 = 0.0f;
      for (int j = 0; j < dim2; ++j) {
        grad_input1 +=
            input2[batch_idx * dim2 + j] * weights_slice[i * dim2 + j];
      }
      atomicAdd(&d_input1[batch_idx * dim1 + i], grad_out * grad_input1);
    }

    // Gradient for input2
    for (int j = 0; j < dim2; ++j) {
      float grad_input2 = 0.0f;
      for (int i = 0; i < dim1; ++i) {
        grad_input2 +=
            input1[batch_idx * dim1 + i] * weights_slice[i * dim2 + j];
      }
      atomicAdd(&d_input2[batch_idx * dim2 + j], grad_out * grad_input2);
    }

    // Gradient for weights
    for (int i = 0; i < dim1; ++i) {
      float input1_val = input1[batch_idx * dim1 + i];
      for (int j = 0; j < dim2; ++j) {
        float input2_val = input2[batch_idx * dim2 + j];
        atomicAdd(&d_weights[out_idx * dim1 * dim2 + i * dim2 + j],
                  grad_out * input1_val * input2_val);
      }
    }
  }
}

extern "C" {
void linear_forward(float *output, const float *input, const float *weights,
                    const float *bias, int batch_size, int input_dim,
                    int output_dim, cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(std::min(output_dim, 1024));

  linear_forward_kernel<<<grid, block, 0, stream>>>(
      output, input, weights, bias, batch_size, input_dim, output_dim);
}
void linear_backward(const float *d_output, const float *input,
                     const float *weights, float *d_input, float *d_weights,
                     float *d_bias, int batch_size, int input_dim,
                     int output_dim) {
  dim3 blockDim(256);
  dim3 gridDim((output_dim * input_dim + blockDim.x - 1) / blockDim.x);

  linear_backward_kernel<<<gridDim, blockDim>>>(
      d_output, input, weights, d_input, d_weights, d_bias, batch_size,
      input_dim, output_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void bilinear_forward(float *output, const float *input1, const float *input2,
                      const float *weights, int batch_size, int dim1, int dim2,
                      int output_dim, cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(std::min(output_dim, 1024));

  bilinear_forward_kernel<<<grid, block, 0, stream>>>(
      output, input1, input2, weights, batch_size, dim1, dim2, output_dim);
}
void bilinear_backward(const float *d_output, const float *input1,
                       const float *input2, const float *weights,
                       float *d_input1, float *d_input2, float *d_weights,
                       int batch_size, int dim1, int dim2, int output_dim) {
  dim3 blockDim(1024);
  dim3 gridDim(batch_size);

  bilinear_backward_kernel<<<gridDim, blockDim>>>(
      d_output, input1, input2, weights, d_input1, d_input2, d_weights,
      batch_size, dim1, dim2, output_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
}
}
