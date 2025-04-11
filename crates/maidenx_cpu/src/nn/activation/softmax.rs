#![allow(clippy::needless_range_loop)]

use half::{bf16, f16};
use rayon::prelude::*;
use std::ops::{Add, Div, Sub};
use std::slice;
use std::sync::{Arc, Mutex};

// Generic traits needed for our operations
pub trait Float:
    Copy + Send + Sync + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Div<Output = Self>
{
    fn neg_infinity() -> Self;
    fn exp(self) -> Self;
}

impl Float for f32 {
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn exp(self) -> Self {
        self.exp()
    }
}

impl Float for f64 {
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    fn exp(self) -> Self {
        self.exp()
    }
}

impl Float for f16 {
    fn neg_infinity() -> Self {
        f16::from_f32(f32::NEG_INFINITY)
    }
    fn exp(self) -> Self {
        f16::from_f32(self.to_f32().exp())
    }
}

impl Float for bf16 {
    fn neg_infinity() -> Self {
        bf16::from_f32(f32::NEG_INFINITY)
    }
    fn exp(self) -> Self {
        bf16::from_f32(self.to_f32().exp())
    }
}

// Helper function to compute product of dimensions
fn product_of_dimensions(dims: &[usize], start: usize, end: usize) -> usize {
    let mut result = 1;
    for i in start..end {
        result *= dims[i];
    }
    result
}

// Macro to define the core softmax computation
macro_rules! softmax_impl {
    ($type:ty, $num_els:expr, $num_dims:expr, $dim:expr, $metadata:expr, $input:expr, $output:expr) => {
        unsafe {
            if $metadata.is_null() || $input.is_null() || $output.is_null() {
                return;
            }

            // Default to last dimension if out of bounds
            let dim = if $dim >= $num_dims { $num_dims - 1 } else { $dim };

            let dims = slice::from_raw_parts($metadata, $num_dims);
            let strides = slice::from_raw_parts($metadata.add($num_dims), $num_dims);
            let offset = *$metadata.add(2 * $num_dims);

            let input_slice = slice::from_raw_parts($input, $num_els + offset);
            let output_slice = slice::from_raw_parts_mut($output, $num_els);

            // Calculate sizes for slicing
            let pre_dim_size = product_of_dimensions(dims, 0, dim);
            let dim_size = dims[dim];
            let post_dim_size = product_of_dimensions(dims, dim + 1, $num_dims);

            // Check if the input is contiguous
            let is_contiguous = {
                let mut acc = 1;
                let mut is_cont = true;

                for d in (0..$num_dims).rev() {
                    if strides[d] != acc {
                        is_cont = false;
                        break;
                    }
                    acc *= dims[d];
                }
                is_cont
            };

            // Create a vector of output indices
            let total_slices = pre_dim_size * post_dim_size;
            let indices: Vec<usize> = (0..total_slices).collect();

            // Create a thread-safe vector to collect results
            let results: Arc<Mutex<Vec<(usize, $type)>>> =
                Arc::new(Mutex::new(Vec::with_capacity(total_slices * dim_size)));

            // Process slices in parallel
            indices.par_iter().for_each(|&slice_idx| {
                let pre_idx = slice_idx / post_dim_size;
                let post_idx = slice_idx % post_dim_size;

                // Find max value in this slice for numerical stability
                let mut max_val = <$type>::neg_infinity();
                for i in 0..dim_size {
                    let idx = if is_contiguous {
                        offset + pre_idx * (dim_size * post_dim_size) + i * post_dim_size + post_idx
                    } else {
                        // Calculate index for non-contiguous tensor
                        let mut idx = offset;
                        let mut remaining = pre_idx;
                        for d in 0..dim {
                            let coord = remaining % dims[d];
                            remaining /= dims[d];
                            idx += coord * strides[d];
                        }
                        idx += i * strides[dim];

                        let mut remaining = post_idx;
                        for d in (dim + 1..$num_dims).rev() {
                            let coord = remaining % dims[d];
                            remaining /= dims[d];
                            idx += coord * strides[d];
                        }
                        idx
                    };

                    if input_slice[idx] > max_val {
                        max_val = input_slice[idx];
                    }
                }

                // Compute sum of exponentials for this slice
                let mut sum = <$type>::exp(<$type>::neg_infinity() - max_val);
                for i in 0..dim_size {
                    let idx = if is_contiguous {
                        offset + pre_idx * (dim_size * post_dim_size) + i * post_dim_size + post_idx
                    } else {
                        // Calculate index for non-contiguous tensor
                        let mut idx = offset;
                        let mut remaining = pre_idx;
                        for d in 0..dim {
                            let coord = remaining % dims[d];
                            remaining /= dims[d];
                            idx += coord * strides[d];
                        }
                        idx += i * strides[dim];

                        let mut remaining = post_idx;
                        for d in (dim + 1..$num_dims).rev() {
                            let coord = remaining % dims[d];
                            remaining /= dims[d];
                            idx += coord * strides[d];
                        }
                        idx
                    };

                    sum += <$type>::exp(input_slice[idx] - max_val);
                }

                // Calculate softmax for each element in this slice
                let mut local_results = Vec::with_capacity(dim_size);

                for i in 0..dim_size {
                    let in_idx = if is_contiguous {
                        offset + pre_idx * (dim_size * post_dim_size) + i * post_dim_size + post_idx
                    } else {
                        // Calculate index for non-contiguous tensor
                        let mut idx = offset;
                        let mut remaining = pre_idx;
                        for d in 0..dim {
                            let coord = remaining % dims[d];
                            remaining /= dims[d];
                            idx += coord * strides[d];
                        }
                        idx += i * strides[dim];

                        let mut remaining = post_idx;
                        for d in (dim + 1..$num_dims).rev() {
                            let coord = remaining % dims[d];
                            remaining /= dims[d];
                            idx += coord * strides[d];
                        }
                        idx
                    };

                    // Calculate output index (assuming contiguous output)
                    let out_idx = pre_idx * (dim_size * post_dim_size) + i * post_dim_size + post_idx;

                    // Store result locally first
                    local_results.push((out_idx, <$type>::exp(input_slice[in_idx] - max_val) / sum));
                }

                // Add results to the shared collection
                let mut results_guard = results.lock().unwrap();
                for result in local_results {
                    results_guard.push(result);
                }
            });

            // Now apply all the results to the output array
            for (idx, value) in Arc::try_unwrap(results).unwrap().into_inner().unwrap() {
                output_slice[idx] = value;
            }
        }
    };
}

// Macro to define softmax functions for different types
macro_rules! define_softmax_function {
    ($name:ident, $type:ty) => {
        /// # Safety
        ///
        /// * `num_els` must be the number of elements in the output tensor.
        /// * `num_dims` must be the number of dimensions in the tensor.
        /// * `dim` must be a valid dimension index (0 <= dim < num_dims).
        /// * `metadata` must be a valid pointer to a layout containing:
        ///   - dims[num_dims]: Tensor dimensions
        ///   - strides[num_dims]: Tensor strides for each dimension
        ///   - offset: Starting offset to the input data
        /// * `input` must be a valid pointer to an array of at least `num_els + offset` elements.
        /// * `output` must be a valid pointer to an array of at least `num_els` elements.
        /// * The memory regions pointed to by `input` and `output` must not overlap.
        #[no_mangle]
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            dim: usize,
            metadata: *const usize,
            input: *const $type,
            output: *mut $type,
        ) {
            softmax_impl!($type, num_els, num_dims, dim, metadata, input, output);
        }
    };
}

// Generate implementations for all supported types
define_softmax_function!(softmax_f32, f32);
define_softmax_function!(softmax_f64, f64);
define_softmax_function!(softmax_f16, f16);
define_softmax_function!(softmax_bf16, bf16);
