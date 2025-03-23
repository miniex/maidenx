use half::{bf16, f16};
use rayon::prelude::*;
use std::cmp::{max, min};
use std::sync::{Arc, Mutex};

macro_rules! pad_with_constant_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions for input
        ///   - output_dims[num_dims]: dimensions for output array
        ///   - paddings[num_dims * 2]: padding values for each dimension (before, after)
        ///   - pad_value: value to use for padding
        /// * `inp` must be a valid pointer to an array of at least `num_els_in` elements
        /// * `out` must be a valid pointer to an array of at least `num_els_out` elements
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
            pad_value: $type,
        ) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let paddings = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims * 2);

            // Copy data to Vec for safe multi-threading
            let input_vec = std::slice::from_raw_parts(inp, num_els_in).to_vec();
            let mut output_vec = vec![$zero; num_els_out];

            // Initialize output with pad value
            for i in 0..num_els_out {
                output_vec[i] = pad_value;
            }

            // Create chunks of work for better parallelism
            let chunk_size = (num_els_out / rayon::current_num_threads()) + 1;
            output_vec.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;

                for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                    let i = start_idx + local_idx;
                    if i >= num_els_out {
                        break;
                    }

                    // Calculate coordinates in output tensor
                    let mut out_coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        out_coords[d] = tmp_i % output_dims[d];
                        tmp_i /= output_dims[d];
                    }

                    // Calculate corresponding coordinates in input tensor
                    let mut in_bounds = true;
                    let mut in_coords = vec![0; num_dims];

                    for d in 0..num_dims {
                        let pad_before = paddings[d * 2];
                        in_coords[d] = out_coords[d] as isize - pad_before as isize;

                        // Check if this coordinate is within input bounds
                        if in_coords[d] < 0 || in_coords[d] >= input_dims[d] as isize {
                            in_bounds = false;
                            break;
                        }
                    }

                    // If within bounds, copy from input to output
                    if in_bounds {
                        // Calculate linear index for input
                        let mut in_idx = 0;
                        let mut stride = 1;
                        for d in (0..num_dims).rev() {
                            in_idx += in_coords[d] as usize * stride;
                            stride *= input_dims[d];
                        }

                        // Copy value from input to output
                        *out_val = input_vec[in_idx];
                    }
                }
            });

            // Copy back to output pointer
            std::ptr::copy_nonoverlapping(output_vec.as_ptr(), out, num_els_out);
        }
    };
}

macro_rules! pad_with_reflection_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions for input
        ///   - output_dims[num_dims]: dimensions for output array
        ///   - paddings[num_dims * 2]: padding values for each dimension (before, after)
        /// * `inp` must be a valid pointer to an array of at least `num_els_in` elements
        /// * `out` must be a valid pointer to an array of at least `num_els_out` elements
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(num_els_in: usize, num_els_out: usize, num_dims: usize, metadata: *const usize, inp: *const $type, out: *mut $type) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let paddings = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims * 2);

            // Copy data to Vec for safe multi-threading
            let input_vec = std::slice::from_raw_parts(inp, num_els_in).to_vec();
            let mut output_vec = vec![$zero; num_els_out];

            // Create chunks of work for better parallelism
            let chunk_size = (num_els_out / rayon::current_num_threads()) + 1;
            output_vec.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;

                for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                    let i = start_idx + local_idx;
                    if i >= num_els_out {
                        break;
                    }

                    // Calculate coordinates in output tensor
                    let mut out_coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        out_coords[d] = tmp_i % output_dims[d];
                        tmp_i /= output_dims[d];
                    }

                    // Calculate corresponding coordinates in input tensor with reflection
                    let mut in_coords = vec![0; num_dims];

                    for d in 0..num_dims {
                        let pad_before = paddings[d * 2] as isize;
                        let dim_size = input_dims[d] as isize;

                        // Get position relative to the padded area
                        let mut pos = out_coords[d] as isize - pad_before;

                        // Apply correct reflection padding
                        // For positions outside the input bounds
                        if pos < 0 {
                            // Reflection logic for negative positions
                            pos = -pos; // First reflection at 0
                        } else if pos >= dim_size {
                            // Reflection logic for positions beyond last element
                            pos = 2 * dim_size - pos - 2; // Reflection at (dim_size-1)
                        }

                        // Handle multiple reflections if needed
                        while pos < 0 || pos >= dim_size {
                            if pos < 0 {
                                pos = -pos; // Reflect at 0
                            } else if pos >= dim_size {
                                pos = 2 * dim_size - pos - 2; // Reflect at (dim_size-1)
                            }
                        }

                        in_coords[d] = pos as usize;
                    }

                    // Calculate linear index for input
                    let mut in_idx = 0;
                    let mut stride = 1;
                    for d in (0..num_dims).rev() {
                        in_idx += in_coords[d] * stride;
                        stride *= input_dims[d];
                    }

                    // Copy value from input to output
                    *out_val = input_vec[in_idx];
                }
            });

            // Copy back to output pointer
            std::ptr::copy_nonoverlapping(output_vec.as_ptr(), out, num_els_out);
        }
    };
}

macro_rules! pad_with_replication_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions for input
        ///   - output_dims[num_dims]: dimensions for output array
        ///   - paddings[num_dims * 2]: padding values for each dimension (before, after)
        /// * `inp` must be a valid pointer to an array of at least `num_els_in` elements
        /// * `out` must be a valid pointer to an array of at least `num_els_out` elements
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(num_els_in: usize, num_els_out: usize, num_dims: usize, metadata: *const usize, inp: *const $type, out: *mut $type) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let paddings = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims * 2);

            // Copy data to Vec for safe multi-threading
            let input_vec = std::slice::from_raw_parts(inp, num_els_in).to_vec();
            let mut output_vec = vec![$zero; num_els_out];

            // Create chunks of work for better parallelism
            let chunk_size = (num_els_out / rayon::current_num_threads()) + 1;
            output_vec.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;

                for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                    let i = start_idx + local_idx;
                    if i >= num_els_out {
                        break;
                    }

                    // Calculate coordinates in output tensor
                    let mut out_coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        out_coords[d] = tmp_i % output_dims[d];
                        tmp_i /= output_dims[d];
                    }

                    // Calculate corresponding coordinates in input tensor with replication
                    let mut in_coords = vec![0; num_dims];

                    for d in 0..num_dims {
                        let pad_before = paddings[d * 2];
                        let mut pos = out_coords[d] as isize - pad_before as isize;

                        // Apply replication (clamp to valid range)
                        pos = max(0, min(pos, input_dims[d] as isize - 1));

                        in_coords[d] = pos as usize;
                    }

                    // Calculate linear index for input
                    let mut in_idx = 0;
                    let mut stride = 1;
                    for d in (0..num_dims).rev() {
                        in_idx += in_coords[d] * stride;
                        stride *= input_dims[d];
                    }

                    // Copy value from input to output
                    *out_val = input_vec[in_idx];
                }
            });

            // Copy back to output pointer
            std::ptr::copy_nonoverlapping(output_vec.as_ptr(), out, num_els_out);
        }
    };
}

// Backward operations
macro_rules! pad_with_constant_backward_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions for input gradient
        ///   - output_dims[num_dims]: dimensions for output gradient array
        ///   - paddings[num_dims * 2]: padding values for each dimension (before, after)
        /// * `grad_out` must be a valid pointer to an array of at least `num_els_out` elements (gradient of output)
        /// * `grad_in` must be a valid pointer to an array of at least `num_els_in` elements (gradient of input)
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const usize,
            grad_out: *const $type,
            grad_in: *mut $type,
        ) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let paddings = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims * 2);

            // Copy data to Vec for safe multi-threading
            let grad_output_vec = std::slice::from_raw_parts(grad_out, num_els_out).to_vec();
            let grad_input_vec = vec![$zero; num_els_in];

            // Create chunks of work for better parallelism
            let chunk_size = (num_els_out / rayon::current_num_threads()) + 1;

            // Use Arc<Mutex<Vec>> for thread-safe accumulation
            let grad_input_shared = Arc::new(Mutex::new(grad_input_vec));

            (0..num_els_out).into_par_iter().chunks(chunk_size).for_each(|chunk| {
                // Local accumulation buffer to minimize mutex contention
                let mut local_grad_input = vec![$zero; num_els_in];

                for i in chunk {
                    // Calculate coordinates in output tensor
                    let mut out_coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        out_coords[d] = tmp_i % output_dims[d];
                        tmp_i /= output_dims[d];
                    }

                    // Calculate corresponding coordinates in input tensor
                    let mut in_bounds = true;
                    let mut in_coords = vec![0; num_dims];

                    for d in 0..num_dims {
                        let pad_before = paddings[d * 2];
                        in_coords[d] = out_coords[d] as isize - pad_before as isize;

                        // Check if this coordinate is within input bounds
                        if in_coords[d] < 0 || in_coords[d] >= input_dims[d] as isize {
                            in_bounds = false;
                            break;
                        }
                    }

                    // If within bounds, accumulate gradient
                    if in_bounds {
                        // Calculate linear index for input
                        let mut in_idx = 0;
                        let mut stride = 1;
                        for d in (0..num_dims).rev() {
                            in_idx += in_coords[d] as usize * stride;
                            stride *= input_dims[d];
                        }

                        // Accumulate in local buffer
                        local_grad_input[in_idx] += grad_output_vec[i];
                    }
                }

                // Merge local results into shared buffer
                let mut shared_grad = grad_input_shared.lock().unwrap();
                for (i, val) in local_grad_input.iter().enumerate() {
                    shared_grad[i] += *val;
                }
            });

            // Copy results back to grad_in
            let final_grad = grad_input_shared.lock().unwrap();
            std::ptr::copy_nonoverlapping(final_grad.as_ptr(), grad_in, num_els_in);
        }
    };
}

macro_rules! pad_with_reflection_backward_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions for input gradient
        ///   - output_dims[num_dims]: dimensions for output gradient array
        ///   - paddings[num_dims * 2]: padding values for each dimension (before, after)
        /// * `grad_out` must be a valid pointer to an array of at least `num_els_out` elements (gradient of output)
        /// * `grad_in` must be a valid pointer to an array of at least `num_els_in` elements (gradient of input)
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const usize,
            grad_out: *const $type,
            grad_in: *mut $type,
        ) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let paddings = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims * 2);

            // Copy data to Vec for safe multi-threading
            let grad_output_vec = std::slice::from_raw_parts(grad_out, num_els_out).to_vec();
            let grad_input_vec = vec![$zero; num_els_in];

            // Use Arc<Mutex<Vec>> for thread-safe accumulation
            let grad_input_shared = Arc::new(Mutex::new(grad_input_vec));

            // Create chunks of work for better parallelism
            let chunk_size = (num_els_out / rayon::current_num_threads()) + 1;

            (0..num_els_out).into_par_iter().chunks(chunk_size).for_each(|chunk| {
                // Local accumulation buffer to minimize mutex contention
                let mut local_grad_input = vec![$zero; num_els_in];

                for i in chunk {
                    // Calculate coordinates in output tensor
                    let mut out_coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        out_coords[d] = tmp_i % output_dims[d];
                        tmp_i /= output_dims[d];
                    }

                    // Calculate corresponding coordinates in input tensor with reflection
                    let mut in_coords = vec![0; num_dims];

                    for d in 0..num_dims {
                        let pad_before = paddings[d * 2] as isize;
                        let dim_size = input_dims[d] as isize;

                        // Get position relative to the padded area
                        let mut pos = out_coords[d] as isize - pad_before;

                        // Apply correct reflection padding
                        // Basic reflection algorithm
                        if pos < 0 {
                            // Reflection for positions before array start
                            pos = -pos; // First reflection at 0
                        } else if pos >= dim_size {
                            // Reflection for positions after array end
                            pos = 2 * dim_size - pos - 2; // Reflect at (dim_size-1)
                        }

                        // Handle multiple reflections if needed
                        while pos < 0 || pos >= dim_size {
                            if pos < 0 {
                                pos = -pos; // Reflect at 0
                            } else if pos >= dim_size {
                                pos = 2 * dim_size - pos - 2; // Reflect at (dim_size-1)
                            }
                        }

                        in_coords[d] = pos as usize;
                    }

                    // Calculate linear index for input
                    let mut in_idx = 0;
                    let mut stride = 1;
                    for d in (0..num_dims).rev() {
                        in_idx += in_coords[d] * stride;
                        stride *= input_dims[d];
                    }

                    // Accumulate in local buffer
                    local_grad_input[in_idx] += grad_output_vec[i];
                }

                // Merge local results into shared buffer
                let mut shared_grad = grad_input_shared.lock().unwrap();
                for (i, val) in local_grad_input.iter().enumerate() {
                    shared_grad[i] += *val;
                }
            });

            // Copy results back to grad_in
            let final_grad = grad_input_shared.lock().unwrap();
            std::ptr::copy_nonoverlapping(final_grad.as_ptr(), grad_in, num_els_in);
        }
    };
}

macro_rules! pad_with_replication_backward_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions for input gradient
        ///   - output_dims[num_dims]: dimensions for output gradient array
        ///   - paddings[num_dims * 2]: padding values for each dimension (before, after)
        /// * `grad_out` must be a valid pointer to an array of at least `num_els_out` elements (gradient of output)
        /// * `grad_in` must be a valid pointer to an array of at least `num_els_in` elements (gradient of input)
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const usize,
            grad_out: *const $type,
            grad_in: *mut $type,
        ) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let paddings = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims * 2);

            // Copy data to Vec for safe multi-threading
            let grad_output_vec = std::slice::from_raw_parts(grad_out, num_els_out).to_vec();
            let grad_input_vec = vec![$zero; num_els_in];

            // Use Arc<Mutex<Vec>> for thread-safe accumulation
            let grad_input_shared = Arc::new(Mutex::new(grad_input_vec));

            // Create chunks of work for better parallelism
            let chunk_size = (num_els_out / rayon::current_num_threads()) + 1;

            (0..num_els_out).into_par_iter().chunks(chunk_size).for_each(|chunk| {
                // Local accumulation buffer to minimize mutex contention
                let mut local_grad_input = vec![$zero; num_els_in];

                for i in chunk {
                    // Calculate coordinates in output tensor
                    let mut out_coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        out_coords[d] = tmp_i % output_dims[d];
                        tmp_i /= output_dims[d];
                    }

                    // Calculate corresponding coordinates in input tensor with replication
                    let mut in_coords = vec![0; num_dims];

                    for d in 0..num_dims {
                        let pad_before = paddings[d * 2];
                        let mut pos = out_coords[d] as isize - pad_before as isize;

                        // Apply replication (clamp to valid range)
                        pos = max(0, min(pos, input_dims[d] as isize - 1));

                        in_coords[d] = pos as usize;
                    }

                    // Calculate linear index for input
                    let mut in_idx = 0;
                    let mut stride = 1;
                    for d in (0..num_dims).rev() {
                        in_idx += in_coords[d] * stride;
                        stride *= input_dims[d];
                    }

                    // Accumulate in local buffer
                    local_grad_input[in_idx] += grad_output_vec[i];
                }

                // Merge local results into shared buffer
                let mut shared_grad = grad_input_shared.lock().unwrap();
                for (i, val) in local_grad_input.iter().enumerate() {
                    shared_grad[i] += *val;
                }
            });

            // Copy results back to grad_in
            let final_grad = grad_input_shared.lock().unwrap();
            std::ptr::copy_nonoverlapping(final_grad.as_ptr(), grad_in, num_els_in);
        }
    };
}
// Forward operations for different types
pad_with_constant_op!(pad_with_constant_bf16, bf16, bf16::from_f32(0.0));
pad_with_constant_op!(pad_with_constant_f16, f16, f16::from_f32(0.0));
pad_with_constant_op!(pad_with_constant_f32, f32, 0.0f32);
pad_with_constant_op!(pad_with_constant_f64, f64, 0.0f64);
pad_with_constant_op!(pad_with_constant_u8, u8, 0u8);
pad_with_constant_op!(pad_with_constant_u16, u16, 0u16);
pad_with_constant_op!(pad_with_constant_u32, u32, 0u32);
pad_with_constant_op!(pad_with_constant_u64, u64, 0u64);
pad_with_constant_op!(pad_with_constant_i8, i8, 0i8);
pad_with_constant_op!(pad_with_constant_i16, i16, 0i16);
pad_with_constant_op!(pad_with_constant_i32, i32, 0i32);
pad_with_constant_op!(pad_with_constant_i64, i64, 0i64);

pad_with_reflection_op!(pad_with_reflection_bf16, bf16, bf16::from_f32(0.0));
pad_with_reflection_op!(pad_with_reflection_f16, f16, f16::from_f32(0.0));
pad_with_reflection_op!(pad_with_reflection_f32, f32, 0.0f32);
pad_with_reflection_op!(pad_with_reflection_f64, f64, 0.0f64);
pad_with_reflection_op!(pad_with_reflection_u8, u8, 0u8);
pad_with_reflection_op!(pad_with_reflection_u16, u16, 0u16);
pad_with_reflection_op!(pad_with_reflection_u32, u32, 0u32);
pad_with_reflection_op!(pad_with_reflection_u64, u64, 0u64);
pad_with_reflection_op!(pad_with_reflection_i8, i8, 0i8);
pad_with_reflection_op!(pad_with_reflection_i16, i16, 0i16);
pad_with_reflection_op!(pad_with_reflection_i32, i32, 0i32);
pad_with_reflection_op!(pad_with_reflection_i64, i64, 0i64);

pad_with_replication_op!(pad_with_replication_bf16, bf16, bf16::from_f32(0.0));
pad_with_replication_op!(pad_with_replication_f16, f16, f16::from_f32(0.0));
pad_with_replication_op!(pad_with_replication_f32, f32, 0.0f32);
pad_with_replication_op!(pad_with_replication_f64, f64, 0.0f64);
pad_with_replication_op!(pad_with_replication_u8, u8, 0u8);
pad_with_replication_op!(pad_with_replication_u16, u16, 0u16);
pad_with_replication_op!(pad_with_replication_u32, u32, 0u32);
pad_with_replication_op!(pad_with_replication_u64, u64, 0u64);
pad_with_replication_op!(pad_with_replication_i8, i8, 0i8);
pad_with_replication_op!(pad_with_replication_i16, i16, 0i16);
pad_with_replication_op!(pad_with_replication_i32, i32, 0i32);
pad_with_replication_op!(pad_with_replication_i64, i64, 0i64);

// Backward operations for different types
pad_with_constant_backward_op!(pad_with_constant_backward_bf16, bf16, bf16::from_f32(0.0));
pad_with_constant_backward_op!(pad_with_constant_backward_f16, f16, f16::from_f32(0.0));
pad_with_constant_backward_op!(pad_with_constant_backward_f32, f32, 0.0f32);
pad_with_constant_backward_op!(pad_with_constant_backward_f64, f64, 0.0f64);
pad_with_constant_backward_op!(pad_with_constant_backward_u8, u8, 0u8);
pad_with_constant_backward_op!(pad_with_constant_backward_u16, u16, 0u16);
pad_with_constant_backward_op!(pad_with_constant_backward_u32, u32, 0u32);
pad_with_constant_backward_op!(pad_with_constant_backward_u64, u64, 0u64);
pad_with_constant_backward_op!(pad_with_constant_backward_i8, i8, 0i8);
pad_with_constant_backward_op!(pad_with_constant_backward_i16, i16, 0i16);
pad_with_constant_backward_op!(pad_with_constant_backward_i32, i32, 0i32);
pad_with_constant_backward_op!(pad_with_constant_backward_i64, i64, 0i64);

pad_with_reflection_backward_op!(pad_with_reflection_backward_bf16, bf16, bf16::from_f32(0.0));
pad_with_reflection_backward_op!(pad_with_reflection_backward_f16, f16, f16::from_f32(0.0));
pad_with_reflection_backward_op!(pad_with_reflection_backward_f32, f32, 0.0f32);
pad_with_reflection_backward_op!(pad_with_reflection_backward_f64, f64, 0.0f64);
pad_with_reflection_backward_op!(pad_with_reflection_backward_u8, u8, 0u8);
pad_with_reflection_backward_op!(pad_with_reflection_backward_u16, u16, 0u16);
pad_with_reflection_backward_op!(pad_with_reflection_backward_u32, u32, 0u32);
pad_with_reflection_backward_op!(pad_with_reflection_backward_u64, u64, 0u64);
pad_with_reflection_backward_op!(pad_with_reflection_backward_i8, i8, 0i8);
pad_with_reflection_backward_op!(pad_with_reflection_backward_i16, i16, 0i16);
pad_with_reflection_backward_op!(pad_with_reflection_backward_i32, i32, 0i32);
pad_with_reflection_backward_op!(pad_with_reflection_backward_i64, i64, 0i64);

pad_with_replication_backward_op!(pad_with_replication_backward_bf16, bf16, bf16::from_f32(0.0));
pad_with_replication_backward_op!(pad_with_replication_backward_f16, f16, f16::from_f32(0.0));
pad_with_replication_backward_op!(pad_with_replication_backward_f32, f32, 0.0f32);
pad_with_replication_backward_op!(pad_with_replication_backward_f64, f64, 0.0f64);
pad_with_replication_backward_op!(pad_with_replication_backward_u8, u8, 0u8);
pad_with_replication_backward_op!(pad_with_replication_backward_u16, u16, 0u16);
pad_with_replication_backward_op!(pad_with_replication_backward_u32, u32, 0u32);
pad_with_replication_backward_op!(pad_with_replication_backward_u64, u64, 0u64);
pad_with_replication_backward_op!(pad_with_replication_backward_i8, i8, 0i8);
pad_with_replication_backward_op!(pad_with_replication_backward_i16, i16, 0i16);
pad_with_replication_backward_op!(pad_with_replication_backward_i32, i32, 0i32);
pad_with_replication_backward_op!(pad_with_replication_backward_i64, i64, 0i64);
