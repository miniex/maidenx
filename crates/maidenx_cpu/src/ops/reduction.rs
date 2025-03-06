use half::{bf16, f16};
use rayon::prelude::*;
use std::sync::Mutex;

macro_rules! sum_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - dims[num_dims]: array dimensions
        ///   - strides[num_dims]: strides for input array
        ///   - sum_dims_l[num_sum_dims]: length of dimensions to sum over
        ///   - sum_dims_s[num_sum_dims]: stride of dimensions to sum over
        /// * `inp` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of appropriate size for the output
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(num_els: usize, num_dims: usize, num_red_dims: usize, metadata: *const usize, inp: *const $type, out: *mut $type) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let sum_dims_l = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_red_dims);
            let sum_dims_s = std::slice::from_raw_parts(metadata.add(2 * num_dims + num_red_dims), num_red_dims);

            let offset = *metadata.add(2 * num_dims + 2 * num_red_dims);

            let input = std::slice::from_raw_parts(inp, num_els);

            // Calculate output size
            let mut out_size = num_els;
            for i in 0..num_red_dims {
                out_size /= sum_dims_l[i];
            }
            let output = std::slice::from_raw_parts_mut(out, out_size);

            // Initialize output array with zeros
            output.fill($zero);

            // Wrap output in Mutex for thread-safe access
            let output = output.iter().map(|_| Mutex::new($zero)).collect::<Vec<_>>();

            let is_contiguous = {
                let mut is_cont = true;
                let mut acc = 1;
                for d in (0..num_dims).rev() {
                    if strides[d] != acc {
                        is_cont = false;
                        break;
                    }
                    acc *= dims[d];
                }
                is_cont
            };

            // Process elements in parallel
            (0..num_els).into_par_iter().for_each(|i| {
                let src_idx = if is_contiguous {
                    offset + i
                } else {
                    let mut idx = offset;
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        idx += i_dim * strides[d];
                        tmp_i /= dims[d];
                    }
                    idx
                };

                // Calculate destination index
                let mut dst_idx = i;
                for nd in 0..num_red_dims {
                    let stride = sum_dims_s[nd];
                    let pre = dst_idx / stride;
                    let post = dst_idx % stride;
                    dst_idx = (pre / sum_dims_l[nd]) * stride + post;
                }

                // Atomic add to output
                let val = input[src_idx];
                if let Ok(mut out) = output[dst_idx].lock() {
                    *out += val;
                }
            });

            // Copy results back to output array
            for (i, mutex) in output.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *std::slice::from_raw_parts_mut(out, out_size).get_unchecked_mut(i) = *val;
                }
            }
        }
    };
}

macro_rules! sum_to_shape_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - dims[num_dims]: array dimensions for input
        ///   - strides[num_dims]: strides for input array
        ///   - output_dims[num_dims]: dimensions for output array
        /// * `inp` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of appropriate size for the output
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(num_els: usize, num_dims: usize, metadata: *const usize, inp: *const $type, out: *mut $type) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let input_strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims);

            let offset = *metadata.add(3 * num_dims);

            let input = std::slice::from_raw_parts(inp, num_els);
            let out_size = output_dims.iter().product();
            let output = std::slice::from_raw_parts_mut(out, out_size);

            // Initialize output array with zeros
            output.fill($zero);

            // Calculate reduction factors for each dimension
            let mut reduction_factors = vec![1; num_dims];
            for d in 0..num_dims {
                reduction_factors[d] = input_dims[d] / output_dims[d];
            }

            // Process elements
            for i in 0..num_els {
                let mut coords = vec![0; num_dims];
                let mut tmp_i = i;

                // Calculate input coordinates
                for d in (0..num_dims).rev() {
                    coords[d] = tmp_i % input_dims[d];
                    tmp_i /= input_dims[d];
                }

                // Calculate output coordinates by integer division
                let mut dst_idx = 0;
                for d in 0..num_dims {
                    let out_coord = coords[d] / reduction_factors[d];
                    dst_idx = dst_idx * output_dims[d] + out_coord;
                }

                // Calculate source index using input strides
                let mut src_idx = offset;
                for d in 0..num_dims {
                    src_idx += coords[d] * input_strides[d];
                }

                // Add to output
                output[dst_idx] += input[src_idx];
            }
        }
    };
}

macro_rules! mean_op {
    ($name:ident, $type:ty, $zero:expr, $convert:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - dims[num_dims]: array dimensions
        ///   - strides[num_dims]: strides for input array
        ///   - mean_dims_l[num_mean_dims]: length of dimensions to average over
        ///   - mean_dims_s[num_mean_dims]: stride of dimensions to average over
        /// * `inp` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of appropriate size for the output
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe extern "C" fn $name(
            num_els: usize,
            num_dims: usize,
            num_red_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let mean_dims_l = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_red_dims);
            let mean_dims_s = std::slice::from_raw_parts(metadata.add(2 * num_dims + num_red_dims), num_red_dims);

            let offset = *metadata.add(2 * num_dims + 2 * num_red_dims);

            let input = std::slice::from_raw_parts(inp, num_els);

            // Calculate output size and reduction factor
            let mut out_size = num_els;
            let mut reduction_factor: usize = 1;
            for i in 0..num_red_dims {
                reduction_factor *= mean_dims_l[i];
                out_size /= mean_dims_l[i];
            }
            let output = std::slice::from_raw_parts_mut(out, out_size);

            // Initialize output array with zeros
            output.fill($zero);

            // Wrap output in Mutex for thread-safe access
            let output = output.iter().map(|_| Mutex::new($zero)).collect::<Vec<_>>();

            let is_contiguous = {
                let mut is_cont = true;
                let mut acc = 1;
                for d in (0..num_dims).rev() {
                    if strides[d] != acc {
                        is_cont = false;
                        break;
                    }
                    acc *= dims[d];
                }
                is_cont
            };

            // Process elements in parallel
            (0..num_els).into_par_iter().for_each(|i| {
                let src_idx = if is_contiguous {
                    offset + i
                } else {
                    let mut idx = offset;
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        idx += i_dim * strides[d];
                        tmp_i /= dims[d];
                    }
                    idx
                };

                // Calculate destination index
                let mut dst_idx = i;
                for nd in 0..num_red_dims {
                    let stride = mean_dims_s[nd];
                    let pre = dst_idx / stride;
                    let post = dst_idx % stride;
                    dst_idx = (pre / mean_dims_l[nd]) * stride + post;
                }

                // Atomic add to output
                let val = input[src_idx];
                if let Ok(mut out) = output[dst_idx].lock() {
                    *out += val;
                }
            });

            // Copy results back to output array and compute mean
            let reduction_factor = $convert(reduction_factor);
            for (i, mutex) in output.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *std::slice::from_raw_parts_mut(out, out_size).get_unchecked_mut(i) = *val / reduction_factor;
                }
            }
        }
    };
}

sum_op!(sum_bf16, bf16, bf16::from_f32(0.0));
sum_op!(sum_f16, f16, f16::from_f32(0.0));
sum_op!(sum_f32, f32, 0.0f32);
sum_op!(sum_f64, f64, 0.0f64);
sum_op!(sum_u8, u8, 0u8);
sum_op!(sum_u32, u32, 0u32);
sum_op!(sum_i8, i8, 0i8);
sum_op!(sum_i32, i32, 0i32);
sum_op!(sum_i64, i64, 0i64);

sum_to_shape_op!(sum_to_shape_bf16, bf16, bf16::from_f32(0.0));
sum_to_shape_op!(sum_to_shape_f16, f16, f16::from_f32(0.0));
sum_to_shape_op!(sum_to_shape_f32, f32, 0.0f32);
sum_to_shape_op!(sum_to_shape_f64, f64, 0.0f64);
sum_to_shape_op!(sum_to_shape_u8, u8, 0u8);
sum_to_shape_op!(sum_to_shape_u32, u32, 0u32);
sum_to_shape_op!(sum_to_shape_i8, i8, 0i8);
sum_to_shape_op!(sum_to_shape_i32, i32, 0i32);
sum_to_shape_op!(sum_to_shape_i64, i64, 0i64);

mean_op!(mean_bf16, bf16, bf16::from_f32(0.0), |x: usize| { bf16::from_f32(x as f32) });
mean_op!(mean_f16, f16, f16::from_f32(0.0), |x: usize| f16::from_f32(x as f32));
mean_op!(mean_f32, f32, 0.0f32, |x: usize| x as f32);
mean_op!(mean_f64, f64, 0.0f64, |x: usize| x as f64);
