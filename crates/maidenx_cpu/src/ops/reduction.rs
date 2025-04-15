use crate::utils::{get_strided_index, is_contiguous};
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
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            num_sum_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let sum_dims_l = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_sum_dims);
            let sum_dims_s = std::slice::from_raw_parts(metadata.add(2 * num_dims + num_sum_dims), num_sum_dims);
            let offset = *metadata.add(2 * num_dims + 2 * num_sum_dims);

            let inp = std::slice::from_raw_parts(inp.add(offset), num_els);

            // Calculate output size
            let mut out_size = num_els;
            for i in 0..num_sum_dims {
                out_size /= sum_dims_l[i];
            }
            let out_slice = std::slice::from_raw_parts_mut(out, out_size);
            // Initialize output array with zeros
            out_slice.fill($zero);
            // Wrap output in Mutex for thread-safe access
            let out_slice = out_slice.iter().map(|_| Mutex::new($zero)).collect::<Vec<_>>();

            if is_contiguous(num_dims, dims, strides) {
                (0..num_els).into_par_iter().for_each(|i| {
                    let mut dst_idx = i;
                    for nd in 0..num_sum_dims {
                        let stride = sum_dims_s[nd];
                        let pre = dst_idx / stride;
                        let post = dst_idx % stride;
                        dst_idx = (pre / sum_dims_l[nd]) * stride + post;
                    }

                    if let Ok(mut out) = out_slice[dst_idx].lock() {
                        *out += inp[i];
                    }
                });
            } else {
                (0..num_els).into_par_iter().for_each(|i| {
                    let strided_i = get_strided_index(i, num_dims, dims, strides);
                    let mut dst_idx = i;
                    for nd in 0..num_sum_dims {
                        let stride = sum_dims_s[nd];
                        if stride != 0 {
                            let pre = dst_idx / stride;
                            let post = dst_idx % stride;
                            dst_idx = (pre / sum_dims_l[nd]) * stride + post;
                        } else {
                            dst_idx /= sum_dims_l[nd];
                        }
                    }

                    if let Ok(mut out) = out_slice[dst_idx].lock() {
                        *out += inp[strided_i];
                    }
                });
            }

            for (i, mutex) in out_slice.iter().enumerate() {
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
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
        ) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let input_strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let output_dims = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims);
            let offset = *metadata.add(3 * num_dims);

            let input = std::slice::from_raw_parts(inp.add(offset), num_els);

            let out_size = output_dims.iter().product();
            // Initialize output array with zeros using Mutex for thread safety
            let output_mutex = (0..out_size).map(|_| Mutex::new($zero)).collect::<Vec<_>>();

            // Calculate reduction factors for each dimension
            let reduction_factors = (0..num_dims)
                .map(|d| input_dims[d] / output_dims[d])
                .collect::<Vec<_>>();

            if is_contiguous(num_dims, input_dims, input_strides) {
                (0..num_els).into_par_iter().for_each(|i| {
                    let mut coords = vec![0; num_dims];
                    let mut tmp_i = i;

                    for d in (0..num_dims).rev() {
                        coords[d] = tmp_i % input_dims[d];
                        tmp_i /= input_dims[d];
                    }

                    let mut dst_idx = 0;
                    for d in 0..num_dims {
                        let out_coord = coords[d] / reduction_factors[d];
                        dst_idx = dst_idx * output_dims[d] + out_coord;
                    }

                    if let Ok(mut out) = output_mutex[dst_idx].lock() {
                        *out += input[i];
                    }
                });
            } else {
                (0..num_els).into_par_iter().for_each(|i| {
                    let mut coords = vec![0; num_dims];
                    let mut tmp_i = i;

                    for d in (0..num_dims).rev() {
                        coords[d] = tmp_i % input_dims[d];
                        tmp_i /= input_dims[d];
                    }

                    let mut dst_idx = 0;
                    for d in 0..num_dims {
                        let out_coord = coords[d] / reduction_factors[d];
                        dst_idx = dst_idx * output_dims[d] + out_coord;
                    }

                    let mut src_idx = 0;
                    for d in 0..num_dims {
                        src_idx += coords[d] * input_strides[d];
                    }
                    let src_value_idx = (src_idx + offset) % num_els;

                    if let Ok(mut out) = output_mutex[dst_idx].lock() {
                        *out += input[src_value_idx];
                    }
                });
            }

            let output = std::slice::from_raw_parts_mut(out, out_size);
            for (i, mutex) in output_mutex.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *output.get_unchecked_mut(i) = *val;
                }
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

            let inp = std::slice::from_raw_parts(inp.add(offset), num_els);

            // Calculate output size and reduction factor
            let mut out_size = num_els;
            let mut reduction_factor: usize = 1;
            for i in 0..num_red_dims {
                reduction_factor *= mean_dims_l[i];
                out_size /= mean_dims_l[i];
            }
            let out_slice = std::slice::from_raw_parts_mut(out, out_size);
            // Initialize output array with zeros
            out_slice.fill($zero);
            // Wrap output in Mutex for thread-safe access
            let out_slice = out_slice.iter().map(|_| Mutex::new($zero)).collect::<Vec<_>>();

            if is_contiguous(num_dims, dims, strides) {
                (0..num_els).into_par_iter().for_each(|i| {
                    let mut dst_idx = i;
                    for nd in 0..num_red_dims {
                        let stride = mean_dims_s[nd];
                        let pre = dst_idx / stride;
                        let post = dst_idx % stride;
                        dst_idx = (pre / mean_dims_l[nd]) * stride + post;
                    }

                    if let Ok(mut out) = out_slice[dst_idx].lock() {
                        *out += inp[i];
                    }
                });
            } else {
                (0..num_els).into_par_iter().for_each(|i| {
                    let strided_i = get_strided_index(i, num_dims, dims, strides);
                    let mut dst_idx = i;
                    for nd in 0..num_red_dims {
                        let stride = mean_dims_s[nd];
                        if stride != 0 {
                            let pre = dst_idx / stride;
                            let post = dst_idx % stride;
                            dst_idx = (pre / mean_dims_l[nd]) * stride + post;
                        } else {
                            dst_idx /= mean_dims_l[nd];
                        }
                    }

                    if let Ok(mut out) = out_slice[dst_idx].lock() {
                        *out += inp[strided_i];
                    }
                });
            }

            // Copy results back to output array and compute mean
            let reduction_factor = $convert(reduction_factor);
            for (i, mutex) in out_slice.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *std::slice::from_raw_parts_mut(out, out_size).get_unchecked_mut(i) = *val / reduction_factor;
                }
            }
        }
    };
}

macro_rules! fold_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - input_dims[num_dims]: array dimensions
        ///   - input_strides[num_dims]: strides for input array
        ///   - fold_dim: dimension to fold
        ///   - window_dim: window dimension
        ///   - fold_size: size of the folded dimension
        ///   - step: step size for folding
        ///   - window_size: size of the window
        /// * `inp` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of appropriate size for the output
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
        ) {
            let input_dims = std::slice::from_raw_parts(metadata, num_dims);
            let input_strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let fold_dim = *metadata.add(2 * num_dims);
            let window_dim = *metadata.add(2 * num_dims + 1);
            let fold_size = *metadata.add(2 * num_dims + 2);
            let step = *metadata.add(2 * num_dims + 3);
            let _window_size = *metadata.add(2 * num_dims + 4);
            let offset = *metadata.add(2 * num_dims + 5);

            let input = std::slice::from_raw_parts(inp.add(offset), num_els);

            // Calculate output size (product of all input dimensions, replacing the window_dim with fold_size)
            let mut out_size = 1;
            for d in 0..num_dims {
                if d == window_dim {
                    continue; // Skip window dimension
                } else if d == fold_dim {
                    out_size *= fold_size;
                } else {
                    out_size *= input_dims[d];
                }
            }

            let output_mutex = (0..out_size).map(|_| Mutex::new($zero)).collect::<Vec<_>>();

            (0..num_els).into_par_iter().for_each(|i| {
                let coords = if is_contiguous(num_dims, input_dims, input_strides) {
                    let mut coords = vec![0; num_dims];
                    let mut tmp_i = i;
                    for d in (0..num_dims).rev() {
                        coords[d] = tmp_i % input_dims[d];
                        tmp_i /= input_dims[d];
                    }
                    coords
                } else {
                    let mut coords = vec![0; num_dims];
                    let mut remaining = i;
                    for d in (0..num_dims).rev() {
                        coords[d] = remaining % input_dims[d];
                        remaining /= input_dims[d];
                    }
                    coords
                };

                let mut src_idx = 0;
                for d in 0..num_dims {
                    src_idx += coords[d] * input_strides[d];
                }
                src_idx %= num_els;

                let window_idx = coords[fold_dim];
                let pos_in_window = coords[window_dim];

                let orig_pos = window_idx * step + pos_in_window;

                if orig_pos >= fold_size {
                    return;
                }

                let mut dst_idx = 0;
                for d in 0..num_dims {
                    if d == window_dim {
                        continue; // Skip window dimension
                    } else if d == fold_dim {
                        dst_idx = dst_idx * fold_size + orig_pos;
                    } else {
                        dst_idx = dst_idx * input_dims[d] + coords[d];
                    }
                }

                if let Ok(mut out) = output_mutex[dst_idx].lock() {
                    *out += input[src_idx];
                }
            });

            let output = std::slice::from_raw_parts_mut(out, out_size);
            for (i, mutex) in output_mutex.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *output.get_unchecked_mut(i) = *val;
                }
            }
        }
    };
}

macro_rules! max_op {
    ($name:ident, $type:ty, $min_value:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - dims[num_dims]: array dimensions
        ///   - strides[num_dims]: strides for input array
        ///   - max_dims_l[num_max_dims]: length of dimensions to max over
        ///   - max_dims_s[num_max_dims]: stride of dimensions to max over
        /// * `inp` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of appropriate size for the output
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            num_red_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let max_dims_l = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_red_dims);
            let max_dims_s = std::slice::from_raw_parts(metadata.add(2 * num_dims + num_red_dims), num_red_dims);
            let offset = *metadata.add(2 * num_dims + 2 * num_red_dims);

            let input = std::slice::from_raw_parts(inp.add(offset), num_els);

            // Calculate output size
            let mut out_size = num_els;
            for i in 0..num_red_dims {
                out_size /= max_dims_l[i];
            }

            let output_mutex = (0..out_size).map(|_| Mutex::new($min_value)).collect::<Vec<_>>();

            if is_contiguous(num_dims, dims, strides) {
                (0..num_els).into_par_iter().for_each(|i| {
                    let mut dst_idx = i;
                    for nd in 0..num_red_dims {
                        let stride = max_dims_s[nd];
                        let pre = dst_idx / stride;
                        let post = dst_idx % stride;
                        dst_idx = (pre / max_dims_l[nd]) * stride + post;
                    }

                    let val = input[i];
                    if let Ok(mut out) = output_mutex[dst_idx].lock() {
                        *out = if val > *out { val } else { *out };
                    }
                });
            } else {
                (0..num_els).into_par_iter().for_each(|i| {
                    let strided_i = get_strided_index(i, num_dims, dims, strides);

                    let mut dst_idx = i;
                    for nd in 0..num_red_dims {
                        let stride = max_dims_s[nd];
                        if stride != 0 {
                            let pre = dst_idx / stride;
                            let post = dst_idx % stride;
                            dst_idx = (pre / max_dims_l[nd]) * stride + post;
                        } else {
                            dst_idx /= max_dims_l[nd];
                        }
                    }

                    let val = input[strided_i];
                    if let Ok(mut out) = output_mutex[dst_idx].lock() {
                        *out = if val > *out { val } else { *out };
                    }
                });
            }

            let output = std::slice::from_raw_parts_mut(out, out_size);
            for (i, mutex) in output_mutex.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *output.get_unchecked_mut(i) = *val;
                }
            }
        }
    };
}

macro_rules! min_op {
    ($name:ident, $type:ty, $max_value:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to an array containing:
        ///   - dims[num_dims]: array dimensions
        ///   - strides[num_dims]: strides for input array
        ///   - min_dims_l[num_min_dims]: length of dimensions to min over
        ///   - min_dims_s[num_min_dims]: stride of dimensions to min over
        /// * `inp` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of appropriate size for the output
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            num_red_dims: usize,
            metadata: *const usize,
            inp: *const $type,
            out: *mut $type,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let min_dims_l = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_red_dims);
            let min_dims_s = std::slice::from_raw_parts(metadata.add(2 * num_dims + num_red_dims), num_red_dims);
            let offset = *metadata.add(2 * num_dims + 2 * num_red_dims);

            let input = std::slice::from_raw_parts(inp.add(offset), num_els);

            // Calculate output size
            let mut out_size = num_els;
            for i in 0..num_red_dims {
                out_size /= min_dims_l[i];
            }

            let output_mutex = (0..out_size).map(|_| Mutex::new($max_value)).collect::<Vec<_>>();

            if is_contiguous(num_dims, dims, strides) {
                (0..num_els).into_par_iter().for_each(|i| {
                    let mut dst_idx = i;
                    for nd in 0..num_red_dims {
                        let stride = min_dims_s[nd];
                        let pre = dst_idx / stride;
                        let post = dst_idx % stride;
                        dst_idx = (pre / min_dims_l[nd]) * stride + post;
                    }

                    let val = input[i];
                    if let Ok(mut out) = output_mutex[dst_idx].lock() {
                        *out = if val < *out { val } else { *out };
                    }
                });
            } else {
                (0..num_els).into_par_iter().for_each(|i| {
                    let strided_i = get_strided_index(i, num_dims, dims, strides);

                    let mut dst_idx = i;
                    for nd in 0..num_red_dims {
                        let stride = min_dims_s[nd];
                        if stride != 0 {
                            let pre = dst_idx / stride;
                            let post = dst_idx % stride;
                            dst_idx = (pre / min_dims_l[nd]) * stride + post;
                        } else {
                            dst_idx /= min_dims_l[nd];
                        }
                    }

                    let val = input[strided_i];
                    if let Ok(mut out) = output_mutex[dst_idx].lock() {
                        *out = if val < *out { val } else { *out };
                    }
                });
            }

            let output = std::slice::from_raw_parts_mut(out, out_size);
            for (i, mutex) in output_mutex.iter().enumerate() {
                if let Ok(val) = mutex.lock() {
                    *output.get_unchecked_mut(i) = *val;
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
sum_op!(sum_u16, u16, 0u16);
sum_op!(sum_u32, u32, 0u32);
sum_op!(sum_u64, u64, 0u64);
sum_op!(sum_i8, i8, 0i8);
sum_op!(sum_i16, i16, 0i16);
sum_op!(sum_i32, i32, 0i32);
sum_op!(sum_i64, i64, 0i64);

sum_to_shape_op!(sum_to_shape_bf16, bf16, bf16::from_f32(0.0));
sum_to_shape_op!(sum_to_shape_f16, f16, f16::from_f32(0.0));
sum_to_shape_op!(sum_to_shape_f32, f32, 0.0f32);
sum_to_shape_op!(sum_to_shape_f64, f64, 0.0f64);
sum_to_shape_op!(sum_to_shape_u8, u8, 0u8);
sum_to_shape_op!(sum_to_shape_u16, u16, 0u16);
sum_to_shape_op!(sum_to_shape_u32, u32, 0u32);
sum_to_shape_op!(sum_to_shape_u64, u64, 0u64);
sum_to_shape_op!(sum_to_shape_i8, i8, 0i8);
sum_to_shape_op!(sum_to_shape_i16, i16, 0i16);
sum_to_shape_op!(sum_to_shape_i32, i32, 0i32);
sum_to_shape_op!(sum_to_shape_i64, i64, 0i64);

mean_op!(mean_bf16, bf16, bf16::from_f32(0.0), |x: usize| {
    bf16::from_f32(x as f32)
});
mean_op!(mean_f16, f16, f16::from_f32(0.0), |x: usize| f16::from_f32(x as f32));
mean_op!(mean_f32, f32, 0.0f32, |x: usize| x as f32);
mean_op!(mean_f64, f64, 0.0f64, |x: usize| x as f64);

fold_op!(fold_bf16, bf16, bf16::from_f32(0.0));
fold_op!(fold_f16, f16, f16::from_f32(0.0));
fold_op!(fold_f32, f32, 0.0f32);
fold_op!(fold_f64, f64, 0.0f64);
fold_op!(fold_u8, u8, 0u8);
fold_op!(fold_u16, u16, 0u16);
fold_op!(fold_u32, u32, 0u32);
fold_op!(fold_u64, u64, 0u64);
fold_op!(fold_i8, i8, 0i8);
fold_op!(fold_i16, i16, 0i16);
fold_op!(fold_i32, i32, 0i32);
fold_op!(fold_i64, i64, 0i64);

max_op!(max_bf16, bf16, bf16::from_f32(f32::MIN));
max_op!(max_f16, f16, f16::from_f32(f32::MIN));
max_op!(max_f32, f32, f32::MIN);
max_op!(max_f64, f64, f64::MIN);
max_op!(max_u8, u8, u8::MIN);
max_op!(max_u16, u16, u16::MIN);
max_op!(max_u32, u32, u32::MIN);
max_op!(max_u64, u64, u64::MIN);
max_op!(max_i8, i8, i8::MIN);
max_op!(max_i16, i16, i16::MIN);
max_op!(max_i32, i32, i32::MIN);
max_op!(max_i64, i64, i64::MIN);

min_op!(min_bf16, bf16, bf16::from_f32(f32::MAX));
min_op!(min_f16, f16, f16::from_f32(f32::MAX));
min_op!(min_f32, f32, f32::MAX);
min_op!(min_f64, f64, f64::MAX);
min_op!(min_u8, u8, u8::MAX);
min_op!(min_u16, u16, u16::MAX);
min_op!(min_u32, u32, u32::MAX);
min_op!(min_u64, u64, u64::MAX);
min_op!(min_i8, i8, i8::MAX);
min_op!(min_i16, i16, i16::MAX);
min_op!(min_i32, i32, i32::MAX);
min_op!(min_i64, i64, i64::MAX);
