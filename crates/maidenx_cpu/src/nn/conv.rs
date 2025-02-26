use half::{bf16, f16};
use rayon::prelude::*;

macro_rules! conv2d_op {
    ($type:ty, $zero:expr) => {
        paste::paste! {
            #[no_mangle]
            /// # Safety
            ///
            /// * `dims_and_strides` must be a valid pointer to array containing:
            ///   - batch_size, channels, height, width, kernel_h, kernel_w, out_h, out_w
            ///   - padding_h, padding_w, stride_h, stride_w
            pub unsafe extern "C" fn [<conv2d_im2col_ $type>](
                num_els: usize,
                dims_and_strides: *const usize,
                input: *const $type,
                col: *mut $type,
            ) {
                if dims_and_strides.is_null() || input.is_null() || col.is_null() {
                    return;
                }

                let batch_size = *dims_and_strides;
                let channels = *dims_and_strides.add(1);
                let height = *dims_and_strides.add(2);
                let width = *dims_and_strides.add(3);
                let kernel_h = *dims_and_strides.add(4);
                let kernel_w = *dims_and_strides.add(5);
                let out_h = *dims_and_strides.add(6);
                let out_w = *dims_and_strides.add(7);
                let pad_h = *dims_and_strides.add(8);
                let pad_w = *dims_and_strides.add(9);
                let stride_h = *dims_and_strides.add(10);
                let stride_w = *dims_and_strides.add(11);

                let input = std::slice::from_raw_parts(input, batch_size * channels * height * width);
                let col = std::slice::from_raw_parts_mut(col, num_els);

                let kernel_size = kernel_h * kernel_w;
                let output_size = out_h * out_w;
                let chunk_size = channels * kernel_size * output_size;

                col.par_chunks_mut(chunk_size).enumerate().for_each(|(b, col_chunk)| {
                    for c in 0..channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let input_offset = b * channels * height * width + c * height * width;
                                let col_offset = (c * kernel_size + kh * kernel_w + kw) * output_size;

                                for h_out in 0..out_h {
                                    let h_in = h_out as isize * stride_h as isize - pad_h as isize + kh as isize;

                                    if h_in >= 0 && h_in < height as isize {
                                        for w_out in 0..out_w {
                                            let w_in = w_out as isize * stride_w as isize - pad_w as isize + kw as isize;
                                            let col_idx = col_offset + h_out * out_w + w_out;

                                            col_chunk[col_idx] = if w_in >= 0 && w_in < width as isize {
                                                input[input_offset + h_in as usize * width + w_in as usize]
                                            } else {
                                                $zero
                                            };
                                        }
                                    } else {
                                        let col_idx = col_offset + h_out * out_w;
                                        col_chunk[col_idx..col_idx + out_w].fill($zero);
                                    }
                                }
                            }
                        }
                    }
                });
            }

            #[no_mangle]
            /// # Safety
            ///
            /// * `dims_and_strides` must be a valid pointer to array containing:
            ///   - batch_size, channels, height, width, kernel_h, kernel_w, in_h, in_w
            ///   - padding_h, padding_w, stride_h, stride_w
            /// * `col` must be a valid pointer to array containing input data
            /// * `output` must be a valid pointer to array for storing the result
            /// * Array sizes must match the dimensions provided
            pub unsafe extern "C" fn [<conv2d_col2im_ $type>](
                num_els: usize,
                dims_and_strides: *const usize,
                col: *const $type,
                output: *mut $type,
            ) {
                if dims_and_strides.is_null() || col.is_null() || output.is_null() {
                    return;
                }

                let batch_size = *dims_and_strides;
                let channels = *dims_and_strides.add(1);
                let height = *dims_and_strides.add(2);
                let width = *dims_and_strides.add(3);
                let kernel_h = *dims_and_strides.add(4);
                let kernel_w = *dims_and_strides.add(5);
                let out_h = *dims_and_strides.add(6);
                let out_w = *dims_and_strides.add(7);
                let pad_h = *dims_and_strides.add(8);
                let pad_w = *dims_and_strides.add(9);
                let stride_h = *dims_and_strides.add(10);
                let stride_w = *dims_and_strides.add(11);

                let col = std::slice::from_raw_parts(col, num_els);
                let output = std::slice::from_raw_parts_mut(output, batch_size * channels * height * width);

                output.fill($zero);

                let chunk_size = channels * height * width;
                output.par_chunks_mut(chunk_size).enumerate().for_each(|(b, out_chunk)| {
                    for c in 0..channels {
                        for h in 0..height {
                            let h_pad = h + pad_h;
                            for w in 0..width {
                                let w_pad = w + pad_w;

                                // Corrected index range calculation
                                let h_start = (h_pad).saturating_sub(kernel_h - 1).div_ceil(stride_h);
                                let w_start = (w_pad).saturating_sub(kernel_w - 1).div_ceil(stride_w);
                                let h_end = std::cmp::min(h_pad / stride_h + 1, out_h);
                                let w_end = std::cmp::min(w_pad / stride_w + 1, out_w);

                                let mut sum = $zero;
                                for h_out in h_start..h_end {
                                    for w_out in w_start..w_end {
                                        let kh = h_pad - h_out * stride_h;
                                        let kw = w_pad - w_out * stride_w;

                                        if kh < kernel_h && kw < kernel_w {
                                            let col_idx = (((b * channels + c) * kernel_h * kernel_w +
                                                         kh * kernel_w + kw) * out_h + h_out) * out_w + w_out;
                                            if col_idx < col.len() {  // Add bounds check
                                                sum += col[col_idx];
                                            }
                                        }
                                    }
                                }
                                let out_idx = (c * height + h) * width + w;
                                if out_idx < out_chunk.len() {  // Add bounds check
                                    out_chunk[out_idx] = sum;
                                }
                            }
                        }
                    }
                });
            }
        }
    };
}

// Forward pass implementations
conv2d_op!(f32, 0.0f32);
conv2d_op!(f64, 0.0f64);
conv2d_op!(f16, f16::from_f32(0.0));
conv2d_op!(bf16, bf16::from_f32(0.0));
