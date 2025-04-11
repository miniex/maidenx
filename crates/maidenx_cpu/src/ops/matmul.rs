use crate::utils::is_contiguous;
use half::{bf16, f16};
use rayon::prelude::*;

macro_rules! matmul_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to array containing:
        ///   - out_ndim, a_ndim, b_ndim
        ///   - out_shape[out_ndim]
        ///   - a_shape[a_ndim]
        ///   - b_shape[b_ndim]
        ///   - a_strides[a_ndim]
        ///   - b_strides[b_ndim]
        /// * `A`, `B`, and `C` must be valid pointers to arrays of appropriate sizes
        /// * The matrix dimensions must be compatible for multiplication
        pub unsafe fn $name(num_els: usize, metadata: *const usize, a: *const $type, b: *const $type, c: *mut $type) {
            if metadata.is_null() || a.is_null() || b.is_null() || c.is_null() {
                return;
            }

            let out_ndim = *metadata;
            let a_ndim = *metadata.add(1);
            let b_ndim = *metadata.add(2);

            if a_ndim == 0 || b_ndim == 0 {
                return;
            }

            let out_shape = std::slice::from_raw_parts(metadata.add(3), out_ndim);
            let a_shape = std::slice::from_raw_parts(metadata.add(3 + out_ndim), a_ndim);
            let b_shape = std::slice::from_raw_parts(metadata.add(3 + out_ndim + a_ndim), b_ndim);

            let a_strides = std::slice::from_raw_parts(metadata.add(3 + out_ndim + a_ndim + b_ndim), a_ndim);
            let b_strides = std::slice::from_raw_parts(metadata.add(3 + out_ndim + a_ndim + b_ndim + a_ndim), b_ndim);

            let a_offset = *metadata.add(3 + out_ndim + a_ndim + b_ndim + a_ndim + b_ndim);
            let b_offset = *metadata.add(3 + out_ndim + a_ndim + b_ndim + a_ndim + b_ndim + 1);

            let m = if out_ndim >= 2 { out_shape[out_ndim - 2] } else { 1 };
            let n = if out_ndim >= 1 { out_shape[out_ndim - 1] } else { 1 };
            let k = if a_ndim >= 1 { a_shape[a_ndim - 1] } else { 1 };

            if m == 0 || n == 0 || k == 0 {
                return;
            }

            let is_a_cont = is_contiguous(a_ndim, a_shape, a_strides);
            let is_b_cont = is_contiguous(b_ndim, b_shape, b_strides);

            let a_size = num_els * k;
            let b_size = num_els * k;
            if n > 0 {
                let _ = a_size / n;
            }
            if m > 0 {
                let _ = b_size / m;
            }

            let a_slice = std::slice::from_raw_parts(a.add(a_offset), a_size);
            let b_slice = std::slice::from_raw_parts(b.add(b_offset), b_size);
            let c_slice = std::slice::from_raw_parts_mut(c, num_els);

            c_slice
                .par_chunks_mut(n.max(1))
                .enumerate()
                .for_each(|(batch_m, c_row)| {
                    let batch_idx = if m > 0 { batch_m / m } else { 0 };
                    let m_idx = if m > 0 { batch_m % m } else { 0 };

                    for n_idx in 0..n {
                        let mut acc = $zero;
                        if is_a_cont && is_b_cont {
                            let a_base = batch_idx * (m * k) + m_idx * k;
                            let b_base = batch_idx * (k * n);

                            for k_idx in 0..k {
                                let a_index = a_base + k_idx;
                                let b_index = b_base + k_idx * n + n_idx;

                                if a_index < a_size && b_index < b_size {
                                    acc += a_slice[a_index] * b_slice[b_index];
                                }
                            }
                        } else {
                            for k_idx in 0..k {
                                let a_idx = if a_ndim >= 2 {
                                    batch_idx * a_strides[0]
                                        + m_idx * a_strides[a_ndim - 2]
                                        + k_idx * a_strides[a_ndim - 1]
                                } else if a_ndim == 1 {
                                    k_idx * a_strides[0]
                                } else {
                                    0
                                };

                                let b_idx = if b_ndim >= 2 {
                                    batch_idx * b_strides[0]
                                        + k_idx * b_strides[b_ndim - 2]
                                        + n_idx * b_strides[b_ndim - 1]
                                } else if b_ndim == 1 {
                                    n_idx * b_strides[0]
                                } else {
                                    0
                                };

                                if a_idx < a_size && b_idx < b_size {
                                    acc += a_slice[a_idx] * b_slice[b_idx];
                                }
                            }
                        }

                        if n_idx < c_row.len() {
                            c_row[n_idx] = acc;
                        }
                    }
                });
        }
    };
}

macro_rules! matmul_backward_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be a valid pointer to array containing dimension info
        /// * All input pointers must be valid and point to arrays of appropriate sizes
        /// * Either grad_a or grad_b can be null, in which case that gradient is not computed
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            num_els_a: usize,
            num_els_b: usize,
            metadata: *const usize,
            grad_output: *const $type,
            a: *const $type,
            b: *const $type,
            grad_a: *mut $type,
            grad_b: *mut $type,
        ) {
            if metadata.is_null() || grad_output.is_null() || a.is_null() || b.is_null() {
                return;
            }

            let out_ndim = *metadata;
            let a_ndim = *metadata.add(1);
            let b_ndim = *metadata.add(2);

            if a_ndim == 0 || b_ndim == 0 {
                return;
            }

            let out_shape = std::slice::from_raw_parts(metadata.add(3), out_ndim);
            let a_shape = std::slice::from_raw_parts(metadata.add(3 + out_ndim), a_ndim);
            let b_shape = std::slice::from_raw_parts(metadata.add(3 + out_ndim + a_ndim), b_ndim);

            let a_offset = *metadata.add(3 + out_ndim + a_ndim + b_ndim + a_ndim + b_ndim);
            let b_offset = *metadata.add(3 + out_ndim + a_ndim + b_ndim + a_ndim + b_ndim + 1);

            let m = if out_ndim >= 2 { out_shape[out_ndim - 2] } else { 1 };
            let n = if out_ndim >= 1 { out_shape[out_ndim - 1] } else { 1 };
            let k = if b_ndim >= 2 {
                b_shape[b_ndim - 2]
            } else if a_ndim >= 1 {
                a_shape[a_ndim - 1]
            } else {
                1
            };

            if m == 0 || n == 0 || k == 0 {
                return;
            }

            let grad_output_size = if out_ndim == 0 {
                1
            } else {
                let mut size = 1;
                for d in 0..out_ndim {
                    size *= out_shape[d];
                }
                size
            };

            if grad_output_size == 0 {
                return;
            }

            let grad_output = std::slice::from_raw_parts(grad_output, grad_output_size);

            // Compute grad_A if needed
            if !grad_a.is_null() {
                let grad_a_slice = std::slice::from_raw_parts_mut(grad_a, num_els_a);
                let b_slice = std::slice::from_raw_parts(b.add(b_offset), num_els_b);

                grad_a_slice
                    .par_chunks_mut(k.max(1))
                    .enumerate()
                    .for_each(|(batch_m, grad_row)| {
                        let batch_idx = if m > 0 { batch_m / m } else { 0 };
                        let m_idx = if m > 0 { batch_m % m } else { 0 };

                        for k_idx in 0..k {
                            let mut acc = $zero;
                            if out_ndim == 0 {
                                if k_idx < num_els_b {
                                    acc = grad_output[0] * b_slice[k_idx];
                                }
                            } else {
                                let grad_base = batch_idx * (m * n) + m_idx * n;
                                let b_base = batch_idx * (k * n);

                                for n_idx in 0..n {
                                    if grad_base + n_idx < grad_output_size && b_base + k_idx * n + n_idx < num_els_b {
                                        acc += grad_output[grad_base + n_idx] * b_slice[b_base + k_idx * n + n_idx];
                                    }
                                }
                            }

                            if k_idx < grad_row.len() {
                                grad_row[k_idx] = acc;
                            }
                        }
                    });
            }

            // Compute grad_B if needed
            if !grad_b.is_null() {
                let grad_b_slice = std::slice::from_raw_parts_mut(grad_b, num_els_b);
                let a_slice = std::slice::from_raw_parts(a.add(a_offset), num_els_a);

                grad_b_slice
                    .par_chunks_mut(n.max(1))
                    .enumerate()
                    .for_each(|(batch_k, grad_row)| {
                        let batch_idx = if k > 0 { batch_k / k } else { 0 };
                        let k_idx = if k > 0 { batch_k % k } else { 0 };

                        for n_idx in 0..n {
                            let mut acc = $zero;
                            if out_ndim == 0 {
                                if k_idx < num_els_a {
                                    acc = grad_output[0] * a_slice[k_idx];
                                }
                            } else {
                                let a_base = batch_idx * (m * k);
                                let grad_base = batch_idx * (m * n);

                                for m_idx in 0..m {
                                    let a_idx = a_base + m_idx * k + k_idx;
                                    let grad_idx = grad_base + m_idx * n + n_idx;

                                    if a_idx < num_els_a && grad_idx < grad_output_size {
                                        acc += a_slice[a_idx] * grad_output[grad_idx];
                                    }
                                }
                            }

                            if n_idx < grad_row.len() {
                                grad_row[n_idx] = acc;
                            }
                        }
                    });
            }
        }
    };
}

// Forward pass implementations
matmul_op!(matmul_f32, f32, 0.0f32);
matmul_op!(matmul_f64, f64, 0.0f64);
matmul_op!(matmul_u8, u8, 0u8);
matmul_op!(matmul_u16, u16, 0u16);
matmul_op!(matmul_u32, u32, 0u32);
matmul_op!(matmul_u64, u64, 0u64);
matmul_op!(matmul_i8, i8, 0i8);
matmul_op!(matmul_i16, i16, 0i16);
matmul_op!(matmul_i32, i32, 0i32);
matmul_op!(matmul_i64, i64, 0i64);
matmul_op!(matmul_f16, f16, f16::from_f32(0.0));
matmul_op!(matmul_bf16, bf16, bf16::from_f32(0.0));

// Backward pass implementations
matmul_backward_op!(matmul_backward_f32, f32, 0.0f32);
matmul_backward_op!(matmul_backward_f64, f64, 0.0f64);
matmul_backward_op!(matmul_backward_u8, u8, 0u8);
matmul_backward_op!(matmul_backward_u16, u16, 0u16);
matmul_backward_op!(matmul_backward_u32, u32, 0u32);
matmul_backward_op!(matmul_backward_u64, u64, 0u64);
matmul_backward_op!(matmul_backward_i8, i8, 0i8);
matmul_backward_op!(matmul_backward_i16, i16, 0i16);
matmul_backward_op!(matmul_backward_i32, i32, 0i32);
matmul_backward_op!(matmul_backward_i64, i64, 0i64);
matmul_backward_op!(matmul_backward_f16, f16, f16::from_f32(0.0));
matmul_backward_op!(matmul_backward_bf16, bf16, bf16::from_f32(0.0));
