use half::{bf16, f16};
use rayon::prelude::*;

macro_rules! matmul_op {
    ($name:ident, $type:ty, $zero:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `dims_and_strides` must be a valid pointer to array containing:
        ///   - out_ndim, a_ndim, b_ndim
        ///   - out_shape[out_ndim]
        ///   - a_shape[a_ndim]
        ///   - b_shape[b_ndim]
        ///   - a_strides[a_ndim]
        ///   - b_strides[b_ndim]
        /// * `A`, `B`, and `C` must be valid pointers to arrays of appropriate sizes
        /// * The matrix dimensions must be compatible for multiplication
        pub unsafe fn $name(num_els: usize, dims_and_strides: *const usize, a: *const $type, b: *const $type, c: *mut $type) {
            if dims_and_strides.is_null() || a.is_null() || b.is_null() || c.is_null() {
                return;
            }

            let out_ndim = *dims_and_strides;
            let a_ndim = *dims_and_strides.add(1);
            let b_ndim = *dims_and_strides.add(2);

            let out_shape = std::slice::from_raw_parts(dims_and_strides.add(3), out_ndim);
            let a_shape = std::slice::from_raw_parts(dims_and_strides.add(3 + out_ndim), a_ndim);
            let b_shape = std::slice::from_raw_parts(dims_and_strides.add(3 + out_ndim + a_ndim), b_ndim);
            let a_strides = std::slice::from_raw_parts(dims_and_strides.add(3 + out_ndim + a_ndim + b_ndim), a_ndim);
            let b_strides = std::slice::from_raw_parts(dims_and_strides.add(3 + out_ndim + a_ndim + b_ndim + a_ndim), b_ndim);

            let m = if out_ndim > 0 { out_shape[out_ndim - 2] } else { 1 };
            let n = if out_ndim > 0 { out_shape[out_ndim - 1] } else { 1 };
            let k = a_shape[a_ndim - 1];

            let is_a_cont = is_contiguous(a_ndim, a_shape, a_strides);
            let is_b_cont = is_contiguous(b_ndim, b_shape, b_strides);

            let a_slice = std::slice::from_raw_parts(a, num_els * k / n);
            let b_slice = std::slice::from_raw_parts(b, num_els * k / m);
            let c_slice = std::slice::from_raw_parts_mut(c, num_els);

            c_slice.par_chunks_mut(n).enumerate().for_each(|(batch_m, c_row)| {
                let batch_idx = batch_m / m;
                let m_idx = batch_m % m;

                for n_idx in 0..n {
                    let mut acc = $zero;
                    if is_a_cont && is_b_cont {
                        let a_base = batch_idx * (m * k) + m_idx * k;
                        let b_base = batch_idx * (k * n);

                        for k_idx in 0..k {
                            acc += a_slice[a_base + k_idx] * b_slice[b_base + k_idx * n + n_idx];
                        }
                    } else {
                        for k_idx in 0..k {
                            let a_idx = if a_ndim > 0 {
                                batch_idx * a_strides[0] + m_idx * a_strides[a_ndim - 2] + k_idx * a_strides[a_ndim - 1]
                            } else {
                                k_idx
                            };

                            let b_idx = if b_ndim > 0 {
                                batch_idx * b_strides[0] + k_idx * b_strides[b_ndim - 2] + n_idx * b_strides[b_ndim - 1]
                            } else {
                                n_idx
                            };

                            acc += a_slice[a_idx] * b_slice[b_idx];
                        }
                    }
                    c_row[n_idx] = acc;
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
        /// * `dims_and_strides` must be a valid pointer to array containing dimension info
        /// * All input pointers must be valid and point to arrays of appropriate sizes
        /// * Either grad_a or grad_b can be null, in which case that gradient is not computed
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            num_els_a: usize,
            num_els_b: usize,
            dims_and_strides: *const usize,
            grad_output: *const $type,
            a: *const $type,
            b: *const $type,
            grad_a: *mut $type,
            grad_b: *mut $type,
        ) {
            if dims_and_strides.is_null() || grad_output.is_null() || a.is_null() || b.is_null() {
                return;
            }

            let out_ndim = *dims_and_strides;
            let b_ndim = *dims_and_strides.add(2);

            let out_shape = std::slice::from_raw_parts(dims_and_strides.add(3), out_ndim);
            let b_shape = std::slice::from_raw_parts(dims_and_strides.add(3 + out_ndim * 2), b_ndim);

            let m = if out_ndim > 0 { out_shape[out_ndim - 2] } else { 1 };
            let n = if out_ndim > 0 { out_shape[out_ndim - 1] } else { 1 };
            let k = b_shape[b_ndim - 2]; // Common dimension for backward

            let grad_output_size = if out_ndim == 0 {
                1
            } else {
                let mut size = 1;
                for d in 0..out_ndim {
                    size *= out_shape[d];
                }
                size
            };
            let grad_output = std::slice::from_raw_parts(grad_output, grad_output_size);

            // Compute grad_A if needed
            if !grad_a.is_null() {
                let grad_a_slice = std::slice::from_raw_parts_mut(grad_a, num_els_a);
                let b_slice = std::slice::from_raw_parts(b, num_els_b);

                grad_a_slice.par_chunks_mut(k).enumerate().for_each(|(batch_m, grad_row)| {
                    let batch_idx = batch_m / m;
                    let m_idx = batch_m % m;

                    for k_idx in 0..k {
                        let mut acc = $zero;
                        if out_ndim == 0 {
                            acc = grad_output[0] * b_slice[k_idx];
                        } else {
                            let grad_base = batch_idx * (m * n) + m_idx * n;
                            let b_base = batch_idx * (k * n);
                            for n_idx in 0..n {
                                acc += grad_output[grad_base + n_idx] * b_slice[b_base + k_idx * n + n_idx];
                            }
                        }
                        grad_row[k_idx] = acc;
                    }
                });
            }

            // Compute grad_B if needed
            if !grad_b.is_null() {
                let grad_b_slice = std::slice::from_raw_parts_mut(grad_b, num_els_b);
                let a_slice = std::slice::from_raw_parts(a, num_els_a);

                grad_b_slice.par_chunks_mut(n).enumerate().for_each(|(batch_k, grad_row)| {
                    let batch_idx = batch_k / k;
                    let k_idx = batch_k % k;

                    for n_idx in 0..n {
                        let mut acc = $zero;
                        if out_ndim == 0 {
                            acc = grad_output[0] * a_slice[k_idx];
                        } else {
                            let grad_base = batch_idx * (m * n);
                            let a_base = batch_idx * (m * k);
                            for m_idx in 0..m {
                                acc += a_slice[a_base + m_idx * k + k_idx] * grad_output[grad_base + m_idx * n + n_idx];
                            }
                        }
                        grad_row[n_idx] = acc;
                    }
                });
            }
        }
    };
}

fn is_contiguous(ndim: usize, shape: &[usize], strides: &[usize]) -> bool {
    let mut acc = 1;
    for d in (0..ndim).rev() {
        if strides[d] != acc {
            return false;
        }
        acc *= shape[d];
    }
    true
}

// Forward pass implementations
matmul_op!(matmul_f32, f32, 0.0f32);
matmul_op!(matmul_f64, f64, 0.0f64);
matmul_op!(matmul_u8, u8, 0u8);
matmul_op!(matmul_u32, u32, 0u32);
matmul_op!(matmul_i8, i8, 0i8);
matmul_op!(matmul_i32, i32, 0i32);
matmul_op!(matmul_i64, i64, 0i64);
matmul_op!(matmul_f16, f16, f16::from_f32(0.0));
matmul_op!(matmul_bf16, bf16, bf16::from_f32(0.0));

// Backward pass implementations
matmul_backward_op!(matmul_backward_f32, f32, 0.0f32);
matmul_backward_op!(matmul_backward_f64, f64, 0.0f64);
matmul_backward_op!(matmul_backward_u8, u8, 0u8);
matmul_backward_op!(matmul_backward_u32, u32, 0u32);
matmul_backward_op!(matmul_backward_i8, i8, 0i8);
matmul_backward_op!(matmul_backward_i32, i32, 0i32);
matmul_backward_op!(matmul_backward_i64, i64, 0i64);
matmul_backward_op!(matmul_backward_f16, f16, f16::from_f32(0.0));
matmul_backward_op!(matmul_backward_bf16, bf16, bf16::from_f32(0.0));
