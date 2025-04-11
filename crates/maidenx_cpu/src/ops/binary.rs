#![allow(clippy::comparison_chain)]

use crate::utils::is_contiguous;
use half::{bf16, f16};
use rayon::prelude::*;

macro_rules! binary_op {
    ($name:ident, $op:expr, $type:ty) => {
        binary_op_with_output!($name, $op, $type, $type);
    };
}

macro_rules! binary_op_with_output {
    ($name:ident, $op:expr, $input_type:ty, $output_type:ty) => {
        #[no_mangle]
        /// # Safety
        ///
        /// Caller must guarantee that:
        /// * `metadata` must be either:
        ///   - null (indicating contiguous arrays) or
        ///   - a valid pointer to an array of `3 * num_dims` elements containing:
        ///     - dims[num_dims]: array dimensions
        ///     - lhs_strides[num_dims]: strides for left-hand side array
        ///     - rhs_strides[num_dims]: strides for right-hand side array
        /// * `lhs` must be a valid pointer to an array of at least `num_els` elements
        /// * `rhs` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of at least `num_els` elements
        /// * The memory regions of `lhs`, `rhs`, and `out` must not overlap
        /// * The alignment requirements of the data type must be respected for all arrays
        /// * All array indices calculated from dims and strides must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            lhs: *const $input_type,
            rhs: *const $input_type,
            out: *mut $output_type,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let lhs_strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let rhs_strides = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims);
            let lhs_offset = if metadata.is_null() {
                0
            } else {
                *metadata.add(3 * num_dims)
            };
            let rhs_offset = if metadata.is_null() {
                0
            } else {
                *metadata.add(3 * num_dims + 1)
            };

            let lhs_cont = is_contiguous(num_dims, dims, lhs_strides);
            let rhs_cont = is_contiguous(num_dims, dims, rhs_strides);

            let lhs = std::slice::from_raw_parts(lhs.add(lhs_offset), num_els);
            let rhs = std::slice::from_raw_parts(rhs.add(rhs_offset), num_els);
            let out = std::slice::from_raw_parts_mut(out, num_els);

            if lhs_cont && rhs_cont {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    *out_val = $op(lhs[i], rhs[i]) as $output_type;
                });
            } else if lhs_cont {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let mut tmp_i = i;
                    let mut rhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        rhs_i += i_dim * rhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    *out_val = $op(lhs[i], rhs[rhs_i]) as $output_type;
                });
            } else if rhs_cont {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let mut tmp_i = i;
                    let mut lhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        lhs_i += i_dim * lhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    *out_val = $op(lhs[lhs_i], rhs[i]) as $output_type;
                });
            } else {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let mut tmp_i = i;
                    let mut lhs_i = 0;
                    let mut rhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        lhs_i += i_dim * lhs_strides[d];
                        rhs_i += i_dim * rhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    *out_val = $op(lhs[lhs_i], rhs[rhs_i]) as $output_type;
                });
            }
        }
    };
}

macro_rules! binary_op_inplace {
    ($name:ident, $op:expr, $type:ty) => {
        #[no_mangle]
        /// # Safety
        ///
        /// Caller must guarantee that:
        /// * `metadata` must be either:
        ///   - null (indicating contiguous arrays) or
        ///   - a valid pointer to an array of `3 * num_dims` elements containing:
        ///     - dims[num_dims]: array dimensions
        ///     - lhs_strides[num_dims]: strides for left-hand side array
        ///     - rhs_strides[num_dims]: strides for right-hand side array
        /// * `lhs` must be a valid pointer to an array of at least `num_els` elements
        /// * `rhs` must be a valid pointer to an array of at least `num_els` elements
        /// * The memory regions of `rhs` and `lhs` may overlap only if they are identical
        /// * The alignment requirements of the data type must be respected for all arrays
        /// * All array indices calculated from dims and strides must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            lhs: *mut $type,
            rhs: *const $type,
        ) {
            // Copy metadata to local variables to avoid thread safety issues
            let (dims, lhs_strides, rhs_strides, lhs_offset, rhs_offset) = if !metadata.is_null() {
                let dims_slice = std::slice::from_raw_parts(metadata, num_dims);
                let lhs_strides_slice = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
                let rhs_strides_slice = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims);

                // Clone these slices to owned Vec to make them thread-safe
                let dims_vec = dims_slice.to_vec();
                let lhs_strides_vec = lhs_strides_slice.to_vec();
                let rhs_strides_vec = rhs_strides_slice.to_vec();

                let lhs_offset = *metadata.add(3 * num_dims);
                let rhs_offset = *metadata.add(3 * num_dims + 1);

                (dims_vec, lhs_strides_vec, rhs_strides_vec, lhs_offset, rhs_offset)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), 0, 0)
            };

            let lhs_cont = is_contiguous(num_dims, &dims, &lhs_strides);
            let rhs_cont = is_contiguous(num_dims, &dims, &rhs_strides);

            if lhs_cont && rhs_cont {
                // Both arrays are contiguous, process directly
                for i in 0..num_els {
                    let lhs_val = &mut *lhs.add(lhs_offset + i);
                    let rhs_val = &*rhs.add(rhs_offset + i);
                    *lhs_val = $op(*lhs_val, *rhs_val);
                }
            } else if lhs_cont {
                // LHS is contiguous, RHS is not
                for i in 0..num_els {
                    let mut tmp_i = i;
                    let mut rhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        rhs_i += i_dim * rhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    let lhs_val = &mut *lhs.add(lhs_offset + i);
                    let rhs_val = &*rhs.add(rhs_offset + rhs_i);
                    *lhs_val = $op(*lhs_val, *rhs_val);
                }
            } else if rhs_cont {
                // RHS is contiguous, LHS is not
                for i in 0..num_els {
                    let mut tmp_i = i;
                    let mut lhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        lhs_i += i_dim * lhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    let lhs_val = &mut *lhs.add(lhs_offset + lhs_i);
                    let rhs_val = &*rhs.add(rhs_offset + i);
                    *lhs_val = $op(*lhs_val, *rhs_val);
                }
            } else {
                // Neither array is contiguous
                for i in 0..num_els {
                    let mut tmp_i = i;
                    let mut lhs_i = 0;
                    let mut rhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        lhs_i += i_dim * lhs_strides[d];
                        rhs_i += i_dim * rhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    let lhs_val = &mut *lhs.add(lhs_offset + lhs_i);
                    let rhs_val = &*rhs.add(rhs_offset + rhs_i);
                    *lhs_val = $op(*lhs_val, *rhs_val);
                }
            }
        }
    };
}

macro_rules! logical_op {
    ($name:ident, $t:ty, $zero:expr, $op:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// Caller must guarantee that:
        /// * `metadata` must be either:
        ///   - null (indicating contiguous arrays) or
        ///   - a valid pointer to an array of `3 * num_dims` elements containing:
        ///     - dims[num_dims]: array dimensions
        ///     - lhs_strides[num_dims]: strides for left-hand side array
        ///     - rhs_strides[num_dims]: strides for right-hand side array
        /// * `lhs` must be a valid pointer to an array of at least `num_els` elements
        /// * `rhs` must be a valid pointer to an array of at least `num_els` elements
        /// * `out` must be a valid pointer to an array of at least `num_els` elements
        /// * The memory regions of `lhs`, `rhs`, and `out` must not overlap
        /// * The alignment requirements of the data type must be respected for all arrays
        /// * All array indices calculated from dims and strides must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            lhs: *const $t,
            rhs: *const $t,
            out: *mut bool,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let lhs_strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let rhs_strides = std::slice::from_raw_parts(metadata.add(2 * num_dims), num_dims);
            let lhs_offset = if metadata.is_null() {
                0
            } else {
                *metadata.add(3 * num_dims)
            };
            let rhs_offset = if metadata.is_null() {
                0
            } else {
                *metadata.add(3 * num_dims + 1)
            };

            let lhs_cont = is_contiguous(num_dims, dims, lhs_strides);
            let rhs_cont = is_contiguous(num_dims, dims, rhs_strides);

            let lhs = std::slice::from_raw_parts(lhs.add(lhs_offset), num_els);
            let rhs = std::slice::from_raw_parts(rhs.add(rhs_offset), num_els);
            let out = std::slice::from_raw_parts_mut(out, num_els);

            if lhs_cont && rhs_cont {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let lhs_bool = lhs[i] != $zero;
                    let rhs_bool = rhs[i] != $zero;
                    *out_val = $op(lhs_bool, rhs_bool);
                });
            } else if lhs_cont {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let mut tmp_i = i;
                    let mut rhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        rhs_i += i_dim * rhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    let lhs_bool = lhs[i] != $zero;
                    let rhs_bool = rhs[rhs_i] != $zero;
                    *out_val = $op(lhs_bool, rhs_bool);
                });
            } else if rhs_cont {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let mut tmp_i = i;
                    let mut lhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        lhs_i += i_dim * lhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    let lhs_bool = lhs[lhs_i] != $zero;
                    let rhs_bool = rhs[i] != $zero;
                    *out_val = $op(lhs_bool, rhs_bool);
                });
            } else {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let mut tmp_i = i;
                    let mut lhs_i = 0;
                    let mut rhs_i = 0;

                    for d in (0..num_dims).rev() {
                        let i_dim = tmp_i % dims[d];
                        lhs_i += i_dim * lhs_strides[d];
                        rhs_i += i_dim * rhs_strides[d];
                        tmp_i /= dims[d];
                    }

                    let lhs_bool = lhs[lhs_i] != $zero;
                    let rhs_bool = rhs[rhs_i] != $zero;
                    *out_val = $op(lhs_bool, rhs_bool);
                });
            }
        }
    };
}

// Arithmetic operations
binary_op!(add_bf16, |a, b| a + b, bf16);
binary_op!(add_f16, |a, b| a + b, f16);
binary_op!(add_f32, |a, b| a + b, f32);
binary_op!(add_f64, |a, b| a + b, f64);
binary_op!(add_bool, |a, b| a | b, bool);
binary_op!(add_u8, |a: u8, b: u8| a.saturating_add(b), u8);
binary_op!(add_u16, |a: u16, b: u16| a.saturating_add(b), u16);
binary_op!(add_u32, |a: u32, b: u32| a.saturating_add(b), u32);
binary_op!(add_u64, |a: u64, b: u64| a.saturating_add(b), u64);
binary_op!(add_i8, |a: i8, b: i8| a.saturating_add(b), i8);
binary_op!(add_i16, |a: i16, b: i16| a.saturating_add(b), i16);
binary_op!(add_i32, |a: i32, b: i32| a.saturating_add(b), i32);
binary_op!(add_i64, |a: i64, b: i64| a.saturating_add(b), i64);

binary_op!(sub_bf16, |a, b| a - b, bf16);
binary_op!(sub_f16, |a, b| a - b, f16);
binary_op!(sub_f32, |a, b| a - b, f32);
binary_op!(sub_f64, |a, b| a - b, f64);
binary_op!(sub_bool, |a, b| a & b, bool);
binary_op!(sub_u8, |a: u8, b: u8| a.saturating_sub(b), u8);
binary_op!(sub_u16, |a: u16, b: u16| a.saturating_sub(b), u16);
binary_op!(sub_u32, |a: u32, b: u32| a.saturating_sub(b), u32);
binary_op!(sub_u64, |a: u64, b: u64| a.saturating_sub(b), u64);
binary_op!(sub_i8, |a: i8, b: i8| a.saturating_sub(b), i8);
binary_op!(sub_i16, |a: i16, b: i16| a.saturating_sub(b), i16);
binary_op!(sub_i32, |a: i32, b: i32| a.saturating_sub(b), i32);
binary_op!(sub_i64, |a: i64, b: i64| a.saturating_sub(b), i64);

binary_op!(mul_bf16, |a, b| a * b, bf16);
binary_op!(mul_f16, |a, b| a * b, f16);
binary_op!(mul_f32, |a, b| a * b, f32);
binary_op!(mul_f64, |a, b| a * b, f64);
binary_op!(mul_bool, |a, b| a ^ b, bool);
binary_op!(mul_u8, |a: u8, b: u8| a.saturating_mul(b), u8);
binary_op!(mul_u16, |a: u16, b: u16| a.saturating_mul(b), u16);
binary_op!(mul_u32, |a: u32, b: u32| a.saturating_mul(b), u32);
binary_op!(mul_u64, |a: u64, b: u64| a.saturating_mul(b), u64);
binary_op!(mul_i8, |a: i8, b: i8| a.saturating_mul(b), i8);
binary_op!(mul_i16, |a: i16, b: i16| a.saturating_mul(b), i16);
binary_op!(mul_i32, |a: i32, b: i32| a.saturating_mul(b), i32);
binary_op!(mul_i64, |a: i64, b: i64| a.saturating_mul(b), i64);

binary_op!(
    div_bf16,
    |a: bf16, b: bf16| if b == bf16::from_f32(0.0) {
        if a == bf16::from_f32(0.0) {
            bf16::from_f32(0.0)
        } else {
            bf16::from_f32(f32::INFINITY)
        }
    } else {
        a / b
    },
    bf16
);
binary_op!(
    div_f16,
    |a: f16, b: f16| if b == f16::from_f32(0.0) {
        if a == f16::from_f32(0.0) {
            f16::from_f32(0.0)
        } else {
            f16::from_f32(f32::INFINITY)
        }
    } else {
        a / b
    },
    f16
);
binary_op!(
    div_f32,
    |a: f32, b: f32| if b == 0.0 {
        if a == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        a / b
    },
    f32
);
binary_op!(
    div_f64,
    |a: f64, b: f64| if b == 0.0 {
        if a == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        a / b
    },
    f64
);
binary_op!(div_bool, |a: bool, b: bool| a & b, bool);
binary_op!(
    div_u8,
    |a: u8, b: u8| if b < 1 {
        if a == 0 {
            0
        } else {
            u8::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u8
);
binary_op!(
    div_u16,
    |a: u16, b: u16| if b < 1 {
        if a == 0 {
            0
        } else {
            u16::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u16
);
binary_op!(
    div_u32,
    |a: u32, b: u32| if b < 1 {
        if a == 0 {
            0
        } else {
            u32::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u32
);
binary_op!(
    div_u64,
    |a: u64, b: u64| if b < 1 {
        if a == 0 {
            0
        } else {
            u64::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u64
);
binary_op!(
    div_i8,
    |a: i8, b: i8| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i8::MIN
        } else {
            i8::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i8
);
binary_op!(
    div_i16,
    |a: i16, b: i16| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i16::MIN
        } else {
            i16::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i16
);
binary_op!(
    div_i32,
    |a: i32, b: i32| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i32
);
binary_op!(
    div_i64,
    |a: i64, b: i64| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i64::MIN
        } else {
            i64::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i64
);

binary_op!(maximum_bf16, |a, b| if a > b { a } else { b }, bf16);
binary_op!(maximum_f16, |a, b| if a > b { a } else { b }, f16);
binary_op!(maximum_f32, |a: f32, b| a.max(b), f32);
binary_op!(maximum_f64, |a: f64, b| a.max(b), f64);
binary_op!(maximum_bool, |a, b| a | b, bool);
binary_op!(maximum_u8, |a: u8, b: u8| a.max(b), u8);
binary_op!(maximum_u16, |a: u16, b: u16| a.max(b), u16);
binary_op!(maximum_u32, |a: u32, b: u32| a.max(b), u32);
binary_op!(maximum_u64, |a: u64, b: u64| a.max(b), u64);
binary_op!(maximum_i8, |a: i8, b: i8| a.max(b), i8);
binary_op!(maximum_i16, |a: i16, b: i16| a.max(b), i16);
binary_op!(maximum_i32, |a: i32, b: i32| a.max(b), i32);
binary_op!(maximum_i64, |a: i64, b: i64| a.max(b), i64);

binary_op!(minimum_bf16, |a, b| if a < b { a } else { b }, bf16);
binary_op!(minimum_f16, |a, b| if a < b { a } else { b }, f16);
binary_op!(minimum_f32, |a: f32, b| a.min(b), f32);
binary_op!(minimum_f64, |a: f64, b| a.min(b), f64);
binary_op!(minimum_bool, |a, b| a & b, bool);
binary_op!(minimum_u8, |a: u8, b: u8| a.min(b), u8);
binary_op!(minimum_u16, |a: u16, b: u16| a.min(b), u16);
binary_op!(minimum_u32, |a: u32, b: u32| a.min(b), u32);
binary_op!(minimum_u64, |a: u64, b: u64| a.min(b), u64);
binary_op!(minimum_i8, |a: i8, b: i8| a.min(b), i8);
binary_op!(minimum_i16, |a: i16, b: i16| a.min(b), i16);
binary_op!(minimum_i32, |a: i32, b: i32| a.min(b), i32);
binary_op!(minimum_i64, |a: i64, b: i64| a.min(b), i64);

// Comparison operations
logical_op!(logical_and_bf16, bf16, bf16::from_f32(0.0), |a, b| a && b);
logical_op!(logical_or_bf16, bf16, bf16::from_f32(0.0), |a, b| a || b);
logical_op!(logical_xor_bf16, bf16, bf16::from_f32(0.0), |a, b| a != b);
logical_op!(logical_and_f16, f16, f16::from_f32(0.0), |a, b| a && b);
logical_op!(logical_or_f16, f16, f16::from_f32(0.0), |a, b| a || b);
logical_op!(logical_xor_f16, f16, f16::from_f32(0.0), |a, b| a != b);
logical_op!(logical_and_f32, f32, 0.0, |a, b| a && b);
logical_op!(logical_or_f32, f32, 0.0, |a, b| a || b);
logical_op!(logical_xor_f32, f32, 0.0, |a, b| a != b);
logical_op!(logical_and_f64, f64, 0.0, |a, b| a && b);
logical_op!(logical_or_f64, f64, 0.0, |a, b| a || b);
logical_op!(logical_xor_f64, f64, 0.0, |a, b| a != b);
logical_op!(logical_and_bool, bool, false, |a, b| a && b);
logical_op!(logical_or_bool, bool, false, |a, b| a || b);
logical_op!(logical_xor_bool, bool, false, |a, b| a != b);
logical_op!(logical_and_u8, u8, 0, |a, b| a && b);
logical_op!(logical_or_u8, u8, 0, |a, b| a || b);
logical_op!(logical_xor_u8, u8, 0, |a, b| a != b);
logical_op!(logical_and_u16, u16, 0, |a, b| a && b);
logical_op!(logical_or_u16, u16, 0, |a, b| a || b);
logical_op!(logical_xor_u16, u16, 0, |a, b| a != b);
logical_op!(logical_and_u32, u32, 0, |a, b| a && b);
logical_op!(logical_or_u32, u32, 0, |a, b| a || b);
logical_op!(logical_xor_u32, u32, 0, |a, b| a != b);
logical_op!(logical_and_u64, u64, 0, |a, b| a && b);
logical_op!(logical_or_u64, u64, 0, |a, b| a || b);
logical_op!(logical_xor_u64, u64, 0, |a, b| a != b);
logical_op!(logical_and_i8, i8, 0, |a, b| a && b);
logical_op!(logical_or_i8, i8, 0, |a, b| a || b);
logical_op!(logical_xor_i8, i8, 0, |a, b| a != b);
logical_op!(logical_and_i16, i16, 0, |a, b| a && b);
logical_op!(logical_or_i16, i16, 0, |a, b| a || b);
logical_op!(logical_xor_i16, i16, 0, |a, b| a != b);
logical_op!(logical_and_i32, i32, 0, |a, b| a && b);
logical_op!(logical_or_i32, i32, 0, |a, b| a || b);
logical_op!(logical_xor_i32, i32, 0, |a, b| a != b);
logical_op!(logical_and_i64, i64, 0, |a, b| a && b);
logical_op!(logical_or_i64, i64, 0, |a, b| a || b);
logical_op!(logical_xor_i64, i64, 0, |a, b| a != b);

binary_op_with_output!(eq_bf16, |a, b| a == b, bf16, bool);
binary_op_with_output!(ne_bf16, |a, b| a != b, bf16, bool);
binary_op_with_output!(lt_bf16, |a, b| a < b, bf16, bool);
binary_op_with_output!(le_bf16, |a, b| a <= b, bf16, bool);
binary_op_with_output!(gt_bf16, |a, b| a > b, bf16, bool);
binary_op_with_output!(ge_bf16, |a, b| a >= b, bf16, bool);
binary_op_with_output!(eq_f16, |a, b| a == b, f16, bool);
binary_op_with_output!(ne_f16, |a, b| a != b, f16, bool);
binary_op_with_output!(lt_f16, |a, b| a < b, f16, bool);
binary_op_with_output!(le_f16, |a, b| a <= b, f16, bool);
binary_op_with_output!(gt_f16, |a, b| a > b, f16, bool);
binary_op_with_output!(ge_f16, |a, b| a >= b, f16, bool);
binary_op_with_output!(eq_f32, |a, b| a == b, f32, bool);
binary_op_with_output!(ne_f32, |a, b| a != b, f32, bool);
binary_op_with_output!(lt_f32, |a, b| a < b, f32, bool);
binary_op_with_output!(le_f32, |a, b| a <= b, f32, bool);
binary_op_with_output!(gt_f32, |a, b| a > b, f32, bool);
binary_op_with_output!(ge_f32, |a, b| a >= b, f32, bool);
binary_op_with_output!(eq_f64, |a, b| a == b, f64, bool);
binary_op_with_output!(ne_f64, |a, b| a != b, f64, bool);
binary_op_with_output!(lt_f64, |a, b| a < b, f64, bool);
binary_op_with_output!(le_f64, |a, b| a <= b, f64, bool);
binary_op_with_output!(gt_f64, |a, b| a > b, f64, bool);
binary_op_with_output!(ge_f64, |a, b| a >= b, f64, bool);
binary_op_with_output!(eq_bool, |a, b| a == b, bool, bool);
binary_op_with_output!(ne_bool, |a, b| a != b, bool, bool);
binary_op_with_output!(lt_bool, |a: bool, b: bool| !a & b, bool, bool);
binary_op_with_output!(le_bool, |a, b| a <= b, bool, bool);
binary_op_with_output!(gt_bool, |a: bool, b: bool| a & !b, bool, bool);
binary_op_with_output!(ge_bool, |a, b| a >= b, bool, bool);
binary_op_with_output!(eq_u8, |a, b| a == b, u8, bool);
binary_op_with_output!(ne_u8, |a, b| a != b, u8, bool);
binary_op_with_output!(lt_u8, |a, b| a < b, u8, bool);
binary_op_with_output!(le_u8, |a, b| a <= b, u8, bool);
binary_op_with_output!(gt_u8, |a, b| a > b, u8, bool);
binary_op_with_output!(ge_u8, |a, b| a >= b, u8, bool);
binary_op_with_output!(eq_u16, |a, b| a == b, u16, bool);
binary_op_with_output!(ne_u16, |a, b| a != b, u16, bool);
binary_op_with_output!(lt_u16, |a, b| a < b, u16, bool);
binary_op_with_output!(le_u16, |a, b| a <= b, u16, bool);
binary_op_with_output!(gt_u16, |a, b| a > b, u16, bool);
binary_op_with_output!(ge_u16, |a, b| a >= b, u16, bool);
binary_op_with_output!(eq_u32, |a, b| a == b, u32, bool);
binary_op_with_output!(ne_u32, |a, b| a != b, u32, bool);
binary_op_with_output!(lt_u32, |a, b| a < b, u32, bool);
binary_op_with_output!(le_u32, |a, b| a <= b, u32, bool);
binary_op_with_output!(gt_u32, |a, b| a > b, u32, bool);
binary_op_with_output!(ge_u32, |a, b| a >= b, u32, bool);
binary_op_with_output!(eq_u64, |a, b| a == b, u64, bool);
binary_op_with_output!(ne_u64, |a, b| a != b, u64, bool);
binary_op_with_output!(lt_u64, |a, b| a < b, u64, bool);
binary_op_with_output!(le_u64, |a, b| a <= b, u64, bool);
binary_op_with_output!(gt_u64, |a, b| a > b, u64, bool);
binary_op_with_output!(ge_u64, |a, b| a >= b, u64, bool);
binary_op_with_output!(eq_i8, |a, b| a == b, i8, bool);
binary_op_with_output!(ne_i8, |a, b| a != b, i8, bool);
binary_op_with_output!(lt_i8, |a, b| a < b, i8, bool);
binary_op_with_output!(le_i8, |a, b| a <= b, i8, bool);
binary_op_with_output!(gt_i8, |a, b| a > b, i8, bool);
binary_op_with_output!(ge_i8, |a, b| a >= b, i8, bool);
binary_op_with_output!(eq_i16, |a, b| a == b, i16, bool);
binary_op_with_output!(ne_i16, |a, b| a != b, i16, bool);
binary_op_with_output!(lt_i16, |a, b| a < b, i16, bool);
binary_op_with_output!(le_i16, |a, b| a <= b, i16, bool);
binary_op_with_output!(gt_i16, |a, b| a > b, i16, bool);
binary_op_with_output!(ge_i16, |a, b| a >= b, i16, bool);
binary_op_with_output!(eq_i32, |a, b| a == b, i32, bool);
binary_op_with_output!(ne_i32, |a, b| a != b, i32, bool);
binary_op_with_output!(lt_i32, |a, b| a < b, i32, bool);
binary_op_with_output!(le_i32, |a, b| a <= b, i32, bool);
binary_op_with_output!(gt_i32, |a, b| a > b, i32, bool);
binary_op_with_output!(ge_i32, |a, b| a >= b, i32, bool);
binary_op_with_output!(eq_i64, |a, b| a == b, i64, bool);
binary_op_with_output!(ne_i64, |a, b| a != b, i64, bool);
binary_op_with_output!(lt_i64, |a, b| a < b, i64, bool);
binary_op_with_output!(le_i64, |a, b| a <= b, i64, bool);
binary_op_with_output!(gt_i64, |a, b| a > b, i64, bool);
binary_op_with_output!(ge_i64, |a, b| a >= b, i64, bool);

binary_op_inplace!(add_inplace_bf16, |a, b| a + b, bf16);
binary_op_inplace!(add_inplace_f16, |a, b| a + b, f16);
binary_op_inplace!(add_inplace_f32, |a, b| a + b, f32);
binary_op_inplace!(add_inplace_f64, |a, b| a + b, f64);
binary_op_inplace!(add_inplace_bool, |a, b| a | b, bool);
binary_op_inplace!(add_inplace_u8, |a: u8, b: u8| a.saturating_add(b), u8);
binary_op_inplace!(add_inplace_u16, |a: u16, b: u16| a.saturating_add(b), u16);
binary_op_inplace!(add_inplace_u32, |a: u32, b: u32| a.saturating_add(b), u32);
binary_op_inplace!(add_inplace_u64, |a: u64, b: u64| a.saturating_add(b), u64);
binary_op_inplace!(add_inplace_i8, |a: i8, b: i8| a.saturating_add(b), i8);
binary_op_inplace!(add_inplace_i16, |a: i16, b: i16| a.saturating_add(b), i16);
binary_op_inplace!(add_inplace_i32, |a: i32, b: i32| a.saturating_add(b), i32);
binary_op_inplace!(add_inplace_i64, |a: i64, b: i64| a.saturating_add(b), i64);

binary_op_inplace!(sub_inplace_bf16, |a, b| a - b, bf16);
binary_op_inplace!(sub_inplace_f16, |a, b| a - b, f16);
binary_op_inplace!(sub_inplace_f32, |a, b| a - b, f32);
binary_op_inplace!(sub_inplace_f64, |a, b| a - b, f64);
binary_op_inplace!(sub_inplace_bool, |a, b| a & b, bool);
binary_op_inplace!(sub_inplace_u8, |a: u8, b: u8| a.saturating_sub(b), u8);
binary_op_inplace!(sub_inplace_u16, |a: u16, b: u16| a.saturating_sub(b), u16);
binary_op_inplace!(sub_inplace_u32, |a: u32, b: u32| a.saturating_sub(b), u32);
binary_op_inplace!(sub_inplace_u64, |a: u64, b: u64| a.saturating_sub(b), u64);
binary_op_inplace!(sub_inplace_i8, |a: i8, b: i8| a.saturating_sub(b), i8);
binary_op_inplace!(sub_inplace_i16, |a: i16, b: i16| a.saturating_sub(b), i16);
binary_op_inplace!(sub_inplace_i32, |a: i32, b: i32| a.saturating_sub(b), i32);
binary_op_inplace!(sub_inplace_i64, |a: i64, b: i64| a.saturating_sub(b), i64);

binary_op_inplace!(mul_inplace_bf16, |a, b| a * b, bf16);
binary_op_inplace!(mul_inplace_f16, |a, b| a * b, f16);
binary_op_inplace!(mul_inplace_f32, |a, b| a * b, f32);
binary_op_inplace!(mul_inplace_f64, |a, b| a * b, f64);
binary_op_inplace!(mul_inplace_bool, |a, b| a ^ b, bool);
binary_op_inplace!(mul_inplace_u8, |a: u8, b: u8| a.saturating_mul(b), u8);
binary_op_inplace!(mul_inplace_u16, |a: u16, b: u16| a.saturating_mul(b), u16);
binary_op_inplace!(mul_inplace_u32, |a: u32, b: u32| a.saturating_mul(b), u32);
binary_op_inplace!(mul_inplace_u64, |a: u64, b: u64| a.saturating_mul(b), u64);
binary_op_inplace!(mul_inplace_i8, |a: i8, b: i8| a.saturating_mul(b), i8);
binary_op_inplace!(mul_inplace_i16, |a: i16, b: i16| a.saturating_mul(b), i16);
binary_op_inplace!(mul_inplace_i32, |a: i32, b: i32| a.saturating_mul(b), i32);
binary_op_inplace!(mul_inplace_i64, |a: i64, b: i64| a.saturating_mul(b), i64);

binary_op_inplace!(
    div_inplace_bf16,
    |a: bf16, b: bf16| if b == bf16::from_f32(0.0) {
        if a == bf16::from_f32(0.0) {
            bf16::from_f32(0.0)
        } else {
            bf16::from_f32(f32::INFINITY)
        }
    } else {
        a / b
    },
    bf16
);
binary_op_inplace!(
    div_inplace_f16,
    |a: f16, b: f16| if b == f16::from_f32(0.0) {
        if a == f16::from_f32(0.0) {
            f16::from_f32(0.0)
        } else {
            f16::from_f32(f32::INFINITY)
        }
    } else {
        a / b
    },
    f16
);
binary_op_inplace!(
    div_inplace_f32,
    |a: f32, b: f32| if b == 0.0 {
        if a == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        a / b
    },
    f32
);
binary_op_inplace!(
    div_inplace_f64,
    |a: f64, b: f64| if b == 0.0 {
        if a == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        a / b
    },
    f64
);
binary_op_inplace!(div_inplace_bool, |a: bool, b: bool| a & b, bool);
binary_op_inplace!(
    div_inplace_u8,
    |a: u8, b: u8| if b < 1 {
        if a == 0 {
            0
        } else {
            u8::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u8
);
binary_op_inplace!(
    div_inplace_u16,
    |a: u16, b: u16| if b < 1 {
        if a == 0 {
            0
        } else {
            u16::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u16
);
binary_op_inplace!(
    div_inplace_u32,
    |a: u32, b: u32| if b < 1 {
        if a == 0 {
            0
        } else {
            u32::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u32
);
binary_op_inplace!(
    div_inplace_u64,
    |a: u64, b: u64| if b < 1 {
        if a == 0 {
            0
        } else {
            u64::MAX
        }
    } else {
        a.saturating_div(b)
    },
    u64
);
binary_op_inplace!(
    div_inplace_i8,
    |a: i8, b: i8| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i8::MIN
        } else {
            i8::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i8
);
binary_op_inplace!(
    div_inplace_i16,
    |a: i16, b: i16| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i16::MIN
        } else {
            i16::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i16
);
binary_op_inplace!(
    div_inplace_i32,
    |a: i32, b: i32| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i32
);
binary_op_inplace!(
    div_inplace_i64,
    |a: i64, b: i64| if b == 0 {
        if a == 0 {
            0
        } else if a < 0 {
            i64::MIN
        } else {
            i64::MAX
        }
    } else {
        a.saturating_div(b)
    },
    i64
);
