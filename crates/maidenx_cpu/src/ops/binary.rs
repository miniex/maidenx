#![allow(clippy::comparison_chain)]

use half::{bf16, f16};
use rayon::prelude::*;

#[inline(always)]
unsafe fn compute_factors(num_dims: usize, dims: &[usize]) -> Vec<usize> {
    let mut factors = vec![1; num_dims];
    // factors[d] = product_{j=d+1}^{num_dims-1} dims[j]
    for d in (0..num_dims).rev() {
        if d + 1 < num_dims {
            factors[d] = factors[d + 1] * dims[d + 1];
        }
    }
    factors
}

#[inline(always)]
fn compute_offset(i: usize, num_dims: usize, _dims: &[usize], factors: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0;
    let mut rem = i;
    for d in 0..num_dims {
        let digit = rem / factors[d];
        offset += digit * strides[d];
        rem %= factors[d];
    }
    offset
}

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
        /// * `dims_and_strides` must be either:
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
            dims_and_strides: *const usize,
            lhs: *const $input_type,
            rhs: *const $input_type,
            out: *mut $output_type,
        ) {
            let dims = if dims_and_strides.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(dims_and_strides, num_dims))
            };

            let lhs_strides = if dims_and_strides.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(dims_and_strides.add(num_dims), num_dims))
            };

            let rhs_strides = if dims_and_strides.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(dims_and_strides.add(2 * num_dims), num_dims))
            };

            let lhs = std::slice::from_raw_parts(lhs, num_els);
            let rhs = std::slice::from_raw_parts(rhs, num_els);
            let out = std::slice::from_raw_parts_mut(out, num_els);

            let is_contiguous = |strides: Option<&[usize]>| -> bool {
                match (dims, strides) {
                    (Some(dims), Some(strides)) => {
                        let mut acc = 1;
                        for d in (0..num_dims).rev() {
                            if strides[d] != acc {
                                return false;
                            }
                            acc *= dims[d];
                        }
                        true
                    }
                    _ => true,
                }
            };

            let lhs_cont = is_contiguous(lhs_strides);
            let rhs_cont = is_contiguous(rhs_strides);

            // 미리 factors를 계산 (비연속인 경우에만)
            let factors = if !lhs_cont || !rhs_cont {
                dims.map(|d| compute_factors(num_dims, d))
            } else {
                None
            };

            out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                let (lhs_idx, rhs_idx) = if !lhs_cont || !rhs_cont {
                    if let (Some(dims), Some(lhs_str), Some(rhs_str), Some(ref fac)) = (dims, lhs_strides, rhs_strides, factors.as_ref()) {
                        (
                            compute_offset(i, num_dims, dims, fac, lhs_str),
                            compute_offset(i, num_dims, dims, fac, rhs_str),
                        )
                    } else {
                        (i, i)
                    }
                } else {
                    (i, i)
                };

                *out_val = $op(lhs[lhs_idx], rhs[rhs_idx]) as $output_type;
            });
        }
    };
}

macro_rules! logical_op {
    ($name:ident, $t:ty, $zero:expr, $op:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// Caller must guarantee that:
        /// * `dims_and_strides` must be either:
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
        pub unsafe fn $name(num_els: usize, num_dims: usize, dims_and_strides: *const usize, lhs: *const $t, rhs: *const $t, out: *mut bool) {
            let dims = if dims_and_strides.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(dims_and_strides, num_dims))
            };
            let lhs_strides = if dims_and_strides.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(dims_and_strides.add(num_dims), num_dims))
            };
            let rhs_strides = if dims_and_strides.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(dims_and_strides.add(2 * num_dims), num_dims))
            };

            let lhs = std::slice::from_raw_parts(lhs, num_els);
            let rhs = std::slice::from_raw_parts(rhs, num_els);
            let out = std::slice::from_raw_parts_mut(out, num_els);

            let is_contiguous = |dims: Option<&[usize]>, strides: Option<&[usize]>| -> bool {
                match (dims, strides) {
                    (Some(d), Some(s)) => {
                        let mut acc = 1;
                        for i in (0..d.len()).rev() {
                            if s[i] != acc {
                                return false;
                            }
                            acc *= d[i];
                        }
                        true
                    }
                    _ => true,
                }
            };

            let lhs_cont = is_contiguous(dims, lhs_strides);
            let rhs_cont = is_contiguous(dims, rhs_strides);

            // 미리 factors 계산 (비연속인 경우)
            let factors = if !lhs_cont || !rhs_cont {
                dims.map(|d| compute_factors(num_dims, d))
            } else {
                None
            };

            out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                let (lhs_idx, rhs_idx) = if !lhs_cont || !rhs_cont {
                    if let (Some(dims), Some(lhs_s), Some(rhs_s), Some(ref fac)) = (dims, lhs_strides, rhs_strides, factors.as_ref()) {
                        (
                            compute_offset(i, num_dims, dims, fac, lhs_s),
                            compute_offset(i, num_dims, dims, fac, rhs_s),
                        )
                    } else {
                        (i, i)
                    }
                } else {
                    (i, i)
                };

                let lhs_bool = lhs[lhs_idx] != $zero;
                let rhs_bool = rhs[rhs_idx] != $zero;
                *out_val = $op(lhs_bool, rhs_bool);
            });
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
binary_op!(add_u32, |a: u32, b: u32| a.saturating_add(b), u32);
binary_op!(add_i8, |a: i8, b: i8| a.saturating_add(b), i8);
binary_op!(add_i32, |a: i32, b: i32| a.saturating_add(b), i32);
binary_op!(add_i64, |a: i64, b: i64| a.saturating_add(b), i64);

binary_op!(sub_bf16, |a, b| a - b, bf16);
binary_op!(sub_f16, |a, b| a - b, f16);
binary_op!(sub_f32, |a, b| a - b, f32);
binary_op!(sub_f64, |a, b| a - b, f64);
binary_op!(sub_bool, |a, b| a & b, bool);
binary_op!(sub_u8, |a: u8, b: u8| a.saturating_sub(b), u8);
binary_op!(sub_u32, |a: u32, b: u32| a.saturating_sub(b), u32);
binary_op!(sub_i8, |a: i8, b: i8| a.saturating_sub(b), i8);
binary_op!(sub_i32, |a: i32, b: i32| a.saturating_sub(b), i32);
binary_op!(sub_i64, |a: i64, b: i64| a.saturating_sub(b), i64);

binary_op!(mul_bf16, |a, b| a * b, bf16);
binary_op!(mul_f16, |a, b| a * b, f16);
binary_op!(mul_f32, |a, b| a * b, f32);
binary_op!(mul_f64, |a, b| a * b, f64);
binary_op!(mul_bool, |a, b| a ^ b, bool);
binary_op!(mul_u8, |a: u8, b: u8| a.saturating_mul(b), u8);
binary_op!(mul_u32, |a: u32, b: u32| a.saturating_mul(b), u32);
binary_op!(mul_i8, |a: i8, b: i8| a.saturating_mul(b), i8);
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
logical_op!(logical_and_u32, u32, 0, |a, b| a && b);
logical_op!(logical_or_u32, u32, 0, |a, b| a || b);
logical_op!(logical_xor_u32, u32, 0, |a, b| a != b);
logical_op!(logical_and_i8, i8, 0, |a, b| a && b);
logical_op!(logical_or_i8, i8, 0, |a, b| a || b);
logical_op!(logical_xor_i8, i8, 0, |a, b| a != b);
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
binary_op_with_output!(eq_u32, |a, b| a == b, u32, bool);
binary_op_with_output!(ne_u32, |a, b| a != b, u32, bool);
binary_op_with_output!(lt_u32, |a, b| a < b, u32, bool);
binary_op_with_output!(le_u32, |a, b| a <= b, u32, bool);
binary_op_with_output!(gt_u32, |a, b| a > b, u32, bool);
binary_op_with_output!(ge_u32, |a, b| a >= b, u32, bool);
binary_op_with_output!(eq_i8, |a, b| a == b, i8, bool);
binary_op_with_output!(ne_i8, |a, b| a != b, i8, bool);
binary_op_with_output!(lt_i8, |a, b| a < b, i8, bool);
binary_op_with_output!(le_i8, |a, b| a <= b, i8, bool);
binary_op_with_output!(gt_i8, |a, b| a > b, i8, bool);
binary_op_with_output!(ge_i8, |a, b| a >= b, i8, bool);
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
