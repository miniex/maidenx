#![allow(clippy::comparison_chain)]
#![allow(clippy::excessive_precision)]

use crate::utils::{get_strided_index, is_contiguous};
use half::{bf16, f16};
use rayon::prelude::*;

macro_rules! unary_op {
    ($name:ident, $type:ty, $func:expr) => {
        unary_op_output!($name, $type, $type, $func);
    };
}

macro_rules! unary_op_output {
    ($name:ident, $input_type:ty, $output_type:ty, $func:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be either:
        ///   - null, indicating contiguous arrays
        ///   - a valid pointer to an array of `2 * num_dims` elements containing:
        ///     * dims[num_dims]: array dimensions
        ///     * strides[num_dims]: strides for array
        /// * `input` must be either:
        ///   - null, in which case output array is used as input
        ///   - a valid pointer to an array of at least `num_els` elements
        /// * `output` must be a valid pointer to an array of at least `num_els` elements
        /// * The memory regions of input (if not null) and output must not overlap
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated from dims and strides must be in bounds
        pub unsafe fn $name(num_els: usize, num_dims: usize, metadata: *const usize, input: *const $input_type, output: *mut $output_type) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let offset = if metadata.is_null() { 0 } else { *metadata.add(2 * num_dims) };

            let inp = std::slice::from_raw_parts(input.add(offset), num_els);
            let out = std::slice::from_raw_parts_mut(output, num_els);

            if is_contiguous(num_dims, dims, strides) {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let x = inp[i];
                    *out_val = $func(x);
                });
            } else {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let strided_i = get_strided_index(i, num_dims, dims, strides);
                    let x = inp[strided_i];
                    *out_val = $func(x);
                });
            }
        }
    };
}

macro_rules! unary_op_with_constant {
    ($name:ident, $type:ty, $func:expr) => {
        unary_op_with_constant_output!($name, $type, $type, $func);
    };
}

macro_rules! unary_op_with_constant_output {
    ($name:ident, $input_type:ty, $output_type:ty, $func:expr) => {
        #[no_mangle]
        /// # Safety
        ///
        /// * `metadata` must be either:
        ///   - null, indicating contiguous arrays
        ///   - a valid pointer to an array of `2 * num_dims` elements containing:
        ///     * dims[num_dims]: array dimensions
        ///     * strides[num_dims]: strides for array
        /// * `input` must be either:
        ///   - null, in which case output array is used as input
        ///   - a valid pointer to an array of at least `num_els` elements
        /// * `constant` must be a valid value for the given type
        /// * `output` must be a valid pointer to an array of at least `num_els` elements
        /// * The memory regions of input (if not null) and output must not overlap
        /// * The alignment requirements of the type must be respected
        /// * All array indices calculated from dims and strides must be in bounds
        pub unsafe fn $name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            input: *const $input_type,
            constant: $input_type,
            output: *mut $output_type,
        ) {
            let dims = std::slice::from_raw_parts(metadata, num_dims);
            let strides = std::slice::from_raw_parts(metadata.add(num_dims), num_dims);
            let offset = if metadata.is_null() { 0 } else { *metadata.add(2 * num_dims) };

            let inp = std::slice::from_raw_parts(input.add(offset), num_els);
            let out = std::slice::from_raw_parts_mut(output, num_els);

            if is_contiguous(num_dims, dims, strides) {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let x = inp[i];
                    *out_val = $func(x, constant);
                });
            } else {
                out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                    let strided_i = get_strided_index(i, num_dims, dims, strides);
                    let x = inp[strided_i];
                    *out_val = $func(x, constant);
                });
            }
        }
    };
}

unary_op!(neg_f32, f32, |x: f32| -x);
unary_op!(neg_f64, f64, |x: f64| -x);
unary_op!(neg_bool, bool, |x: bool| !x);
unary_op!(neg_i8, i8, |x: i8| -x);
unary_op!(neg_i16, i16, |x: i16| -x);
unary_op!(neg_i32, i32, |x: i32| -x);
unary_op!(neg_i64, i64, |x: i64| -x);
unary_op!(neg_f16, f16, |x: f16| -x);
unary_op!(neg_bf16, bf16, |x: bf16| -x);

unary_op!(abs_f32, f32, |x: f32| x.abs());
unary_op!(abs_f64, f64, |x: f64| x.abs());
unary_op!(abs_bool, bool, |x: bool| x);
unary_op!(abs_i8, i8, |x: i8| x.abs());
unary_op!(abs_i16, i16, |x: i16| x.abs());
unary_op!(abs_i32, i32, |x: i32| x.abs());
unary_op!(abs_i64, i64, |x: i64| x.abs());
unary_op!(abs_f16, f16, |x: f16| f16::from_f32(x.to_f32().abs()));
unary_op!(abs_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().abs()));

unary_op!(sign_f32, f32, |x| if x > 0.0 {
    1.0
} else if x < 0.0 {
    -1.0
} else {
    0.0
});
unary_op!(sign_f64, f64, |x| if x > 0.0 {
    1.0
} else if x < 0.0 {
    -1.0
} else {
    0.0
});
unary_op!(sign_bool, bool, |x: bool| x);
unary_op!(sign_u8, u8, |x| if x > 0 { 1 } else { 0 });
unary_op!(sign_u16, u16, |x| if x > 0 { 1 } else { 0 });
unary_op!(sign_u32, u32, |x| if x > 0 { 1 } else { 0 });
unary_op!(sign_u64, u64, |x| if x > 0 { 1 } else { 0 });
unary_op!(sign_i8, i8, |x| if x > 0 {
    1
} else if x < 0 {
    -1
} else {
    0
});
unary_op!(sign_i16, i16, |x| if x > 0 {
    1
} else if x < 0 {
    -1
} else {
    0
});
unary_op!(sign_i32, i32, |x| if x > 0 {
    1
} else if x < 0 {
    -1
} else {
    0
});
unary_op!(sign_i64, i64, |x| if x > 0 {
    1
} else if x < 0 {
    -1
} else {
    0
});
unary_op!(sign_f16, f16, |x| if x > f16::from_f32(0.0) {
    f16::from_f32(1.0)
} else if x < f16::from_f32(0.0) {
    f16::from_f32(-1.0)
} else {
    f16::from_f32(0.0)
});
unary_op!(sign_bf16, bf16, |x| if x > bf16::from_f32(0.0) {
    bf16::from_f32(1.0)
} else if x < bf16::from_f32(0.0) {
    bf16::from_f32(-1.0)
} else {
    bf16::from_f32(0.0)
});

unary_op!(square_f32, f32, |x: f32| x * x);
unary_op!(square_f64, f64, |x: f64| x * x);
unary_op!(square_bool, bool, |x: bool| x);
unary_op!(square_u8, u8, |x: u8| x * x);
unary_op!(square_u16, u16, |x: u16| x * x);
unary_op!(square_u32, u32, |x: u32| x * x);
unary_op!(square_u64, u64, |x: u64| x * x);
unary_op!(square_i8, i8, |x: i8| x * x);
unary_op!(square_i16, i16, |x: i16| x * x);
unary_op!(square_i32, i32, |x: i32| x * x);
unary_op!(square_i64, i64, |x: i64| x * x);
unary_op!(square_f16, f16, |x: f16| x * x);
unary_op!(square_bf16, bf16, |x: bf16| x * x);

unary_op!(sqrt_f32, f32, |x: f32| x.sqrt());
unary_op!(sqrt_f64, f64, |x: f64| x.sqrt());
unary_op!(sqrt_bool, bool, |x: bool| x);
unary_op!(sqrt_u8, u8, |x: u8| (x as f32).sqrt() as u8);
unary_op!(sqrt_u16, u16, |x: u16| (x as f32).sqrt() as u16);
unary_op!(sqrt_u32, u32, |x: u32| (x as f32).sqrt() as u32);
unary_op!(sqrt_u64, u64, |x: u64| (x as f32).sqrt() as u64);
unary_op!(sqrt_i8, i8, |x: i8| (x.abs() as f32).sqrt() as i8);
unary_op!(sqrt_i16, i16, |x: i16| (x.abs() as f32).sqrt() as i16);
unary_op!(sqrt_i32, i32, |x: i32| (x.abs() as f32).sqrt() as i32);
unary_op!(sqrt_i64, i64, |x: i64| (x.abs() as f64).sqrt() as i64);
unary_op!(sqrt_f16, f16, |x: f16| f16::from_f32(x.to_f32().sqrt()));
unary_op!(sqrt_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().sqrt()));

unary_op!(relu_f32, f32, |x: f32| if x > 0.0 { x } else { 0.0 });
unary_op!(relu_f64, f64, |x: f64| if x > 0.0 { x } else { 0.0 });
unary_op!(relu_bool, bool, |x: bool| x);
unary_op!(relu_f16, f16, |x: f16| {
    if x > f16::from_f32(0.0) {
        x
    } else {
        f16::from_f32(0.0)
    }
});
unary_op!(relu_bf16, bf16, |x: bf16| {
    if x > bf16::from_f32(0.0) {
        x
    } else {
        bf16::from_f32(0.0)
    }
});

unary_op!(sigmoid_f32, f32, |x: f32| 1.0 / (1.0 + (-x).exp()));
unary_op!(sigmoid_f64, f64, |x: f64| 1.0 / (1.0 + (-x).exp()));
unary_op!(sigmoid_bool, bool, |x: bool| x);
unary_op!(sigmoid_f16, f16, |x: f16| { f16::from_f32(1.0 / (1.0 + (-x.to_f32()).exp())) });
unary_op!(sigmoid_bf16, bf16, |x: bf16| { bf16::from_f32(1.0 / (1.0 + (-x.to_f32()).exp())) });

unary_op!(tanh_f32, f32, |x: f32| x.tanh());
unary_op!(tanh_f64, f64, |x: f64| x.tanh());
unary_op!(tanh_bool, bool, |x: bool| x);
unary_op!(tanh_f16, f16, |x: f16| f16::from_f32(x.to_f32().tanh()));
unary_op!(tanh_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().tanh()));

unary_op!(gelu_f32, f32, |x: f32| {
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
    0.5 * x * (1.0 + tanh_arg.tanh())
});
unary_op!(gelu_f64, f64, |x: f64| {
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
    0.5 * x * (1.0 + tanh_arg.tanh())
});
unary_op!(gelu_f16, f16, |x: f16| {
    let x_f32 = x.to_f32();
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let tanh_arg = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32);
    f16::from_f32(0.5 * x_f32 * (1.0 + tanh_arg.tanh()))
});
unary_op!(gelu_bf16, bf16, |x: bf16| {
    let x_f32 = x.to_f32();
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let tanh_arg = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32);
    bf16::from_f32(0.5 * x_f32 * (1.0 + tanh_arg.tanh()))
});

unary_op!(sin_f32, f32, |x: f32| x.sin());
unary_op!(sin_f64, f64, |x: f64| x.sin());
unary_op!(sin_bool, bool, |x: bool| x);
unary_op!(sin_f16, f16, |x: f16| f16::from_f32(x.to_f32().sin()));
unary_op!(sin_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().sin()));

unary_op!(cos_f32, f32, |x: f32| x.cos());
unary_op!(cos_f64, f64, |x: f64| x.cos());
unary_op!(cos_bool, bool, |x: bool| x);
unary_op!(cos_f16, f16, |x: f16| f16::from_f32(x.to_f32().cos()));
unary_op!(cos_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().cos()));

unary_op!(tan_f32, f32, |x: f32| x.tan());
unary_op!(tan_f64, f64, |x: f64| x.tan());
unary_op!(tan_bool, bool, |x: bool| x);
unary_op!(tan_f16, f16, |x: f16| f16::from_f32(x.to_f32().tan()));
unary_op!(tan_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().tan()));

unary_op!(ln_f32, f32, |x: f32| x.ln());
unary_op!(ln_f64, f64, |x: f64| x.ln());
unary_op!(ln_f16, f16, |x: f16| f16::from_f32(x.to_f32().ln()));
unary_op!(ln_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().ln()));

unary_op!(log10_f32, f32, |x: f32| x.log10());
unary_op!(log10_f64, f64, |x: f64| x.log10());
unary_op!(log10_f16, f16, |x: f16| f16::from_f32(x.to_f32().log10()));
unary_op!(log10_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().log10()));

unary_op!(log2_f32, f32, |x: f32| x.log2());
unary_op!(log2_f64, f64, |x: f64| x.log2());
unary_op!(log2_f16, f16, |x: f16| f16::from_f32(x.to_f32().log2()));
unary_op!(log2_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().log2()));

unary_op!(exp_f32, f32, |x: f32| x.exp());
unary_op!(exp_f64, f64, |x: f64| x.exp());
unary_op!(exp_f16, f16, |x: f16| f16::from_f32(x.to_f32().exp()));
unary_op!(exp_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().exp()));

unary_op!(exp10_f32, f32, |x: f32| 10.0f32.powf(x));
unary_op!(exp10_f64, f64, |x: f64| 10.0f64.powf(x));
unary_op!(exp10_f16, f16, |x: f16| f16::from_f32(10.0f32.powf(x.to_f32())));
unary_op!(exp10_bf16, bf16, |x: bf16| bf16::from_f32(10.0f32.powf(x.to_f32())));

unary_op!(exp2_f32, f32, |x: f32| x.exp2());
unary_op!(exp2_f64, f64, |x: f64| x.exp2());
unary_op!(exp2_f16, f16, |x: f16| f16::from_f32(x.to_f32().exp2()));
unary_op!(exp2_bf16, bf16, |x: bf16| bf16::from_f32(x.to_f32().exp2()));

unary_op!(softplus_f32, f32, |x: f32| (1.0 + x.exp()).ln());
unary_op!(softplus_f64, f64, |x: f64| (1.0 + x.exp()).ln());
unary_op!(softplus_f16, f16, |x: f16| {
    let val = x.to_f32();
    f16::from_f32((1.0 + val.exp()).ln())
});
unary_op!(softplus_bf16, bf16, |x: bf16| {
    let val = x.to_f32();
    bf16::from_f32((1.0 + val.exp()).ln())
});

unary_op!(recip_f32, f32, |x: f32| 1.0 / x);
unary_op!(recip_f64, f64, |x: f64| 1.0 / x);
unary_op!(recip_f16, f16, |x: f16| f16::from_f32(1.0 / x.to_f32()));
unary_op!(recip_bf16, bf16, |x: bf16| bf16::from_f32(1.0 / x.to_f32()));

unary_op_output!(logical_not_f32, f32, bool, |x: f32| x == 0.0);
unary_op_output!(logical_not_f64, f64, bool, |x: f64| x == 0.0);
unary_op_output!(logical_not_bool, bool, bool, |x: bool| !x);
unary_op_output!(logical_not_u8, u8, bool, |x: u8| x == 0);
unary_op_output!(logical_not_u16, u16, bool, |x: u16| x == 0);
unary_op_output!(logical_not_u32, u32, bool, |x: u32| x == 0);
unary_op_output!(logical_not_u64, u64, bool, |x: u64| x == 0);
unary_op_output!(logical_not_i8, i8, bool, |x: i8| x == 0);
unary_op_output!(logical_not_i16, i16, bool, |x: i16| x == 0);
unary_op_output!(logical_not_i32, i32, bool, |x: i32| x == 0);
unary_op_output!(logical_not_i64, i64, bool, |x: i64| x == 0);
unary_op_output!(logical_not_f16, f16, bool, |x: f16| x == f16::from_f32(0.0));
unary_op_output!(logical_not_bf16, bf16, bool, |x: bf16| x == bf16::from_f32(0.0));

// Operations with constant
unary_op_with_constant!(add_scalar_f32, f32, |x, c| x + c);
unary_op_with_constant!(add_scalar_f64, f64, |x, c| x + c);
unary_op_with_constant!(add_scalar_bool, bool, |x, c| x || c);
unary_op_with_constant!(add_scalar_u8, u8, |x: u8, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_u16, u16, |x: u16, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_u32, u32, |x: u32, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_u64, u64, |x: u64, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_i8, i8, |x: i8, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_i16, i16, |x: i16, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_i32, i32, |x: i32, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_i64, i64, |x: i64, c| x.saturating_add(c));
unary_op_with_constant!(add_scalar_f16, f16, |x, c| x + c);
unary_op_with_constant!(add_scalar_bf16, bf16, |x, c| x + c);

unary_op_with_constant!(sub_scalar_f32, f32, |x, c| x - c);
unary_op_with_constant!(sub_scalar_f64, f64, |x, c| x - c);
unary_op_with_constant!(sub_scalar_bool, bool, |x, c| x ^ c);
unary_op_with_constant!(sub_scalar_u8, u8, |x: u8, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_u16, u16, |x: u16, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_u32, u32, |x: u32, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_u64, u64, |x: u64, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_i8, i8, |x: i8, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_i16, i16, |x: i16, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_i32, i32, |x: i32, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_i64, i64, |x: i64, c| x.saturating_sub(c));
unary_op_with_constant!(sub_scalar_f16, f16, |x, c| x - c);
unary_op_with_constant!(sub_scalar_bf16, bf16, |x, c| x - c);

unary_op_with_constant!(mul_scalar_f32, f32, |x, c| x * c);
unary_op_with_constant!(mul_scalar_f64, f64, |x, c| x * c);
unary_op_with_constant!(mul_scalar_bool, bool, |x, c| x && c);
unary_op_with_constant!(mul_scalar_u8, u8, |x: u8, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_u16, u16, |x: u16, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_u32, u32, |x: u32, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_u64, u64, |x: u64, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_i8, i8, |x: i8, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_i16, i16, |x: i16, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_i32, i32, |x: i32, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_i64, i64, |x: i64, c| x.saturating_mul(c));
unary_op_with_constant!(mul_scalar_f16, f16, |x, c| x * c);
unary_op_with_constant!(mul_scalar_bf16, bf16, |x, c| x * c);

unary_op_with_constant!(div_scalar_bf16, bf16, |x: bf16, c: bf16| if c == bf16::from_f32(0.0) {
    if x == bf16::from_f32(0.0) {
        bf16::from_f32(0.0)
    } else {
        bf16::from_f32(f32::INFINITY)
    }
} else {
    x / c
});
unary_op_with_constant!(div_scalar_f16, f16, |x: f16, c: f16| if c == f16::from_f32(0.0) {
    if x == f16::from_f32(0.0) {
        f16::from_f32(0.0)
    } else {
        f16::from_f32(f32::INFINITY)
    }
} else {
    x / c
});
unary_op_with_constant!(div_scalar_f32, f32, |x: f32, c: f32| if c == 0.0 {
    if x == 0.0 {
        0.0
    } else {
        f32::INFINITY
    }
} else {
    x / c
});
unary_op_with_constant!(div_scalar_f64, f64, |x: f64, c: f64| if c == 0.0 {
    if x == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
} else {
    x / c
});
unary_op_with_constant!(div_scalar_bool, bool, |x: bool, c: bool| x && c);
unary_op_with_constant!(div_scalar_u8, u8, |x: u8, c: u8| if c < 1 {
    if x == 0 {
        0
    } else {
        u8::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_u16, u16, |x: u16, c: u16| if c < 1 {
    if x == 0 {
        0
    } else {
        u16::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_u32, u32, |x: u32, c: u32| if c < 1 {
    if x == 0 {
        0
    } else {
        u32::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_u64, u64, |x: u64, c: u64| if c < 1 {
    if x == 0 {
        0
    } else {
        u64::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_i8, i8, |x: i8, c: i8| if c == 0 {
    if x == 0 {
        0
    } else if x < 0 {
        i8::MIN
    } else {
        i8::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_i16, i16, |x: i16, c: i16| if c == 0 {
    if x == 0 {
        0
    } else if x < 0 {
        i16::MIN
    } else {
        i16::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_i32, i32, |x: i32, c: i32| if c == 0 {
    if x == 0 {
        0
    } else if x < 0 {
        i32::MIN
    } else {
        i32::MAX
    }
} else {
    x.saturating_div(c)
});
unary_op_with_constant!(div_scalar_i64, i64, |x: i64, c: i64| if c == 0 {
    if x == 0 {
        0
    } else if x < 0 {
        i64::MIN
    } else {
        i64::MAX
    }
} else {
    x.saturating_div(c)
});

unary_op_with_constant!(maximum_scalar_bf16, bf16, |x, c| if x > c { x } else { c });
unary_op_with_constant!(maximum_scalar_f16, f16, |x, c| if x > c { x } else { c });
unary_op_with_constant!(maximum_scalar_f32, f32, |x: f32, c| x.max(c));
unary_op_with_constant!(maximum_scalar_f64, f64, |x: f64, c| x.max(c));
unary_op_with_constant!(maximum_scalar_bool, bool, |x, c| x | c);
unary_op_with_constant!(maximum_scalar_u8, u8, |x: u8, c: u8| x.max(c));
unary_op_with_constant!(maximum_scalar_u16, u16, |x: u16, c: u16| x.max(c));
unary_op_with_constant!(maximum_scalar_u32, u32, |x: u32, c: u32| x.max(c));
unary_op_with_constant!(maximum_scalar_u64, u64, |x: u64, c: u64| x.max(c));
unary_op_with_constant!(maximum_scalar_i8, i8, |x: i8, c: i8| x.max(c));
unary_op_with_constant!(maximum_scalar_i16, i16, |x: i16, c: i16| x.max(c));
unary_op_with_constant!(maximum_scalar_i32, i32, |x: i32, c: i32| x.max(c));
unary_op_with_constant!(maximum_scalar_i64, i64, |x: i64, c: i64| x.max(c));

unary_op_with_constant!(minimum_scalar_bf16, bf16, |x, c| if x < c { x } else { c });
unary_op_with_constant!(minimum_scalar_f16, f16, |x, c| if x < c { x } else { c });
unary_op_with_constant!(minimum_scalar_f32, f32, |x: f32, c| x.min(c));
unary_op_with_constant!(minimum_scalar_f64, f64, |x: f64, c| x.min(c));
unary_op_with_constant!(minimum_scalar_bool, bool, |x, c| x & c);
unary_op_with_constant!(minimum_scalar_u8, u8, |x: u8, c: u8| x.min(c));
unary_op_with_constant!(minimum_scalar_u16, u16, |x: u16, c: u16| x.min(c));
unary_op_with_constant!(minimum_scalar_u32, u32, |x: u32, c: u32| x.min(c));
unary_op_with_constant!(minimum_scalar_u64, u64, |x: u64, c: u64| x.min(c));
unary_op_with_constant!(minimum_scalar_i8, i8, |x: i8, c: i8| x.min(c));
unary_op_with_constant!(minimum_scalar_i16, i16, |x: i16, c: i16| x.min(c));
unary_op_with_constant!(minimum_scalar_i32, i32, |x: i32, c: i32| x.min(c));
unary_op_with_constant!(minimum_scalar_i64, i64, |x: i64, c: i64| x.min(c));

unary_op_with_constant!(pow_f32, f32, |x: f32, c: f32| x.powf(c));
unary_op_with_constant!(pow_f64, f64, |x: f64, c: f64| x.powf(c));
unary_op_with_constant!(pow_bool, bool, |x: bool, _c: bool| x);
unary_op_with_constant!(pow_u8, u8, |x: u8, c: u8| (x as f32).powf(c as f32) as u8);
unary_op_with_constant!(pow_u16, u16, |x: u16, c: u16| (x as f64).powf(c as f64) as u16);
unary_op_with_constant!(pow_u32, u32, |x: u32, c: u32| (x as f64).powf(c as f64) as u32);
unary_op_with_constant!(pow_u64, u64, |x: u64, c: u64| (x as f64).powf(c as f64) as u64);
unary_op_with_constant!(pow_i8, i8, |x: i8, c: i8| (x as f64).powf(c as f64) as i8);
unary_op_with_constant!(pow_i16, i16, |x: i16, c: i16| (x as f64).powf(c as f64) as i16);
unary_op_with_constant!(pow_i32, i32, |x: i32, c: i32| (x as f64).powf(c as f64) as i32);
unary_op_with_constant!(pow_i64, i64, |x: i64, c: i64| (x as f64).powf(c as f64) as i64);
unary_op_with_constant!(pow_f16, f16, |x: f16, c: f16| { f16::from_f32(x.to_f32().powf(c.to_f32())) });
unary_op_with_constant!(pow_bf16, bf16, |x: bf16, c: bf16| { bf16::from_f32(x.to_f32().powf(c.to_f32())) });

unary_op_with_constant!(leaky_relu_f32, f32, |x: f32, alpha: f32| if x > 0.0 { x } else { alpha * x });
unary_op_with_constant!(leaky_relu_f64, f64, |x: f64, alpha: f64| if x > 0.0 { x } else { alpha * x });
unary_op_with_constant!(leaky_relu_f16, f16, |x: f16, alpha: f16| {
    if x > f16::from_f32(0.0) {
        x
    } else {
        alpha * x
    }
});
unary_op_with_constant!(leaky_relu_bf16, bf16, |x: bf16, alpha: bf16| {
    if x > bf16::from_f32(0.0) {
        x
    } else {
        alpha * x
    }
});

unary_op_with_constant!(elu_f32, f32, |x: f32, alpha: f32| {
    if x > 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
});
unary_op_with_constant!(elu_f64, f64, |x: f64, alpha: f64| {
    if x > 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
});
unary_op_with_constant!(elu_f16, f16, |x: f16, alpha: f16| {
    let x_f32 = x.to_f32();
    let alpha_f32 = alpha.to_f32();
    if x_f32 > 0.0 {
        x
    } else {
        f16::from_f32(alpha_f32 * (x_f32.exp() - 1.0))
    }
});
unary_op_with_constant!(elu_bf16, bf16, |x: bf16, alpha: bf16| {
    let x_f32 = x.to_f32();
    let alpha_f32 = alpha.to_f32();
    if x_f32 > 0.0 {
        x
    } else {
        bf16::from_f32(alpha_f32 * (x_f32.exp() - 1.0))
    }
});

// Comparison ops with constant (output = u8)
// -- f32
unary_op_with_constant_output!(eq_scalar_f32, f32, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_f32, f32, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_f32, f32, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_f32, f32, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_f32, f32, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_f32, f32, bool, |x, c| x >= c);

// -- f64
unary_op_with_constant_output!(eq_scalar_f64, f64, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_f64, f64, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_f64, f64, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_f64, f64, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_f64, f64, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_f64, f64, bool, |x, c| x >= c);

// -- bool
unary_op_with_constant_output!(eq_scalar_bool, bool, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_bool, bool, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_bool, bool, bool, |x: bool, c: bool| (!x) && c);
unary_op_with_constant_output!(le_scalar_bool, bool, bool, |x: bool, c: bool| (!x) || c);
unary_op_with_constant_output!(gt_scalar_bool, bool, bool, |x: bool, c: bool| x && (!c));
unary_op_with_constant_output!(ge_scalar_bool, bool, bool, |x: bool, c: bool| x || (!c));

// -- u8
unary_op_with_constant_output!(eq_scalar_u8, u8, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_u8, u8, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_u8, u8, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_u8, u8, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_u8, u8, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_u8, u8, bool, |x, c| x >= c);

// -- u16
unary_op_with_constant_output!(eq_scalar_u16, u16, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_u16, u16, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_u16, u16, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_u16, u16, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_u16, u16, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_u16, u16, bool, |x, c| x >= c);

// -- u32
unary_op_with_constant_output!(eq_scalar_u32, u32, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_u32, u32, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_u32, u32, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_u32, u32, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_u32, u32, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_u32, u32, bool, |x, c| x >= c);

// -- u64
unary_op_with_constant_output!(eq_scalar_u64, u64, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_u64, u64, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_u64, u64, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_u64, u64, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_u64, u64, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_u64, u64, bool, |x, c| x >= c);

// -- i8
unary_op_with_constant_output!(eq_scalar_i8, i8, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_i8, i8, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_i8, i8, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_i8, i8, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_i8, i8, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_i8, i8, bool, |x, c| x >= c);

// -- i16
unary_op_with_constant_output!(eq_scalar_i16, i16, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_i16, i16, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_i16, i16, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_i16, i16, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_i16, i16, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_i16, i16, bool, |x, c| x >= c);

// -- i32
unary_op_with_constant_output!(eq_scalar_i32, i32, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_i32, i32, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_i32, i32, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_i32, i32, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_i32, i32, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_i32, i32, bool, |x, c| x >= c);

// -- i64
unary_op_with_constant_output!(eq_scalar_i64, i64, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_i64, i64, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_i64, i64, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_i64, i64, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_i64, i64, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_i64, i64, bool, |x, c| x >= c);

// -- f16
unary_op_with_constant_output!(eq_scalar_f16, f16, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_f16, f16, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_f16, f16, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_f16, f16, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_f16, f16, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_f16, f16, bool, |x, c| x >= c);

// -- bf16
unary_op_with_constant_output!(eq_scalar_bf16, bf16, bool, |x, c| x == c);
unary_op_with_constant_output!(ne_scalar_bf16, bf16, bool, |x, c| x != c);
unary_op_with_constant_output!(lt_scalar_bf16, bf16, bool, |x, c| x < c);
unary_op_with_constant_output!(le_scalar_bf16, bf16, bool, |x, c| x <= c);
unary_op_with_constant_output!(gt_scalar_bf16, bf16, bool, |x, c| x > c);
unary_op_with_constant_output!(ge_scalar_bf16, bf16, bool, |x, c| x >= c);
