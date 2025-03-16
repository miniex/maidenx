use half::{bf16, f16};

#[link(name = "ops")]
extern "C" {}

#[macro_export]
macro_rules! declare_extern_unary_ops {
    ($(
        $dtype:ident => {
            type: $ty:ty,
            ops: [
                $($arithmetic_op:ident),* ;
                $($comparison_op:ident),*
            ]
        }
    ),*) => {
        paste::paste! {
            extern "C" {
                $(
                    $(
                        pub fn [<cuda_ $arithmetic_op _ $dtype:lower>](
                            num_els: usize,
                            num_dims: usize,
                            metadata: *const usize,
                            input: *const $ty,
                            output: *mut $ty,
                        );
                    )*
                    $(
                        pub fn [<cuda_ $comparison_op _ $dtype:lower>](
                            num_els: usize,
                            num_dims: usize,
                            metadata: *const usize,
                            input: *const $ty,
                            output: *mut bool,
                        );
                    )*
                )*
            }
        }
    }
}

#[macro_export]
macro_rules! declare_extern_unary_ops_with_constant {
    ($(
        $dtype:ident => {
            type: $ty:ty,
            ops: [
                $($arithmetic_op:ident),* ;
                $($comparison_op:ident),*
            ]
        }
    ),*) => {
        paste::paste! {
            extern "C" {
                $(
                    $(
                        pub fn [<cuda_ $arithmetic_op _ $dtype:lower>](
                            num_els: usize,
                            num_dims: usize,
                            metadata: *const usize,
                            input: *const $ty,
                            constant: $ty,
                            output: *mut $ty,
                        );
                    )*
                    $(
                        pub fn [<cuda_ $comparison_op _ $dtype:lower>](
                            num_els: usize,
                            num_dims: usize,
                            metadata: *const usize,
                            input: *const $ty,
                            constant: $ty,
                            output: *mut bool,
                        );
                    )*
                )*
            }
        }
    }
}

declare_extern_unary_ops! {
    BF16 => {
        type: bf16,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    F16 => {
        type: f16,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    F32 => {
        type: f32,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    F64 => {
        type: f64,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    BOOL => {
        type: bool,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh; logical_not]
    },
    U8 => {
        type: u8,
        ops: [sign, square, sqrt; logical_not]
    },
    U32 => {
        type: u32,
        ops: [sign, square, sqrt; logical_not]
    },
    I8 => {
        type: i8,
        ops: [neg, abs, sign, square, sqrt; logical_not]
    },
    I32 => {
        type: i32,
        ops: [neg, abs, sign, square, sqrt; logical_not]
    },
    I64 => {
        type: i64,
        ops: [neg, abs, sign, square, sqrt; logical_not]
    }
}

declare_extern_unary_ops_with_constant! {
    BF16 => {
        type: bf16,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    F16 => {
        type: f16,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    F32 => {
        type: f32,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    F64 => {
        type: f64,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    BOOL => {
        type: bool,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    U8 => {
        type: u8,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    U32 => {
        type: u32,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I8 => {
        type: i8,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I32 => {
        type: i32,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I64 => {
        type: i64,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    }
}
