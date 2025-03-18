use half::{bf16, f16};

#[link(name = "ops")]
extern "C" {}

#[macro_export]
macro_rules! declare_extern_reduction_ops {
    ($(
        $dtype:ident => {
            type: $ty:ty,
            standard_ops: [$($std_op:ident),*],
            shape_ops: [$($shape_op:ident),*]
        }
    ),*) => {
        paste::paste! {
            extern "C" {
                $(
                    $(
                        pub fn [<cuda_ $std_op _ $dtype:lower>](
                            num_els: usize,
                            num_dims: usize,
                            num_red_dims: usize,
                            metadata: *const usize,
                            inp: *const $ty,
                            out: *mut $ty,
                        );
                    )*
                    $(
                        pub fn [<cuda_ $shape_op _ $dtype:lower>](
                            num_els: usize,
                            num_dims: usize,
                            metadata: *const usize,
                            inp: *const $ty,
                            out: *mut $ty,
                        );
                    )*
                )*
            }
        }
    }
}

declare_extern_reduction_ops! {
    BF16 => {
        type: bf16,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    F16 => {
        type: f16,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    F32 => {
        type: f32,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    F64 => {
        type: f64,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    U8 => {
        type: u8,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    U32 => {
        type: u32,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I8 => {
        type: i8,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I32 => {
        type: i32,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I64 => {
        type: i64,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    }
}
