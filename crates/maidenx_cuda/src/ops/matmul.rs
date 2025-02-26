use half::{bf16, f16};

#[link(name = "ops")]
extern "C" {}

#[macro_export]
macro_rules! declare_extern_matmul_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            extern "C" {
                $(
                    pub fn [<cuda_matmul_ $dtype>](
                        num_els: usize,
                        dims_and_strides: *const usize,
                        a: *const $ty,
                        b: *const $ty,
                        c: *mut $ty,
                    );

                    pub fn [<cuda_matmul_backward_ $dtype>](
                        num_els_a: usize,
                        num_els_b: usize,
                        dims_and_strides: *const usize,
                        grad_output: *const $ty,
                        a: *const $ty,
                        b: *const $ty,
                        grad_a: *mut $ty,
                        grad_b: *mut $ty,
                    );
                )*
            }
        }
    }
}

declare_extern_matmul_ops! {
    bf16 => bf16,
    f16 => f16,
    f32 => f32,
    f64 => f64,
    u8 => u8,
    u32 => u32,
    i8 => i8,
    i32 => i32,
    i64 => i64
}
