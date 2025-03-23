use half::{bf16, f16};

#[link(name = "ops")]
extern "C" {}

#[macro_export]
macro_rules! declare_padding_ops {
    ($($dtype:ident: $ty:ty),*) => {
        paste::paste! {
            extern "C" {
                $(
                    // Constant padding (forward)
                    pub fn [<cuda_pad_with_constant_ $dtype:lower>](
                        num_els_in: usize,
                        num_els_out: usize,
                        num_dims: usize,
                        info: *const usize,
                        inp: *const $ty,
                        out: *mut $ty,
                        pad_value: $ty
                    );

                    // Reflection padding (forward)
                    pub fn [<cuda_pad_with_reflection_ $dtype:lower>](
                        num_els_in: usize,
                        num_els_out: usize,
                        num_dims: usize,
                        info: *const usize,
                        inp: *const $ty,
                        out: *mut $ty
                    );

                    // Replication padding (forward)
                    pub fn [<cuda_pad_with_replication_ $dtype:lower>](
                        num_els_in: usize,
                        num_els_out: usize,
                        num_dims: usize,
                        info: *const usize,
                        inp: *const $ty,
                        out: *mut $ty
                    );

                    // Constant padding (backward)
                    pub fn [<cuda_pad_with_constant_backward_ $dtype:lower>](
                        num_els_in: usize,
                        num_els_out: usize,
                        num_dims: usize,
                        info: *const usize,
                        grad_out: *const $ty,
                        grad_in: *mut $ty
                    );

                    // Reflection padding (backward)
                    pub fn [<cuda_pad_with_reflection_backward_ $dtype:lower>](
                        num_els_in: usize,
                        num_els_out: usize,
                        num_dims: usize,
                        info: *const usize,
                        grad_out: *const $ty,
                        grad_in: *mut $ty
                    );

                    // Replication padding (backward)
                    pub fn [<cuda_pad_with_replication_backward_ $dtype:lower>](
                        num_els_in: usize,
                        num_els_out: usize,
                        num_dims: usize,
                        info: *const usize,
                        grad_out: *const $ty,
                        grad_in: *mut $ty
                    );
                )*
            }
        }
    }
}

declare_padding_ops! {
    BF16: bf16,
    F16: f16,
    F32: f32,
    F64: f64,
    U8: u8,
    U16: u16,
    U32: u32,
    U64: u64,
    I8: i8,
    I16: i16,
    I32: i32,
    I64: i64
}
