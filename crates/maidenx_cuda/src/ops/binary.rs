use half::{bf16, f16};

#[link(name = "ops")]
extern "C" {}

#[macro_export]
macro_rules! declare_extern_binary_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            extern "C" {
                $(
                    pub fn [<cuda_add_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut $ty,
                    );
                    pub fn [<cuda_sub_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut $ty,
                    );
                    pub fn [<cuda_mul_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut $ty,
                    );
                    pub fn [<cuda_div_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut $ty,
                    );

                    pub fn [<cuda_logical_and_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_logical_or_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_logical_xor_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );

                    pub fn [<cuda_eq_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_ne_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_lt_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_le_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_gt_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                    pub fn [<cuda_ge_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        lhs: *const $ty,
                        rhs: *const $ty,
                        out: *mut bool,
                    );
                )*
            }
        }
    }
}

declare_extern_binary_ops! {
    bf16 => bf16,
    f16 => f16,
    f32 => f32,
    f64 => f64,
    bool => bool,
    u8 => u8,
    u32 => u32,
    i8 => i8,
    i32 => i32,
    i64 => i64
}
