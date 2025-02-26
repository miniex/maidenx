use half::{bf16, f16};

#[link(name = "nn")]
extern "C" {}

#[macro_export]
macro_rules! declare_extern_conv_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            extern "C" {
                $(
                    pub fn [<cuda_conv2d_im2col_ $dtype>](
                        num_els: usize,
                        dims_and_strides: *const usize,
                        input: *const $ty,
                        col: *mut $ty,
                    );

                    pub fn [<cuda_conv2d_col2im_ $dtype>](
                        num_els: usize,
                        dims_and_strides: *const usize,
                        col: *const $ty,
                        output: *mut $ty,
                    );
                )*
            }
        }
    }
}

declare_extern_conv_ops! {
    bf16 => bf16,
    f16 => f16,
    f32 => f32,
    f64 => f64
}
