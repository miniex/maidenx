use half::{bf16, f16};

#[link(name = "nn")]
extern "C" {}

#[macro_export]
macro_rules! declare_extern_softmax_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            extern "C" {
                $(
                    pub fn [<cuda_softmax_ $dtype>](
                        num_els: usize,
                        num_dims: usize,
                        dim: usize,
                        metadata: *const usize,
                        input: *const $ty,
                        output: *mut $ty,
                    );
                )*
            }
        }
    }
}

declare_extern_softmax_ops! {
    bf16 => bf16,
    f16 => f16,
    f32 => f32,
    f64 => f64
}
