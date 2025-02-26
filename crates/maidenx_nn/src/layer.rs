use maidenx_core::error::Result;
pub use maidenx_nn_macros::Layer;
use maidenx_tensor::Tensor;

pub trait Layer<I = &'static Tensor> {
    fn forward(&self, input: I) -> Result<Tensor>;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
}
