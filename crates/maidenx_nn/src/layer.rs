use maidenx_core::error::Result;
pub use maidenx_nn_macros::Layer;
use maidenx_tensor::Tensor;

pub trait Layer<I = &'static Tensor> {
    fn forward(&self, input: I) -> Result<Tensor>;
    fn parameters(&mut self) -> Vec<&mut Tensor>;

    fn is_training(&self) -> bool;
    fn train(&mut self);
    fn eval(&mut self);
}

#[derive(Clone)]
pub struct LayerState {
    training: bool,
}

impl Default for LayerState {
    fn default() -> Self {
        Self { training: true }
    }
}

impl LayerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_training(&self) -> bool {
        self.training
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}
