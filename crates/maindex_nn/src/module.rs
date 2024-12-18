use crate::error::{NnError, NnResult};
pub use maidenx_nn_macros::Module;
use maidenx_tensor::Tensor;

pub trait Module<I = &'static Tensor> {
    fn forward(&self, input: I) -> NnResult<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
}

#[derive(Default, Module)]
pub struct ModuleBuilder {
    layers: Vec<Box<dyn for<'a> Module<&'a Tensor>>>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer<M: for<'a> Module<&'a Tensor> + 'static>(mut self, module: M) -> Self {
        self.layers.push(Box::new(module));
        self
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x).map_err(|e| {
                NnError::InvalidOperation(format!("Layer forward pass failed: {}", e))
            })?;
        }
        Ok(x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
