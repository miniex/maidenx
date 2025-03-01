use crate::layer::Layer;
use maidenx_core::error::Result;
use maidenx_tensor::Tensor;

#[derive(Layer, Clone)]
pub struct ReLU {}

impl ReLU {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.sigmoid()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
pub struct Tanh {}

impl Tanh {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanh()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}
