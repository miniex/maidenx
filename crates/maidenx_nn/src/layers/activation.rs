pub mod softmax;

use crate::layer::{Layer, LayerState};
use maidenx_core::{error::Result, scalar::Scalar};
use maidenx_tensor::Tensor;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
// Re-exports
pub use softmax::*;

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReLU {
    state: LayerState,
}

impl ReLU {
    pub fn new() -> Result<Self> {
        Ok(Self {
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sigmoid {
    state: LayerState,
}

impl Sigmoid {
    pub fn new() -> Result<Self> {
        Ok(Self {
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.sigmoid()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tanh {
    state: LayerState,
}

impl Tanh {
    pub fn new() -> Result<Self> {
        Ok(Self {
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanh()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LeakyReLU {
    exponent: Scalar,

    state: LayerState,
}

impl LeakyReLU {
    pub fn new(exponent: impl Into<Scalar>) -> Result<Self> {
        Ok(Self {
            exponent: exponent.into(),
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.leaky_relu(self.exponent)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GELU {
    state: LayerState,
}

impl GELU {
    pub fn new() -> Result<Self> {
        Ok(Self {
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.gelu()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Layer, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ELU {
    exponent: Scalar,

    state: LayerState,
}

impl ELU {
    pub fn new(exponent: impl Into<Scalar>) -> Result<Self> {
        Ok(Self {
            exponent: exponent.into(),
            state: LayerState::new(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.elu(self.exponent)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}
