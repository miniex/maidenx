pub mod adapter;
mod creation;
mod d;
mod f;
mod ops;
pub mod utils;
mod vec;
mod wt;

use maidenx_core::{
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

#[derive(Clone)]
pub struct TensorData {
    buffer: Arc<RwLock<dyn Buffer>>,
    layout: Layout,
    grad: Option<Arc<Mutex<Tensor>>>,
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct TensorNode {
    op_name: String,
    inputs: Vec<Tensor>,
    backward_fn: Option<Arc<Box<dyn Fn(&[Tensor], &Tensor) -> Result<Vec<Tensor>> + Send + Sync>>>,
}

#[derive(Clone)]
pub struct Tensor {
    data: TensorData,
    node: Option<TensorNode>,
    device: Device,
    dtype: DType,
    requires_grad: bool,
}

#[allow(clippy::type_complexity)]
impl TensorNode {
    pub fn new(
        op_name: String,
        inputs: Vec<Tensor>,
        backward_fn: Option<Box<dyn Fn(&[Tensor], &Tensor) -> Result<Vec<Tensor>> + Send + Sync>>,
    ) -> Self {
        Self {
            op_name,
            inputs,
            backward_fn: backward_fn.map(|f| Arc::new(f)),
        }
    }

    pub fn op_name(&self) -> &str {
        &self.op_name
    }

    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub fn backward_fn(&self) -> Option<&Arc<Box<dyn Fn(&[Tensor], &Tensor) -> Result<Vec<Tensor>> + Send + Sync>>> {
        self.backward_fn.as_ref()
    }

    pub fn backward(&self, grad_output: &Tensor) -> Result<()> {
        if let Some(ref func) = self.backward_fn {
            let grads_for_inputs = (func)(&self.inputs, grad_output)?;
            for (input, grad_in) in self.inputs.iter().zip(grads_for_inputs.iter()) {
                if input.requires_grad() {
                    input.accumulate_grad(grad_in)?;
                    input._backward(grad_in)?;
                }
            }
        }
        Ok(())
    }
}

impl Tensor {
    // data

    pub fn buffer(&self) -> Result<RwLockReadGuard<'_, dyn Buffer>> {
        self.data.buffer.read().map_err(|_| Error::BufferLocked)
    }

    pub fn with_buffer_mut<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&mut dyn Buffer) -> Result<R>,
    {
        let mut guard = self.data.buffer.write().map_err(|_| Error::BufferLocked)?;

        func(&mut *guard)
    }

    pub fn layout(&self) -> &Layout {
        &self.data.layout
    }

    pub fn layout_mut(&mut self) -> &mut Layout {
        &mut self.data.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.data.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.data.layout.strides()
    }

    pub fn size(&self) -> usize {
        self.data.layout.size()
    }

    pub fn ndim(&self) -> usize {
        self.data.layout.ndim()
    }

    pub fn size_dim(&self, dim: usize) -> Option<usize> {
        self.data.layout.size_dim(dim)
    }

    // data - grad

    pub fn grad(&self) -> Result<Option<Tensor>> {
        Ok(match &self.data.grad {
            Some(g) => Some((*g.lock().map_err(|_| Error::GradLocked)?).clone()),
            None => None,
        })
    }

    pub fn accumulate_grad(&self, grad_in: &Tensor) -> Result<()> {
        if let Some(grad_mutex) = &self.data.grad {
            let mut guard = grad_mutex.lock().map_err(|_| Error::GradLocked)?;
            let updated = guard.add(grad_in)?;
            *guard = updated;
        }
        Ok(())
    }

    pub fn zero_grad(&self) -> Result<()> {
        if let Some(grad_mutex) = &self.data.grad {
            let mut guard = grad_mutex.lock().map_err(|_| Error::GradLocked)?;
            let zero_tensor = Tensor::zeros_like(&guard)?;
            *guard = zero_tensor;
        }

        if let Some(node) = &self.node {
            for input in node.inputs() {
                if input.requires_grad() {
                    input.zero_grad()?;
                }
            }
        }

        Ok(())
    }

    // node

    pub fn set_node(&mut self, node: TensorNode) {
        self.node = Some(node);
    }

    pub fn backward(&self) -> Result<()> {
        if self.requires_grad {
            let grad_out = Self::ones_like(self)?;
            self._backward(&grad_out)?;
        }

        Ok(())
    }

    fn _backward(&self, grad_out: &Tensor) -> Result<()> {
        if let Some(ref node) = self.node {
            node.backward(grad_out)?;
        }

        Ok(())
    }

    // etc

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}
