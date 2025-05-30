pub mod adapter;
mod creation;
mod d;
mod f;
mod iterator;
mod operators;
pub(crate) mod ops;
#[cfg(feature = "serde")]
mod serde;
pub mod utils;
mod vec;
mod wt;

use iterator::TensorIterator;
#[cfg(feature = "cuda")]
use maidenx_core::buffer::cuda::CudaBuffer;
#[cfg(feature = "mps")]
use maidenx_core::buffer::mps::MpsBuffer;
use maidenx_core::{
    buffer::{cpu::CpuBuffer, Buffer},
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};
use std::{
    collections::HashSet,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct TensorData {
    buffer: Arc<dyn Buffer>,
    grad: Option<Arc<Mutex<Tensor>>>,
}

#[derive(Clone)]
pub struct TensorMetadata {
    device: Device,
    dtype: DType,
    layout: Layout,
    requires_grad: bool,
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
    metadata: TensorMetadata,
    node: Option<TensorNode>,
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
        let mut visited = std::collections::HashSet::<u64>::new();
        self.backward_with_tracking(grad_output, &mut visited)
    }

    pub fn backward_with_tracking(
        &self,
        grad_output: &Tensor,
        visited: &mut std::collections::HashSet<u64>,
    ) -> Result<()> {
        if let Some(ref func) = self.backward_fn {
            let grads_for_inputs = (func)(&self.inputs, grad_output)?;

            for (idx, input) in self.inputs.iter().enumerate() {
                if input.requires_grad() {
                    if let Some(grad) = grads_for_inputs.get(idx) {
                        input.accumulate_grad(grad)?;
                        input._backward_with_tracking(grad, visited)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Tensor {
    // data

    pub fn buffer(&self) -> &dyn Buffer {
        Arc::as_ref(&self.data.buffer)
    }

    // Don't use it!!
    pub fn buffer_mut(&mut self) -> &mut dyn Buffer {
        unsafe {
            let buffer_ptr = Arc::as_ptr(&self.data.buffer) as *mut dyn Buffer;
            &mut *buffer_ptr
        }
    }

    fn buffer_clone(&self) -> Result<Arc<dyn Buffer>> {
        let src_buffer = self.buffer();
        let device = src_buffer.device();
        let dtype = src_buffer.dtype();
        let size = src_buffer.len();

        let new_buffer: Arc<dyn Buffer> = match device {
            Device::CPU => {
                let buffer = CpuBuffer::new(size, dtype)?;
                Arc::new(buffer)
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let buffer = CudaBuffer::new(size, dtype, device_id)?;
                Arc::new(buffer)
            },
            #[cfg(feature = "mps")]
            Device::MPS => {
                let buffer = MpsBuffer::new(size, dtype)?;
                Arc::new(buffer)
            },
        };

        unsafe {
            let buffer_ptr = Arc::into_raw(new_buffer) as *mut dyn Buffer;
            (*buffer_ptr).copy_from(src_buffer, 0, 0, src_buffer.len())?;
            let new_buffer = Arc::from_raw(buffer_ptr);
            Ok(new_buffer)
        }
    }

    pub fn with_buffer_mut<F, R>(&mut self, func: F) -> Result<R>
    where
        F: FnOnce(&mut dyn Buffer) -> Result<R>,
    {
        if Arc::strong_count(&self.data.buffer) == 1 {
            let buffer = Arc::get_mut(&mut self.data.buffer).ok_or(Error::BufferShared)?;
            func(buffer)
        } else {
            let mut new_buffer = self.buffer_clone()?;
            let buffer = Arc::get_mut(&mut new_buffer).ok_or(Error::BufferShared)?;
            let result = func(buffer)?;
            self.data.buffer = new_buffer;
            Ok(result)
        }
    }

    pub fn layout(&self) -> &Layout {
        &self.metadata.layout
    }

    pub fn layout_mut(&mut self) -> &mut Layout {
        &mut self.metadata.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.metadata.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.metadata.layout.strides()
    }

    pub fn offset(&self) -> usize {
        self.metadata.layout.offset()
    }

    pub fn size(&self) -> usize {
        self.metadata.layout.size()
    }

    pub fn ndim(&self) -> usize {
        self.metadata.layout.ndim()
    }

    pub fn dim_size(&self, dim: usize) -> Option<usize> {
        self.metadata.layout.dim_size(dim)
    }

    pub fn is_contiguous(&self) -> bool {
        self.metadata.layout.is_contiguous()
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
            if Arc::strong_count(grad_mutex) == 1 {
                let grad = unsafe {
                    let raw_ptr = Arc::as_ptr(grad_mutex);
                    &mut *(raw_ptr as *mut Mutex<Tensor>)
                };

                if let Ok(tensor) = grad.get_mut() {
                    tensor.add_(grad_in)?;
                } else {
                    let mut guard = grad_mutex.lock().map_err(|_| Error::GradLocked)?;
                    guard.add_(grad_in)?;
                }
            } else {
                let mut guard = grad_mutex.lock().map_err(|_| Error::GradLocked)?;
                guard.add_(grad_in)?;
            }
        }
        Ok(())
    }

    pub fn zero_grad(&self) -> Result<()> {
        let mut visited = HashSet::<u64>::new();
        self._zero_grad_impl(&mut visited)
    }

    fn _zero_grad_impl(&self, visited: &mut HashSet<u64>) -> Result<()> {
        let ptr = Arc::as_ptr(&self.data.buffer);
        let mut hasher = DefaultHasher::new();
        ptr.hash(&mut hasher);
        let tensor_id = hasher.finish();

        if !visited.insert(tensor_id) {
            return Ok(());
        }

        if let Some(grad_mutex) = &self.data.grad {
            if Arc::strong_count(grad_mutex) == 1 {
                let grad = unsafe {
                    let raw_ptr = Arc::as_ptr(grad_mutex);
                    &mut *(raw_ptr as *mut Mutex<Tensor>)
                };

                if let Ok(tensor) = grad.get_mut() {
                    *tensor = Tensor::zeros_like(&tensor)?;
                } else {
                    let mut guard = grad_mutex.lock().map_err(|_| Error::GradLocked)?;
                    *guard = Tensor::zeros_like(&guard)?;
                }
            } else {
                let mut guard = grad_mutex.lock().map_err(|_| Error::GradLocked)?;
                *guard = Tensor::zeros_like(&guard)?;
            }
        }

        if let Some(node) = &self.node {
            for input in node.inputs() {
                if input.requires_grad() {
                    input._zero_grad_impl(visited)?;
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
        if self.requires_grad() {
            let grad_out = Self::ones_like(self)?;
            let mut visited = std::collections::HashSet::<u64>::new();
            self._backward_with_tracking(&grad_out, &mut visited)?;
        }

        Ok(())
    }

    fn _backward(&self, grad_out: &Tensor) -> Result<()> {
        let mut visited = std::collections::HashSet::<u64>::new();
        self._backward_with_tracking(grad_out, &mut visited)
    }

    fn _backward_with_tracking(&self, grad_out: &Tensor, visited: &mut std::collections::HashSet<u64>) -> Result<()> {
        let ptr = Arc::as_ptr(&self.data.buffer);
        let mut hasher = DefaultHasher::new();
        ptr.hash(&mut hasher);
        let tensor_id = hasher.finish();

        if !visited.insert(tensor_id) {
            return Ok(());
        }

        if let Some(ref node) = self.node {
            node.backward_with_tracking(grad_out, visited)?;
        }

        Ok(())
    }

    // utils

    pub fn any(&self) -> Result<bool> {
        utils::logical::any_true(self)
    }

    pub fn get(&self, indices: &[usize]) -> Result<Scalar> {
        utils::indexing::get_index(self, indices)
    }

    pub fn set(&mut self, indices: &[usize], data: impl Into<Scalar>) -> Result<()> {
        utils::indexing::set_index(self, indices, data)
    }

    pub fn select(&self, dim: impl Into<Scalar>, index: impl Into<Scalar>) -> Result<Self> {
        utils::indexing::select_dim(self, dim, index)
    }

    pub fn item(&self) -> Result<Scalar> {
        if self.size() != 1 {
            return Err(Error::InvalidArgument(format!(
                "item() can only be called on a tensor with a single element, but got tensor with {} elements",
                self.size()
            )));
        }

        let indices = vec![0; self.ndim()];
        crate::utils::indexing::get_index(self, &indices)
    }

    // iterator

    pub fn index_iter(&self) -> Result<TensorIterator> {
        Ok(TensorIterator {
            shape: self.shape().to_vec(),
            current: vec![0; self.ndim()],
            done: self.ndim() == 0,
        })
    }

    // etc

    pub fn device(&self) -> Device {
        self.metadata.device
    }

    pub fn dtype(&self) -> DType {
        self.metadata.dtype
    }

    pub fn requires_grad(&self) -> bool {
        self.metadata.requires_grad
    }
}
