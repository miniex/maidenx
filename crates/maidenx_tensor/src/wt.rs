use crate::Tensor;
use maidenx_core::{
    buffer::BufferManager,
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::{Arc, Mutex};

impl Tensor {
    pub fn with_shape(&mut self, shape: &[usize]) -> Result<()> {
        if self.size() != Layout::compute_size(shape) {
            return Err(Error::InvalidShape {
                message: format!(
                    "Shape mismatch: expected total size {}, but got {} for shape {:?}",
                    self.size(),
                    Layout::compute_size(shape),
                    shape
                ),
            });
        }

        let offset = self.offset();
        self.metadata.layout = Layout::from_shape(shape);
        self.metadata.layout.set_offset(offset);

        Ok(())
    }

    pub fn to_shape(&self, shape: &[usize]) -> Result<Self> {
        let mut tensor = self.clone();
        tensor.with_shape(shape)?;
        Ok(tensor)
    }

    pub fn with_device(&mut self, device: Device) -> Result<()> {
        let cur_device = self.device();
        if cur_device == device {
            return Ok(());
        }

        let buffer_len = self.buffer().len();
        let dtype = self.dtype();

        let mut buffer = BufferManager::create(buffer_len, device, dtype)?;

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_device(self.buffer(), 0, 0, self.buffer().len())?;
        }

        self.data.buffer = buffer;
        self.metadata.device = device;

        Ok(())
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        let mut tensor = self.clone();
        tensor.with_device(device)?;
        Ok(tensor)
    }

    pub fn with_dtype(&mut self, dtype: DType) -> Result<()> {
        let buffer_len = self.buffer().len();
        let device = self.device();

        let mut buffer = BufferManager::create(buffer_len, device, dtype)?;

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(self.buffer(), 0, 0, self.buffer().len())?;
        }

        self.data.buffer = buffer;
        self.metadata.dtype = dtype;

        Ok(())
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        let mut tensor = self.clone();
        tensor.with_dtype(dtype)?;
        Ok(tensor)
    }

    pub fn with_grad(&mut self) -> Result<()> {
        if !self.dtype().is_float() {
            return Err(Error::UnsupportedDType);
        }

        self.metadata.requires_grad = true;
        if self.data.grad.is_none() {
            let grad_storage = Tensor::zeros_like(self)?;
            self.data.grad = Some(Arc::new(Mutex::new(grad_storage)));
        }

        Ok(())
    }
}
