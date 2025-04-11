use std::sync::Arc;

use crate::{Tensor, TensorData, TensorMetadata};
use maidenx_core::{
    buffer::BufferManager,
    device::Device,
    error::{Error, Result},
};

impl Tensor {
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let mut result = Self::empty_like(self)?;

        match self.device() {
            Device::CPU => {
                for indices in self.index_iter()? {
                    let value = self.get(&indices)?;
                    result.set(&indices, value)?;
                }
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let temp = self.to_device(Device::CPU)?;
                let contiguous_temp = temp.contiguous()?;
                result = contiguous_temp.to_device(Device::CUDA(device_id))?;
            },
            #[cfg(feature = "mps")]
            Device::MPS => {
                let temp = self.to_device(Device::CPU)?;
                let contiguous_temp = temp.contiguous()?;
                result = contiguous_temp.to_device(Device::MPS)?;
            },
        }

        Ok(result)
    }

    pub fn copy(&self) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype();
        let layout = self.layout().clone();

        let mut buffer = BufferManager::create(self.buffer().len(), device, dtype)?;

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            unsafe {
                buffer_mut.copy_from(self.buffer(), 0, 0, self.buffer().len())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: false,
            },
            node: None,
        })
    }

    pub fn detach(&self) -> Result<Self> {
        let mut result = self.clone();
        result.metadata.requires_grad = false;
        result.node = None;

        Ok(result)
    }
}
