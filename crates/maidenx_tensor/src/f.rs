use crate::Tensor;
use maidenx_core::{device::Device, error::Result};

impl Tensor {
    pub fn is_contiguous(&self) -> bool {
        if self.ndim() == 0 {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.ndim()).rev() {
            if self.strides()[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape()[i];
        }

        true
    }

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
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let temp = self.to_device(Device::CPU)?;
                let contiguous_temp = temp.contiguous()?;
                result = contiguous_temp.to_device(Device::CUDA(device_id))?;
            }
        }

        Ok(result)
    }

    pub fn detach(&self) -> Result<Self> {
        let mut result = self.clone();
        result.metadata.requires_grad = false;
        result.node = None;

        Ok(result)
    }
}
