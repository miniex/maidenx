use crate::error::CoreResult;
use maidenx_cpu::buffer::CpuBuffer;
#[cfg(feature = "cuda")]
use maidenx_cuda::buffer::CudaBuffer;
use maidenx_device::Device;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub enum Buffer {
    Cpu(Arc<RwLock<CpuBuffer>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<RwLock<CudaBuffer>>),
}

impl Buffer {
    pub fn new(size: usize, device: &Device) -> CoreResult<Self> {
        match device {
            Device::Cpu => {
                let buffer = CpuBuffer::new(size)?;
                Ok(Buffer::Cpu(Arc::new(RwLock::new(buffer))))
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(device_index) => {
                let buffer = CudaBuffer::new(size, *device_index)?;
                Ok(Buffer::Cuda(Arc::new(RwLock::new(buffer))))
            }
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        match self {
            Buffer::Cpu(buffer) => buffer.read().unwrap().size(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.read().unwrap().size(),
        }
    }

    #[inline]
    pub fn device_index(&self) -> usize {
        match self {
            Buffer::Cpu(buffer) => buffer.read().unwrap().device_index(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.read().unwrap().device_index(),
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        match self {
            Buffer::Cpu(buffer) => buffer.read().unwrap().as_ptr(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.read().unwrap().as_ptr(),
        }
    }

    #[inline]
    pub fn as_mut_ptr(&self) -> *mut f32 {
        match self {
            Buffer::Cpu(buffer) => buffer.write().unwrap().as_mut_ptr(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.write().unwrap().as_mut_ptr(),
        }
    }

    pub fn copy_from_host(&self, src: &[f32]) -> CoreResult<()> {
        match self {
            Buffer::Cpu(buffer) => buffer
                .write()
                .unwrap()
                .copy_from_host(src)
                .map_err(Into::into),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer
                .write()
                .unwrap()
                .copy_from_host(src)
                .map_err(Into::into),
        }
    }

    pub fn copy_to_host(&self, dst: &mut [f32]) -> CoreResult<()> {
        match self {
            Buffer::Cpu(buffer) => buffer.read().unwrap().copy_to_host(dst).map_err(Into::into),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.read().unwrap().copy_to_host(dst).map_err(Into::into),
        }
    }
}

