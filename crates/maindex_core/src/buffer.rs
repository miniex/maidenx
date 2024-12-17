use crate::error::CoreResult;
use maidenx_cpu::buffer::CpuBuffer;
#[cfg(feature = "cuda")]
use maidenx_cuda::buffer::CudaBuffer;
use maidenx_device::Device;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Buffer {
    Cpu(Arc<CpuBuffer>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaBuffer>),
}

impl Buffer {
    pub fn new(size: usize, device: &Device) -> CoreResult<Self> {
        match device {
            Device::Cpu => {
                let buffer = CpuBuffer::new(size)?;
                Ok(Buffer::Cpu(Arc::new(buffer)))
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(device_index) => {
                let buffer = CudaBuffer::new(size, *device_index)?;
                Ok(Buffer::Cuda(Arc::new(buffer)))
            }
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        match self {
            Buffer::Cpu(buffer) => buffer.size(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.size(),
        }
    }

    #[inline]
    pub fn device_index(&self) -> usize {
        match self {
            Buffer::Cpu(buffer) => buffer.device_index(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.device_index(),
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        match self {
            Buffer::Cpu(buffer) => buffer.as_ptr(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.as_ptr(),
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self {
            Buffer::Cpu(buffer) => Arc::get_mut(buffer)
                .expect("Failed to get mutable reference to CPU buffer")
                .as_mut_ptr(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => Arc::get_mut(buffer)
                .expect("Failed to get mutable reference to CUDA buffer")
                .as_mut_ptr(),
        }
    }

    pub fn copy_from_host(&mut self, src: &[f32]) -> CoreResult<()> {
        match self {
            Buffer::Cpu(buffer) => Arc::get_mut(buffer)
                .expect("Failed to get mutable reference to CPU buffer")
                .copy_from_host(src)
                .map_err(Into::into),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => Arc::get_mut(buffer)
                .expect("Failed to get mutable reference to CUDA buffer")
                .copy_from_host(src)
                .map_err(Into::into),
        }
    }

    pub fn copy_to_host(&self, dst: &mut [f32]) -> CoreResult<()> {
        match self {
            Buffer::Cpu(buffer) => buffer.copy_to_host(dst).map_err(Into::into),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(buffer) => buffer.copy_to_host(dst).map_err(Into::into),
        }
    }
}
