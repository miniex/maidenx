use crate::error::{CudaError, CudaErrorCode, CudaResult};

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
    device_index: usize,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    pub fn new(size: usize, device_index: usize) -> CudaResult<Self> {
        if size == 0 {
            return Err(CudaError {
                code: CudaErrorCode::InvalidBufferSize,
                message: "Buffer size must be greater than 0".to_string(),
            });
        }

        crate::set_device(device_index as i32)?;

        let ptr = crate::malloc::<f32>(size)?;

        if ptr.is_null() {
            return Err(CudaError {
                code: CudaErrorCode::MemoryAllocationFailed,
                message: "CUDA malloc returned null".to_string(),
            });
        }

        Ok(Self {
            ptr,
            size,
            device_index,
        })
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
    #[inline]
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    pub fn copy_from_host(&mut self, src: &[f32]) -> CudaResult<()> {
        if src.len() != self.size {
            return Err(CudaError {
                code: CudaErrorCode::HostToDeviceCopyFailed,
                message: "Source size does not match buffer size".to_string(),
            });
        }

        crate::set_device(self.device_index as i32)?;
        crate::memcpy(
            self.ptr,
            src.as_ptr(),
            self.size,
            crate::MemcpyKind::HostToDevice,
        )
    }

    pub fn copy_to_host(&self, dst: &mut [f32]) -> CudaResult<()> {
        if dst.len() != self.size {
            return Err(CudaError {
                code: CudaErrorCode::DeviceToHostCopyFailed,
                message: "Destination size does not match buffer size".to_string(),
            });
        }

        crate::set_device(self.device_index as i32)?;
        crate::memcpy(
            dst.as_mut_ptr(),
            self.ptr,
            self.size,
            crate::MemcpyKind::DeviceToHost,
        )
    }

    /// Sets all elements in the buffer to the specified value
    pub fn fill(&mut self, value: f32) -> CudaResult<()> {
        crate::set_device(self.device_index as i32)?;
        crate::memset(self.ptr, value.to_bits() as i32, self.size)
    }

    /// Synchronizes the device this buffer is allocated on
    pub fn synchronize(&self) -> CudaResult<()> {
        crate::set_device(self.device_index as i32)?;
        crate::device_synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_creation() {
        let buffer = CudaBuffer::new(100, 0).unwrap();
        assert_eq!(buffer.size, 100);
        assert_eq!(buffer.device_index, 0);
    }
}
