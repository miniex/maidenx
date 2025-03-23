use crate::{
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
use maidenx_cuda::{cuda_free, cuda_malloc, cuda_memcpy_d2d, cuda_memcpy_d2h, cuda_memcpy_h2d, cuda_set_device};
use std::ffi::c_void;

pub struct CudaBuffer {
    ptr: *mut c_void,
    size: usize,
    dtype: DType,
    device_id: usize,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    pub fn new(size: usize, dtype: DType, device_id: usize) -> Result<Self> {
        let total_size = size
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::InvalidArgument("Overflow in allocation".into()))?;
        let mut ptr = std::ptr::null_mut();
        unsafe {
            if cuda_set_device(device_id as i32) != 0 {
                return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
            }
            if cuda_malloc(&mut ptr, total_size) != 0 {
                return Err(Error::OutOfMemory);
            }
        }
        Ok(Self { ptr, size, dtype, device_id })
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            if cuda_set_device(self.device_id as i32) == 0 {
                cuda_free(self.ptr);
            }
        }
    }
}

impl Buffer for CudaBuffer {
    fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    fn len(&self) -> usize {
        self.size
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::CUDA(self.device_id)
    }

    unsafe fn copy_from(&mut self, other: &dyn Buffer) -> Result<()> {
        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument("DType mismatch in copy_from".into()));
        }
        let size_in_bytes = self.size * self.dtype.size_in_bytes();
        if cuda_set_device(self.device_id as i32) != 0 {
            return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
        }
        let status = match other.device() {
            Device::CPU => cuda_memcpy_h2d(self.ptr, other.as_ptr(), size_in_bytes),
            Device::CUDA(_) => cuda_memcpy_d2d(self.ptr, other.as_ptr(), size_in_bytes),
            #[cfg(feature = "mps")]
            Device::MPS => {
                return Err(Error::InvalidArgument("Direct copy from MPS to CUDA is not supported".into()));
            }
        };
        if status != 0 {
            return Err(Error::InvalidArgument(format!("CUDA memcpy failed: {}", status)));
        }
        Ok(())
    }

    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize) -> Result<()> {
        let expected = self.size * self.dtype.size_in_bytes();
        if size_in_bytes != expected {
            return Err(Error::InvalidArgument("Size mismatch in copy_from_host".into()));
        }
        if cuda_set_device(self.device_id as i32) != 0 {
            return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
        }
        let status = cuda_memcpy_h2d(self.ptr, src, size_in_bytes);
        if status != 0 {
            return Err(Error::InvalidArgument(format!("CUDA H2D memcpy failed: {}", status)));
        }
        Ok(())
    }

    unsafe fn copy_to_host(&self, dest: *mut c_void, size_in_bytes: usize) -> Result<()> {
        let available = self.size * self.dtype.size_in_bytes();
        if size_in_bytes > available {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_to_host: requested {}, available {}",
                size_in_bytes, available
            )));
        }
        if cuda_set_device(self.device_id as i32) != 0 {
            return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
        }
        let status = cuda_memcpy_d2h(dest, self.ptr, size_in_bytes);
        if status != 0 {
            return Err(Error::InvalidArgument(format!("CUDA D2H memcpy failed: {}", status)));
        }
        Ok(())
    }
}
