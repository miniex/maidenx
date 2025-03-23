use crate::{
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
use maidenx_mps::{mps_free, mps_malloc, mps_memcpy_d2d, mps_memcpy_d2h, mps_memcpy_h2d};
use std::ffi::c_void;

pub struct MpsBuffer {
    ptr: *mut c_void,
    size: usize,
    dtype: DType,
}

unsafe impl Send for MpsBuffer {}
unsafe impl Sync for MpsBuffer {}

impl MpsBuffer {
    pub fn new(size: usize, dtype: DType) -> Result<Self> {
        let total_size = size
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::InvalidArgument("Overflow in allocation".into()))?;

        let mut ptr = std::ptr::null_mut();
        unsafe {
            if mps_malloc(&mut ptr, total_size) != 0 {
                return Err(Error::OutOfMemory);
            }
        }

        Ok(Self { ptr, size, dtype })
    }
}

impl Drop for MpsBuffer {
    fn drop(&mut self) {
        unsafe {
            mps_free(self.ptr);
        }
    }
}

impl Buffer for MpsBuffer {
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
        Device::MPS
    }

    unsafe fn copy_from(&mut self, other: &dyn Buffer) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::InvalidArgument("Buffer size mismatch".into()));
        }
        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument("DType mismatch".into()));
        }

        let size_in_bytes = self.size * self.dtype.size_in_bytes();

        let status = match other.device() {
            Device::CPU => mps_memcpy_h2d(self.ptr, other.as_ptr(), size_in_bytes),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                return Err(Error::InvalidArgument("Direct copy from CUDA to MPS is not supported".into()));
            }
            Device::MPS => mps_memcpy_d2d(self.ptr, other.as_ptr(), size_in_bytes),
        };

        if status != 0 {
            return Err(Error::InvalidArgument(format!("MPS memcpy failed: {}", maidenx_mps::mps_error(status))));
        }

        Ok(())
    }

    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize) -> Result<()> {
        let expected = self.size * self.dtype.size_in_bytes();
        if size_in_bytes != expected {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_from_host: expected {}, got {}",
                expected, size_in_bytes
            )));
        }

        let status = mps_memcpy_h2d(self.ptr, src, size_in_bytes);
        if status != 0 {
            return Err(Error::InvalidArgument(format!(
                "MPS H2D memcpy failed: {}",
                maidenx_mps::mps_error(status)
            )));
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

        let status = mps_memcpy_d2h(dest, self.ptr, size_in_bytes);
        if status != 0 {
            return Err(Error::InvalidArgument(format!(
                "MPS D2H memcpy failed: {}",
                maidenx_mps::mps_error(status)
            )));
        }

        Ok(())
    }
}
