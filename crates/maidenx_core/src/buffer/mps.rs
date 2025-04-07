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

    // Helper function to calculate byte offset from element index
    fn byte_offset(&self, element_offset: usize) -> usize {
        element_offset * self.dtype.size_in_bytes()
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

    unsafe fn copy_from(&mut self, other: &dyn Buffer, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        Ok(())
    }

    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()> {
        Ok(())
    }

    unsafe fn copy_to_host(&self, dest: *mut c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()> {
        Ok(())
    }
}

