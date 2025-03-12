use crate::{
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
use std::{ffi::c_void, ptr};

pub struct CpuBuffer {
    data: Vec<u8>,
    dtype: DType,
}

unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    pub fn new(size: usize, dtype: DType) -> Result<Self> {
        let total_size = size
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::InvalidArgument("Overflow in allocation".into()))?;
        Ok(Self {
            data: vec![0; total_size],
            dtype,
        })
    }
}

impl Buffer for CpuBuffer {
    fn as_ptr(&self) -> *const c_void {
        self.data.as_ptr() as *const _
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.data.as_mut_ptr() as *mut _
    }

    fn len(&self) -> usize {
        self.data.len() / self.dtype.size_in_bytes()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    unsafe fn copy_from(&mut self, other: &dyn Buffer) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::InvalidArgument("Buffer size mismatch".into()));
        }
        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument("DType mismatch".into()));
        }
        ptr::copy_nonoverlapping(other.as_ptr() as *const u8, self.data.as_mut_ptr(), self.data.len());
        Ok(())
    }

    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize) -> Result<()> {
        if size_in_bytes != self.data.len() {
            return Err(Error::InvalidArgument("Size mismatch in copy_from_host".into()));
        }
        ptr::copy_nonoverlapping(src as *const u8, self.data.as_mut_ptr(), self.data.len());
        Ok(())
    }

    unsafe fn copy_to_host(&self, dest: *mut c_void, size_in_bytes: usize) -> Result<()> {
        if size_in_bytes > self.data.len() {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_to_host: requested {}, available {}",
                size_in_bytes,
                self.data.len()
            )));
        }
        ptr::copy_nonoverlapping(self.data.as_ptr(), dest as *mut u8, size_in_bytes);
        Ok(())
    }
}
