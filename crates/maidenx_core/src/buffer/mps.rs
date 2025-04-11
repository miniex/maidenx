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

    unsafe fn copy_from(
        &mut self,
        other: &dyn Buffer,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        if src_offset + count > other.len() || dst_offset + count > self.len() {
            return Err(Error::InvalidArgument(
                "Offset and count exceed buffer dimensions".into(),
            ));
        }

        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument("DType mismatch".into()));
        }

        let element_size = self.dtype.size_in_bytes();
        let size_in_bytes = count * element_size;
        let dst_byte_offset = self.byte_offset(dst_offset);
        let src_byte_offset = src_offset * element_size;

        let status = match other.device() {
            Device::CPU => {
                let src_ptr = other.as_ptr();
                let element_size = self.dtype.size_in_bytes();
                let src_byte_offset = src_offset * element_size;

                mps_memcpy_h2d(self.ptr, src_ptr, size_in_bytes, dst_byte_offset, src_byte_offset)
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                return Err(Error::InvalidArgument(
                    "Direct copy from CUDA to MPS is not supported".into(),
                ));
            },
            Device::MPS => {
                // For MPS source, use the base pointer
                let src_ptr = other.as_ptr();

                // Let the MPS implementation handle offsets
                mps_memcpy_d2d(self.ptr, src_ptr, size_in_bytes, dst_byte_offset, src_byte_offset)
            },
        };

        if status != 0 {
            return Err(Error::InvalidArgument(format!(
                "MPS memcpy failed: {}",
                maidenx_mps::mps_error(status)
            )));
        }

        Ok(())
    }

    unsafe fn copy_from_host(
        &mut self,
        src: *const c_void,
        size_in_bytes: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        // Check if destination has enough space
        let dst_byte_offset = self.byte_offset(dst_offset);
        let max_available = (self.size * self.dtype.size_in_bytes()).saturating_sub(dst_byte_offset);

        if size_in_bytes > max_available {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_from_host: requested {}, available {}",
                size_in_bytes, max_available
            )));
        }

        // Calculate source byte offset
        let element_size = self.dtype.size_in_bytes();
        let src_byte_offset = src_offset * element_size;

        // Use the base pointers and let MPS handle the offsets
        let status = mps_memcpy_h2d(self.ptr, src, size_in_bytes, dst_byte_offset, src_byte_offset);
        if status != 0 {
            return Err(Error::InvalidArgument(format!(
                "MPS H2D memcpy failed: {}",
                maidenx_mps::mps_error(status)
            )));
        }

        Ok(())
    }

    unsafe fn copy_to_host(
        &self,
        dest: *mut c_void,
        size_in_bytes: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        // Check if source has enough data
        let src_byte_offset = self.byte_offset(src_offset);
        let available = (self.size * self.dtype.size_in_bytes()).saturating_sub(src_byte_offset);

        if size_in_bytes > available {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_to_host: requested {}, available {}",
                size_in_bytes, available
            )));
        }

        // Calculate destination byte offset
        let element_size = self.dtype.size_in_bytes();
        let dst_byte_offset = dst_offset * element_size;

        // Use the base pointers and let MPS handle the offsets
        let status = mps_memcpy_d2h(dest, self.ptr, size_in_bytes, dst_byte_offset, src_byte_offset);
        if status != 0 {
            return Err(Error::InvalidArgument(format!(
                "MPS D2H memcpy failed: {}",
                maidenx_mps::mps_error(status)
            )));
        }

        Ok(())
    }
}
