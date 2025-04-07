use crate::{
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
#[cfg(feature = "cuda")]
use maidenx_cuda::cuda_memcpy_d2h;
#[cfg(feature = "mps")]
use maidenx_mps::mps_memcpy_d2h;
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

    // Helper function to calculate byte offset from element index
    fn byte_offset(&self, element_offset: usize) -> usize {
        element_offset * self.dtype.size_in_bytes()
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

    unsafe fn copy_from(&mut self, other: &dyn Buffer, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        if src_offset + count > other.len() || dst_offset + count > self.len() {
            return Err(Error::InvalidArgument("Offset and count exceed buffer dimensions".into()));
        }

        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument("DType mismatch".into()));
        }

        let element_size = self.dtype().size_in_bytes();
        let byte_count = count * element_size;
        let dst_byte_offset = self.byte_offset(dst_offset);
        let src_byte_offset = src_offset * element_size;

        match other.device() {
            Device::CPU => {
                let src_ptr = self.data.as_mut_ptr().add(dst_byte_offset);
                let dst_ptr = (other.as_ptr() as *const u8).add(src_byte_offset);

                ptr::copy_nonoverlapping(dst_ptr, src_ptr, byte_count);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                let dst_ptr = self.data.as_mut_ptr().add(dst_byte_offset) as *mut c_void;
                let src_ptr = (other.as_ptr() as *const u8).add(src_offset * element_size) as *const c_void;

                cuda_memcpy_d2h(dst_ptr, src_ptr, byte_count);
                Ok(())
            }
            #[cfg(feature = "mps")]
            Device::MPS => {
                let dst_ptr = self.data.as_mut_ptr().add(dst_byte_offset) as *mut c_void;

                // Use base pointer and pass explicit offsets
                mps_memcpy_d2h(dst_ptr, other.as_ptr(), byte_count, 0, src_byte_offset);
                Ok(())
            }
        }
    }

    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()> {
        let dst_byte_offset = self.byte_offset(dst_offset);
        let element_size = self.dtype().size_in_bytes();
        let src_byte_offset = src_offset * element_size;

        // Calculate total available space in destination
        let available_bytes = self.data.len() - dst_byte_offset;

        if size_in_bytes > available_bytes {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_from_host: requested {}, available {}",
                size_in_bytes, available_bytes
            )));
        }

        let src_ptr = (src as *const u8).add(src_byte_offset);
        let dst_ptr = self.data.as_mut_ptr().add(dst_byte_offset);

        ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_in_bytes);
        Ok(())
    }

    unsafe fn copy_to_host(&self, dest: *mut c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()> {
        let src_byte_offset = self.byte_offset(src_offset);
        let element_size = self.dtype().size_in_bytes();
        let dst_byte_offset = dst_offset * element_size;

        // Calculate available bytes from the source offset
        let available_bytes = self.data.len() - src_byte_offset;

        if size_in_bytes > available_bytes {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_to_host: requested {}, available {}",
                size_in_bytes, available_bytes
            )));
        }

        let src_ptr = self.data.as_ptr().add(src_byte_offset);
        let dst_ptr = (dest as *mut u8).add(dst_byte_offset);

        ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_in_bytes);
        Ok(())
    }
}
