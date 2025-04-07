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

    // Helper function to calculate byte offset from element index
    fn byte_offset(&self, element_offset: usize) -> usize {
        element_offset * self.dtype.size_in_bytes()
    }

    // Helper function to get a pointer with the byte offset applied
    unsafe fn ptr_with_offset(&self, element_offset: usize) -> *const c_void {
        (self.ptr as *const u8).add(self.byte_offset(element_offset)) as *const c_void
    }

    // Helper function to get a mutable pointer with the byte offset applied
    unsafe fn mut_ptr_with_offset(&mut self, element_offset: usize) -> *mut c_void {
        (self.ptr as *mut u8).add(self.byte_offset(element_offset)) as *mut c_void
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

    unsafe fn copy_from(&mut self, other: &dyn Buffer, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        if src_offset + count > other.len() || dst_offset + count > self.len() {
            return Err(Error::InvalidArgument("Offset and count exceed buffer dimensions".into()));
        }

        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument("DType mismatch in copy_from".into()));
        }

        let element_size = self.dtype.size_in_bytes();
        let size_in_bytes = count * element_size;

        if cuda_set_device(self.device_id as i32) != 0 {
            return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
        }

        // Get destination pointer with offset
        let dst_ptr = self.mut_ptr_with_offset(dst_offset);

        let status = match other.device() {
            Device::CPU => {
                // Get source pointer with offset for CPU
                let src_ptr = (other.as_ptr() as *const u8).add(src_offset * element_size);
                cuda_memcpy_h2d(dst_ptr, src_ptr as *const c_void, size_in_bytes)
            }
            Device::CUDA(_) => {
                // Get source pointer with offset for CUDA
                let src_ptr = (other.as_ptr() as *const u8).add(src_offset * element_size);
                cuda_memcpy_d2d(dst_ptr, src_ptr as *const c_void, size_in_bytes)
            }
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

    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()> {
        // Check if destination has enough space
        let dst_byte_offset = self.byte_offset(dst_offset);
        let max_available = (self.size * self.dtype.size_in_bytes()).saturating_sub(dst_byte_offset);

        if size_in_bytes > max_available {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_from_host: requested {}, available {}",
                size_in_bytes, max_available
            )));
        }

        if cuda_set_device(self.device_id as i32) != 0 {
            return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
        }

        // Calculate source pointer with offset
        let element_size = self.dtype.size_in_bytes();
        let src_byte_offset = src_offset * element_size;
        let src_ptr = (src as *const u8).add(src_byte_offset) as *const c_void;

        // Get destination pointer with offset
        let dst_ptr = self.mut_ptr_with_offset(dst_offset);

        let status = cuda_memcpy_h2d(dst_ptr, src_ptr, size_in_bytes);
        if status != 0 {
            return Err(Error::InvalidArgument(format!("CUDA H2D memcpy failed: {}", status)));
        }

        Ok(())
    }

    unsafe fn copy_to_host(&self, dest: *mut c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()> {
        // Check if source has enough data
        let src_byte_offset = self.byte_offset(src_offset);
        let available = (self.size * self.dtype.size_in_bytes()).saturating_sub(src_byte_offset);

        if size_in_bytes > available {
            return Err(Error::InvalidArgument(format!(
                "Size mismatch in copy_to_host: requested {}, available {}",
                size_in_bytes, available
            )));
        }

        if cuda_set_device(self.device_id as i32) != 0 {
            return Err(Error::InvalidArgument("Failed to set CUDA device".into()));
        }

        // Calculate destination pointer with offset
        let element_size = self.dtype.size_in_bytes();
        let dst_byte_offset = dst_offset * element_size;
        let dst_ptr = (dest as *mut u8).add(dst_byte_offset) as *mut c_void;

        // Get source pointer with offset
        let src_ptr = self.ptr_with_offset(src_offset);

        let status = cuda_memcpy_d2h(dst_ptr, src_ptr, size_in_bytes);
        if status != 0 {
            return Err(Error::InvalidArgument(format!("CUDA D2H memcpy failed: {}", status)));
        }

        Ok(())
    }
}
