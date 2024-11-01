use crate::error::{CudaError, CudaResult};
use std::{fmt, mem, ptr::NonNull};

#[derive(Clone)]
pub struct CudaBuffer {
    ptr: NonNull<f32>,
    size: usize,
}

impl fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .finish()
    }
}

impl CudaBuffer {
    pub fn new(size: usize) -> CudaResult<Self> {
        let mut ptr = std::ptr::null_mut();
        let result = unsafe { super::cudaMalloc(&mut ptr, size) };
        if result != 0 {
            return Err(CudaError::AllocationFailed);
        }
        Ok(Self {
            ptr: NonNull::new(ptr as *mut f32).unwrap(),
            size,
        })
    }

    pub fn copy_from_host(&mut self, data: &[f32]) -> CudaResult<()> {
        if mem::size_of_val(data) > self.size {
            return Err(CudaError::InvalidValue);
        }
        let result = unsafe {
            super::cudaMemcpy(
                self.ptr.as_ptr() as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                mem::size_of_val(data),
                super::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        if result != 0 {
            return Err(CudaError::MemcpyFailed);
        }
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [f32]) -> CudaResult<()> {
        if mem::size_of_val(data) > self.size {
            return Err(CudaError::InvalidValue);
        }
        let result = unsafe {
            super::cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr.as_ptr() as *const std::ffi::c_void,
                mem::size_of_val(data),
                super::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        if result != 0 {
            return Err(CudaError::MemcpyFailed);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            let result = super::cudaFree(self.ptr.as_ptr() as *mut std::ffi::c_void);
            if result != 0 {
                eprintln!("Warning: cudaFree failed for CudaBuffer during drop");
            }
        }
    }
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}
