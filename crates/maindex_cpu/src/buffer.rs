use crate::error::{CpuError, CpuErrorCode, CpuResult};

#[derive(Debug, Clone)]
pub struct CpuBuffer {
    data: Vec<f32>,
    size: usize,
    device_index: usize,
}

unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    pub fn new(size: usize) -> CpuResult<Self> {
        if size == 0 {
            return Err(CpuError {
                code: CpuErrorCode::InvalidBufferSize,
                message: "Buffer size must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            data: vec![0.0; size],
            size,
            device_index: 0,
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
        self.data.as_ptr()
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    pub fn copy_from_host(&mut self, src: &[f32]) -> CpuResult<()> {
        if src.len() > self.size {
            return Err(CpuError {
                code: CpuErrorCode::HostToDeviceCopyFailed,
                message: "Source size does not match buffer size".to_string(),
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.data.as_mut_ptr(), src.len());
        }

        Ok(())
    }

    pub fn copy_to_host(&self, dst: &mut [f32]) -> CpuResult<()> {
        if dst.len() > self.size {
            return Err(CpuError {
                code: CpuErrorCode::DeviceToHostCopyFailed,
                message: "Destination size does not match buffer size".to_string(),
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(self.data.as_ptr(), dst.as_mut_ptr(), dst.len());
        }

        Ok(())
    }
}
