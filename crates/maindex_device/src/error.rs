use maidenx_cpu::error::CpuError;
#[cfg(feature = "cuda")]
use maidenx_cuda::error::CudaError;
use std::fmt;

#[derive(Debug)]
pub enum DeviceError {
    Cpu(CpuError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
}

pub type DeviceResult<T> = Result<T, DeviceError>;

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::Cpu(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            DeviceError::Cuda(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for DeviceError {}

impl From<CpuError> for DeviceError {
    fn from(err: CpuError) -> Self {
        DeviceError::Cpu(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for DeviceError {
    fn from(err: CudaError) -> Self {
        DeviceError::Cuda(err)
    }
}
