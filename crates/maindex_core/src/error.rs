use maidenx_cpu::error::CpuError;
#[cfg(feature = "cuda")]
use maidenx_cuda::error::CudaError;
use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    Cpu(CpuError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
}

pub type CoreResult<T> = Result<T, CoreError>;

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::Cpu(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            CoreError::Cuda(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for CoreError {}

impl From<CpuError> for CoreError {
    fn from(err: CpuError) -> Self {
        CoreError::Cpu(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for CoreError {
    fn from(err: CudaError) -> Self {
        CoreError::Cuda(err)
    }
}
