use crate::core::error::CoreError;
use crate::cpu::error::CpuError;
#[cfg(feature = "cuda")]
use crate::cuda::error::CudaError;
use crate::device::error::DeviceError;
#[cfg(feature = "nn")]
use crate::nn::error::NnError;
use crate::tensor::error::TensorError;

#[derive(Debug)]
pub enum MaidenxError {
    Core(CoreError),
    Cpu(CpuError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
    Device(DeviceError),
    #[cfg(feature = "nn")]
    Nn(NnError),
    Tensor(TensorError),
}

pub type MaidenxResult<T> = Result<T, MaidenxError>;

impl std::fmt::Display for MaidenxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaidenxError::Core(e) => write!(f, "{}", e),
            MaidenxError::Cpu(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            MaidenxError::Cuda(e) => write!(f, "{}", e),
            MaidenxError::Device(e) => write!(f, "{}", e),
            #[cfg(feature = "nn")]
            MaidenxError::Nn(e) => write!(f, "{}", e),
            MaidenxError::Tensor(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for MaidenxError {}

impl From<CoreError> for MaidenxError {
    fn from(err: CoreError) -> Self {
        MaidenxError::Core(err)
    }
}

impl From<CpuError> for MaidenxError {
    fn from(err: CpuError) -> Self {
        MaidenxError::Cpu(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for MaidenxError {
    fn from(err: CudaError) -> Self {
        MaidenxError::Cuda(err)
    }
}

impl From<DeviceError> for MaidenxError {
    fn from(err: DeviceError) -> Self {
        MaidenxError::Device(err)
    }
}

#[cfg(feature = "nn")]
impl From<NnError> for MaidenxError {
    fn from(err: NnError) -> Self {
        MaidenxError::Nn(err)
    }
}

impl From<TensorError> for MaidenxError {
    fn from(err: TensorError) -> Self {
        MaidenxError::Tensor(err)
    }
}
