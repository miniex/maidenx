use maidenx_core::error::CoreError;
use maidenx_cpu::error::CpuError;
#[cfg(feature = "cuda")]
use maidenx_cuda::error::CudaError;
use maidenx_device::error::DeviceError;
use maidenx_tensor::error::TensorError;

#[derive(Debug)]
pub enum NnError {
    InvalidOperation(String),

    // deps
    Core(CoreError),
    Cpu(CpuError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
    Device(DeviceError),
    Tensor(TensorError),
}

pub type NnResult<T> = Result<T, NnError>;

impl std::fmt::Display for NnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NnError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            NnError::Core(e) => write!(f, "{}", e),
            NnError::Cpu(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            NnError::Cuda(e) => write!(f, "{}", e),
            NnError::Device(e) => write!(f, "{}", e),
            NnError::Tensor(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for NnError {}

impl From<CoreError> for NnError {
    fn from(err: CoreError) -> Self {
        NnError::Core(err)
    }
}

impl From<CpuError> for NnError {
    fn from(err: CpuError) -> Self {
        NnError::Cpu(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for NnError {
    fn from(err: CudaError) -> Self {
        NnError::Cuda(err)
    }
}

impl From<DeviceError> for NnError {
    fn from(err: DeviceError) -> Self {
        NnError::Device(err)
    }
}

impl From<TensorError> for NnError {
    fn from(err: TensorError) -> Self {
        NnError::Tensor(err)
    }
}
