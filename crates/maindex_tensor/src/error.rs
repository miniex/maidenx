use maidenx_core::error::CoreError;
use maidenx_cpu::error::CpuError;
#[cfg(feature = "cuda")]
use maidenx_cuda::error::CudaError;
use maidenx_device::error::DeviceError;

#[derive(Debug)]
pub enum TensorError {
    // Shape related errors
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    DeviceMismatch {
        expected: String,
        got: String,
    },
    InvalidShape {
        reason: String,
    },
    DimensionMismatch {
        expected_dims: usize,
        got_dims: usize,
    },
    GradientError {
        reason: String,
    },
    OperationError {
        reason: String,
    },
    EmptyTensor,

    // deps
    Core(CoreError),
    Cpu(CpuError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
    Device(DeviceError),
}

pub type TensorResult<T> = Result<T, TensorError>;

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, but got {:?}",
                    expected, got
                )
            }
            TensorError::DeviceMismatch { expected, got } => {
                write!(f, "Device mismatch: expected {}, but got {}", expected, got)
            }
            TensorError::InvalidShape { reason } => {
                write!(f, "Invalid shape: {}", reason)
            }
            TensorError::DimensionMismatch {
                expected_dims,
                got_dims,
            } => {
                write!(
                    f,
                    "Dimension mismatch: expected {} dims, got {} dims",
                    expected_dims, got_dims
                )
            }
            TensorError::GradientError { reason } => {
                write!(f, "Gradient error: {}", reason)
            }
            TensorError::OperationError { reason } => {
                write!(f, "Operation error: {}", reason)
            }
            TensorError::EmptyTensor => {
                write!(f, "Operation cannot be performed on empty tensor")
            }
            TensorError::Core(e) => write!(f, "{}", e),
            TensorError::Cpu(e) => write!(f, "{}", e),
            #[cfg(feature = "cuda")]
            TensorError::Cuda(e) => write!(f, "{}", e),
            TensorError::Device(e) => write!(f, "{}", e),
        }
    }
}

impl From<CoreError> for TensorError {
    fn from(err: CoreError) -> Self {
        TensorError::Core(err)
    }
}

impl From<CpuError> for TensorError {
    fn from(err: CpuError) -> Self {
        TensorError::Cpu(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for TensorError {
    fn from(err: CudaError) -> Self {
        TensorError::Cuda(err)
    }
}

impl From<DeviceError> for TensorError {
    fn from(err: DeviceError) -> Self {
        TensorError::Device(err)
    }
}
