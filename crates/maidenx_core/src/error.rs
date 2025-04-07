use crate::{device::Device, dtype::DType};
#[cfg(feature = "cuda")]
use maidenx_cuda::cuda_error;
#[cfg(feature = "mps")]
use maidenx_mps::mps_error;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    OutOfMemory,
    DTypeMismatch {
        expected: DType,
        got: DType,
    },
    DeviceMismatch {
        expected: Device,
        got: Device,
    },
    UnsupportedDType,
    InvalidArgument(String),
    InvalidDevice(String),
    IncompatibleShape(String),
    #[cfg(feature = "cuda")]
    CudaError(String),
    #[cfg(feature = "mps")]
    MpsError(String),
    //
    BufferLocked,
    BufferShared,
    GradLocked,
    InvalidShape {
        message: String,
    },
    ShapeMismatch {
        expected: usize,
        got: usize,
        msg: String,
    },
    DimensionMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    DimensionOutOfBounds {
        dim: i32,
        ndim: usize,
    },
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },
    ConversionError(String),
    // serde
    #[cfg(feature = "serde")]
    SerializationError(String),
    #[cfg(feature = "serde")]
    DeserializationError(String),
    //
    Internal {
        message: String,
    },
    External {
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::DTypeMismatch { expected, got } => {
                write!(f, "DType mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::DeviceMismatch { expected, got } => {
                write!(f, "Device mismatch: expected {}, got {}", expected.name(), got.name())
            }
            Self::UnsupportedDType => write!(f, "Unsupported data type"),
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            Self::InvalidDevice(msg) => write!(f, "Invalid device: {}", msg),
            Self::IncompatibleShape(msg) => write!(f, "Incompatible shape: {}", msg),
            #[cfg(feature = "cuda")]
            Self::CudaError(msg) => write!(f, "CUDA error: {}", msg),
            #[cfg(feature = "mps")]
            Self::MpsError(msg) => write!(f, "MPS error: {}", msg),

            Self::BufferLocked => write!(f, "Buffer is locked"),
            Self::BufferShared => write!(f, "Buffer is shared"),
            Self::GradLocked => write!(f, "Grad is locked"),
            Self::InvalidShape { message } => {
                write!(f, "Invalid shape: {}", message)
            }
            Self::ShapeMismatch { expected, got, msg } => {
                write!(f, "Shape mismatch ({}): expected {}, got {}", msg, expected, got)
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::DimensionOutOfBounds { dim, ndim } => {
                write!(
                    f,
                    "Dimension out of bounds: dimension {} is not valid for tensor with {} dimensions",
                    dim, ndim
                )
            }
            Self::IndexOutOfBounds { index, size } => {
                write!(f, "Index out of bounds: index {} is out of bounds for tensor with size {}", index, size)
            }
            Self::ConversionError(msg) => {
                write!(f, "Type conversion error: {}", msg)
            }
            #[cfg(feature = "serde")]
            Self::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            #[cfg(feature = "serde")]
            Self::DeserializationError(msg) => {
                write!(f, "Deserialization error: {}", msg)
            }
            Self::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
            Self::External { message } => {
                write!(f, "External error: {}", message)
            }
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    #[cfg(feature = "cuda")]
    pub fn from_cuda_error(error_code: i32) -> Self {
        if error_code == 2 {
            Self::OutOfMemory
        } else {
            Self::CudaError(cuda_error(error_code))
        }
    }
}

impl Error {
    #[cfg(feature = "mps")]
    pub fn from_mps_error(error_code: i32) -> Self {
        Self::MpsError(mps_error(error_code))
    }
}

