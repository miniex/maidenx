use crate::cudaGetErrorString;
use std::ffi::CStr;

pub type CudaResult<T> = Result<T, CudaError>;

#[derive(Debug, Clone)]
pub struct CudaError {
    pub code: CudaErrorCode,
    pub message: String,
}

impl CudaError {
    pub fn from_code(code: i32) -> Self {
        let message = unsafe {
            let error_str = cudaGetErrorString(code);
            CStr::from_ptr(error_str).to_string_lossy().into_owned()
        };
        Self {
            code: CudaErrorCode::from_i32(code)
                .unwrap_or_else(|| panic!("Unknown CUDA error code: {}", code)),
            message,
        }
    }
}

impl std::error::Error for CudaError {}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA error {}: {}", self.code as i32, self.message)
    }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CudaErrorCode {
    // cuda api or common
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidContext = 201,
    InvalidMemoryAllocation = 400,
    UnmappedMemory = 401,
    ArrayIsMapped = 402,
    AlreadyMapped = 403,
    NoBinaryForGpu = 404,
    AlreadyAcquired = 405,
    NotMapped = 406,
    NotMappedAsArray = 407,
    NotMappedAsPointer = 408,
    MemoryAllocation = 409,
    InvalidDeviceFunction = 98,
    InvalidConfiguration = 9,
    InvalidPitchValue = 12,
    InvalidSymbol = 13,
    LaunchFailure = 719,
    LaunchTimeout = 702,
    LaunchOutOfResources = 701,
    UnsupportedPtxVersion = 137,
    TextureNotBound = 705,
    SynchronizationError = 704,
    InvalidFilterSetting = 706,
    InvalidNormSetting = 707,
    MixedDeviceExecution = 708,
    InitializationError = 127,
    // buffer
    InvalidBufferSize = -1,
    MemoryAllocationFailed = -2,
    HostToDeviceCopyFailed = -3,
    DeviceToHostCopyFailed = -4,
}

impl CudaErrorCode {
    pub fn from_i32(error_code: i32) -> Option<Self> {
        use CudaErrorCode::*;
        match error_code {
            0 => Some(Success),
            1 => Some(InvalidValue),
            2 => Some(OutOfMemory),
            3 => Some(NotInitialized),
            4 => Some(Deinitialized),
            100 => Some(NoDevice),
            101 => Some(InvalidDevice),
            201 => Some(InvalidContext),
            400 => Some(InvalidMemoryAllocation),
            401 => Some(UnmappedMemory),
            402 => Some(ArrayIsMapped),
            403 => Some(AlreadyMapped),
            404 => Some(NoBinaryForGpu),
            405 => Some(AlreadyAcquired),
            406 => Some(NotMapped),
            407 => Some(NotMappedAsArray),
            408 => Some(NotMappedAsPointer),
            409 => Some(MemoryAllocation),
            98 => Some(InvalidDeviceFunction),
            9 => Some(InvalidConfiguration),
            12 => Some(InvalidPitchValue),
            13 => Some(InvalidSymbol),
            719 => Some(LaunchFailure),
            702 => Some(LaunchTimeout),
            701 => Some(LaunchOutOfResources),
            137 => Some(UnsupportedPtxVersion),
            705 => Some(TextureNotBound),
            704 => Some(SynchronizationError),
            706 => Some(InvalidFilterSetting),
            707 => Some(InvalidNormSetting),
            708 => Some(MixedDeviceExecution),
            127 => Some(InitializationError),
            -1 => Some(InvalidBufferSize),
            -2 => Some(MemoryAllocationFailed),
            -3 => Some(HostToDeviceCopyFailed),
            -4 => Some(DeviceToHostCopyFailed),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::InvalidValue => "invalid value",
            Self::OutOfMemory => "out of memory",
            Self::NotInitialized => "not initialized",
            Self::Deinitialized => "deinitialized",
            Self::NoDevice => "no device",
            Self::InvalidDevice => "invalid device",
            Self::InvalidContext => "invalid context",
            Self::InvalidMemoryAllocation => "invalid memory allocation",
            Self::UnmappedMemory => "unmapped memory",
            Self::ArrayIsMapped => "array is mapped",
            Self::AlreadyMapped => "already mapped",
            Self::NoBinaryForGpu => "no binary for gpu",
            Self::AlreadyAcquired => "already acquired",
            Self::NotMapped => "not mapped",
            Self::NotMappedAsArray => "not mapped as array",
            Self::NotMappedAsPointer => "not mapped as pointer",
            Self::MemoryAllocation => "memory allocation",
            Self::InvalidDeviceFunction => "invalid device function",
            Self::InvalidConfiguration => "invalid configuration",
            Self::InvalidPitchValue => "invalid pitch value",
            Self::InvalidSymbol => "invalid symbol",
            Self::LaunchFailure => "launch failure",
            Self::LaunchTimeout => "launch timeout",
            Self::LaunchOutOfResources => "launch out of resources",
            Self::UnsupportedPtxVersion => "unsupported ptx version",
            Self::TextureNotBound => "texture not bound",
            Self::SynchronizationError => "synchronization error",
            Self::InvalidFilterSetting => "invalid filter setting",
            Self::InvalidNormSetting => "invalid norm setting",
            Self::MixedDeviceExecution => "mixed device execution",
            Self::InitializationError => "initialization error",
            // buffer
            Self::InvalidBufferSize => "invalid buffer size",
            Self::MemoryAllocationFailed => "memory allocation failed",
            Self::HostToDeviceCopyFailed => "host-to-device copy failed",
            Self::DeviceToHostCopyFailed => "device-to-host copy failed",
        }
    }
}
