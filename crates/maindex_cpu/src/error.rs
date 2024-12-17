use std::error::Error;
use std::fmt;

pub type CpuResult<T> = Result<T, CpuError>;

#[derive(Debug, Clone)]
pub struct CpuError {
    pub code: CpuErrorCode,
    pub message: String,
}

impl CpuError {
    pub fn from_code(code: i32) -> Self {
        let message = CpuErrorCode::message_from_code(code)
            .unwrap_or_else(|| format!("Unknown CPU error code: {}", code));

        Self {
            code: CpuErrorCode::from_i32(code)
                .unwrap_or_else(|| panic!("Unknown CPU error code: {}", code)),
            message,
        }
    }
}

impl Error for CpuError {}

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CPU error {}: {}", self.code as i32, self.message)
    }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CpuErrorCode {
    InvalidBufferSize = -1,
    MemoryAllocationFailed = -2,
    HostToDeviceCopyFailed = -3,
    DeviceToHostCopyFailed = -4,
}

impl CpuErrorCode {
    pub fn from_i32(error_code: i32) -> Option<Self> {
        match error_code {
            -1 => Some(Self::InvalidBufferSize),
            -2 => Some(Self::MemoryAllocationFailed),
            -3 => Some(Self::HostToDeviceCopyFailed),
            -4 => Some(Self::DeviceToHostCopyFailed),
            _ => None,
        }
    }

    pub fn message_from_code(error_code: i32) -> Option<String> {
        match error_code {
            -1 => Some("Invalid buffer size".to_string()),
            -2 => Some("Memory allocation failed".to_string()),
            -3 => Some("Host-to-device copy failed".to_string()),
            -4 => Some("Device-to-host copy failed".to_string()),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidBufferSize => "invalid buffer size",
            Self::MemoryAllocationFailed => "memory allocation failed",
            Self::HostToDeviceCopyFailed => "host-to-device copy failed",
            Self::DeviceToHostCopyFailed => "device-to-host copy failed",
        }
    }
}


