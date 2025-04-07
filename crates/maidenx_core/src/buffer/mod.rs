pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "mps")]
pub mod mps;

use crate::{
    device::Device,
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};
use cpu::CpuBuffer;
#[cfg(feature = "cuda")]
use cuda::CudaBuffer;
#[cfg(feature = "mps")]
use mps::MpsBuffer;
use std::{ffi::c_void, sync::Arc};

pub struct BufferManager {}

impl BufferManager {
    pub fn create(size: usize, device: Device, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
            #[cfg(feature = "mps")]
            Device::MPS => Arc::new(MpsBuffer::new(size, dtype)?),
        };

        Ok(buffer)
    }
}

pub trait Buffer: Send + Sync {
    fn as_ptr(&self) -> *const c_void;
    fn as_mut_ptr(&mut self) -> *mut c_void;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;

    /// # Safety
    /// Requires both buffers to have the same size and no memory overlap
    unsafe fn copy_from(&mut self, other: &dyn Buffer, src_offset: usize, dst_offset: usize, count: usize) -> Result<()>;

    /// # Safety
    /// Requires valid source pointer and matching size_in_bytes with no memory overlap
    unsafe fn copy_from_host(&mut self, src: *const c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()>;

    /// # Safety
    /// Requires valid destination pointer and matching size_in_bytes with no memory overlap
    unsafe fn copy_to_host(&self, dest: *mut c_void, size_in_bytes: usize, src_offset: usize, dst_offset: usize) -> Result<()>;

    fn copy_from_with_device(&mut self, other: &dyn Buffer, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        if src_offset + count > other.len() || dst_offset + count > self.len() {
            return Err(Error::InvalidArgument("Offset and count exceed buffer dimensions".into()));
        }

        match (self.device(), other.device()) {
            (Device::CPU, Device::CPU) => unsafe { self.copy_from(other, src_offset, dst_offset, count) },
            #[cfg(feature = "cuda")]
            (Device::CUDA(dst_id), Device::CUDA(src_id)) => {
                if dst_id == src_id {
                    unsafe { self.copy_from(other, src_offset, dst_offset, count) }
                } else {
                    // Try direct copy first, fall back to CPU if needed
                    match unsafe { self.copy_from(other, src_offset, dst_offset, count) } {
                        Ok(_) => Ok(()),
                        Err(_) => {
                            let mut temp = CpuBuffer::new(count, self.dtype())?;
                            unsafe {
                                temp.copy_from(other, src_offset, 0, count)?;
                                self.copy_from(&temp, 0, dst_offset, count)
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "cuda")]
            (Device::CPU, Device::CUDA(_)) | (Device::CUDA(_), Device::CPU) => unsafe { self.copy_from(other, src_offset, dst_offset, count) },

            #[cfg(feature = "mps")]
            (Device::MPS, Device::MPS) => unsafe { self.copy_from(other, src_offset, dst_offset, count) },
            #[cfg(feature = "mps")]
            (Device::CPU, Device::MPS) | (Device::MPS, Device::CPU) => unsafe { self.copy_from(other, src_offset, dst_offset, count) },

            #[cfg(all(feature = "cuda", feature = "mps"))]
            (Device::CUDA(_), Device::MPS) | (Device::MPS, Device::CUDA(_)) => {
                let mut temp = CpuBuffer::new(count, self.dtype())?;
                unsafe {
                    temp.copy_from(other, src_offset, 0, count)?;
                    self.copy_from(&temp, 0, dst_offset, count)
                }
            }
        }
    }

    fn copy_from_with_dtype_cast(&mut self, other: &dyn Buffer, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        if src_offset + count > other.len() || dst_offset + count > self.len() {
            return Err(Error::InvalidArgument("Offset and count exceed buffer dimensions".into()));
        }

        let from_dtype = other.dtype();
        let to_dtype = self.dtype();

        if from_dtype == to_dtype {
            unsafe {
                return self.copy_from(other, src_offset, dst_offset, count);
            }
        }

        // Helper macros for type conversion
        macro_rules! convert_buffer_primitive {
            ($from_ty:ty => $mid_ty:ty => $to_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<$to_ty> = temp_buf.iter().map(|&x| (x as $mid_ty) as $to_ty).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<$to_ty>(),
                        0,
                        dst_offset,
                    )
                }
            }};
            ($from_ty:ty => $to_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<$to_ty> = temp_buf.iter().map(|&x| x as $to_ty).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<$to_ty>(),
                        0,
                        dst_offset,
                    )
                }
            }};
        }

        macro_rules! convert_buffer_from {
            ($from_ty:ty => $to_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<$to_ty> = temp_buf.iter().map(|&x| <$to_ty>::from(x)).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<$to_ty>(),
                        0,
                        dst_offset,
                    )
                }
            }};
        }

        macro_rules! convert_buffer_through_from {
            ($from_ty:ty => $mid_ty:ty => $to_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<$to_ty> = temp_buf.iter().map(|&x| <$to_ty>::from(x as $mid_ty)).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<$to_ty>(),
                        0,
                        dst_offset,
                    )
                }
            }};
        }

        macro_rules! convert_buffer_to_bool {
            ($from_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<bool> = temp_buf.iter().map(|&x| x != <$from_ty>::default()).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<bool>(),
                        0,
                        dst_offset,
                    )
                }
            }};
        }

        macro_rules! convert_buffer_f32_to_half {
            ($from_ty:ty => $to_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<$to_ty> = temp_buf.iter().map(|&x| <$to_ty>::from_f32(x as f32)).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<$to_ty>(),
                        0,
                        dst_offset,
                    )
                }
            }};
        }

        macro_rules! convert_buffer_half_to_numeric {
            // half -> half
            (half::bf16 => half::f16) => {{
                let mut temp_buf: Vec<half::bf16> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<half::f16> = temp_buf.iter().map(|&x| half::f16::from_f32(x.to_f32())).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<half::f16>(),
                        0,
                        dst_offset,
                    )
                }
            }};
            (half::f16 => half::bf16) => {{
                let mut temp_buf: Vec<half::f16> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<half::bf16> = temp_buf.iter().map(|&x| half::bf16::from_f32(x.to_f32())).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<half::bf16>(),
                        0,
                        dst_offset,
                    )
                }
            }};
            // half -> bool
            ($from_ty:ty => bool) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<bool> = temp_buf.iter().map(|&x| <$from_ty>::to_f32(x) != 0.0).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<bool>(),
                        0,
                        dst_offset,
                    )
                }
            }};
            // half -> numeric
            ($from_ty:ty => $to_ty:ty) => {{
                let mut temp_buf: Vec<$from_ty> = Vec::with_capacity(count);
                temp_buf.resize(count, Default::default());
                let size = count * from_dtype.size_in_bytes();
                unsafe {
                    other.copy_to_host(temp_buf.as_mut_ptr() as *mut std::ffi::c_void, size, src_offset, 0)?;
                    let converted: Vec<$to_ty> = temp_buf.iter().map(|&x| <$from_ty>::to_f32(x) as $to_ty).collect();
                    self.copy_from_host(
                        converted.as_ptr() as *const std::ffi::c_void,
                        converted.len() * std::mem::size_of::<$to_ty>(),
                        0,
                        dst_offset,
                    )
                }
            }};
        }

        // Define allowed type conversions
        match (from_dtype, to_dtype) {
            // From BF16
            (DType::BF16, DType::F16) => convert_buffer_half_to_numeric!(half::bf16 => half::f16),
            (DType::BF16, DType::F32) => convert_buffer_from!(half::bf16 => f32),
            (DType::BF16, DType::F64) => convert_buffer_half_to_numeric!(half::bf16 => f64),
            (DType::BF16, DType::BOOL) => convert_buffer_half_to_numeric!(half::bf16 => bool),
            (DType::BF16, DType::U8) => convert_buffer_half_to_numeric!(half::bf16 => u8),
            (DType::BF16, DType::U16) => convert_buffer_half_to_numeric!(half::bf16 => u16),
            (DType::BF16, DType::U32) => convert_buffer_half_to_numeric!(half::bf16 => u32),
            (DType::BF16, DType::U64) => convert_buffer_half_to_numeric!(half::bf16 => u64),
            (DType::BF16, DType::I8) => convert_buffer_half_to_numeric!(half::bf16 => i8),
            (DType::BF16, DType::I16) => convert_buffer_half_to_numeric!(half::bf16 => i16),
            (DType::BF16, DType::I32) => convert_buffer_half_to_numeric!(half::bf16 => i32),
            (DType::BF16, DType::I64) => convert_buffer_half_to_numeric!(half::bf16 => i64),

            // From F16
            (DType::F16, DType::BF16) => convert_buffer_half_to_numeric!(half::f16 => half::bf16),
            (DType::F16, DType::F32) => convert_buffer_from!(half::f16 => f32),
            (DType::F16, DType::F64) => convert_buffer_half_to_numeric!(half::f16 => f64),
            (DType::F16, DType::BOOL) => convert_buffer_half_to_numeric!(half::f16 => bool),
            (DType::F16, DType::U8) => convert_buffer_half_to_numeric!(half::f16 => u8),
            (DType::F16, DType::U16) => convert_buffer_half_to_numeric!(half::f16 => u16),
            (DType::F16, DType::U32) => convert_buffer_half_to_numeric!(half::f16 => u32),
            (DType::F16, DType::U64) => convert_buffer_half_to_numeric!(half::f16 => u64),
            (DType::F16, DType::I8) => convert_buffer_half_to_numeric!(half::f16 => i8),
            (DType::F16, DType::I16) => convert_buffer_half_to_numeric!(half::f16 => i16),
            (DType::F16, DType::I32) => convert_buffer_half_to_numeric!(half::f16 => i32),
            (DType::F16, DType::I64) => convert_buffer_half_to_numeric!(half::f16 => i64),

            // From F32
            (DType::F32, DType::BF16) => convert_buffer_f32_to_half!(f32 => half::bf16),
            (DType::F32, DType::F16) => convert_buffer_f32_to_half!(f32 => half::f16),
            (DType::F32, DType::F64) => convert_buffer_primitive!(f32 => f64),
            (DType::F32, DType::BOOL) => convert_buffer_to_bool!(f32),
            (DType::F32, DType::U8) => convert_buffer_primitive!(f32 => u8),
            (DType::F32, DType::U16) => convert_buffer_primitive!(f32 => u16),
            (DType::F32, DType::U32) => convert_buffer_primitive!(f32 => u32),
            (DType::F32, DType::U64) => convert_buffer_primitive!(f32 => u64),
            (DType::F32, DType::I8) => convert_buffer_primitive!(f32 => i8),
            (DType::F32, DType::I16) => convert_buffer_primitive!(f32 => i16),
            (DType::F32, DType::I32) => convert_buffer_primitive!(f32 => i32),
            (DType::F32, DType::I64) => convert_buffer_primitive!(f32 => i64),

            // From F64
            (DType::F64, DType::BF16) => convert_buffer_f32_to_half!(f64 => half::bf16),
            (DType::F64, DType::F16) => convert_buffer_f32_to_half!(f64 => half::f16),
            (DType::F64, DType::F32) => convert_buffer_primitive!(f64 => f32),
            (DType::F64, DType::BOOL) => convert_buffer_to_bool!(f64),
            (DType::F64, DType::U8) => convert_buffer_primitive!(f64 => u8),
            (DType::F64, DType::U16) => convert_buffer_primitive!(f64 => u16),
            (DType::F64, DType::U32) => convert_buffer_primitive!(f64 => u32),
            (DType::F64, DType::U64) => convert_buffer_primitive!(f64 => u64),
            (DType::F64, DType::I8) => convert_buffer_primitive!(f64 => i8),
            (DType::F64, DType::I16) => convert_buffer_primitive!(f64 => i16),
            (DType::F64, DType::I32) => convert_buffer_primitive!(f64 => i32),
            (DType::F64, DType::I64) => convert_buffer_primitive!(f64 => i64),

            // From BOOL
            (DType::BOOL, DType::BF16) => convert_buffer_through_from!(bool => u8 => half::bf16),
            (DType::BOOL, DType::F16) => convert_buffer_through_from!(bool => u8 => half::f16),
            (DType::BOOL, DType::F32) => convert_buffer_primitive!(bool => u8 => f32),
            (DType::BOOL, DType::F64) => convert_buffer_primitive!(bool => u8 => f64),
            (DType::BOOL, DType::U8) => convert_buffer_primitive!(bool => u8),
            (DType::BOOL, DType::U16) => convert_buffer_primitive!(bool => u16),
            (DType::BOOL, DType::U32) => convert_buffer_primitive!(bool => u32),
            (DType::BOOL, DType::U64) => convert_buffer_primitive!(bool => u64),
            (DType::BOOL, DType::I8) => convert_buffer_primitive!(bool => i8),
            (DType::BOOL, DType::I16) => convert_buffer_primitive!(bool => i16),
            (DType::BOOL, DType::I32) => convert_buffer_primitive!(bool => i32),
            (DType::BOOL, DType::I64) => convert_buffer_primitive!(bool => i64),

            // From U8
            (DType::U8, DType::BF16) => convert_buffer_from!(u8 => half::bf16),
            (DType::U8, DType::F16) => convert_buffer_from!(u8 => half::f16),
            (DType::U8, DType::F32) => convert_buffer_primitive!(u8 => f32),
            (DType::U8, DType::F64) => convert_buffer_primitive!(u8 => f64),
            (DType::U8, DType::BOOL) => convert_buffer_to_bool!(u8),
            (DType::U8, DType::U16) => convert_buffer_primitive!(u8 => u16),
            (DType::U8, DType::U32) => convert_buffer_primitive!(u8 => u32),
            (DType::U8, DType::U64) => convert_buffer_primitive!(u8 => u64),
            (DType::U8, DType::I8) => convert_buffer_primitive!(u8 => i8),
            (DType::U8, DType::I16) => convert_buffer_primitive!(u8 => i16),
            (DType::U8, DType::I32) => convert_buffer_primitive!(u8 => i32),
            (DType::U8, DType::I64) => convert_buffer_primitive!(u8 => i64),

            // From U16
            (DType::U16, DType::BF16) => convert_buffer_through_from!(u16 => u8 => half::bf16),
            (DType::U16, DType::F16) => convert_buffer_through_from!(u16 => u8 => half::f16),
            (DType::U16, DType::F32) => convert_buffer_primitive!(u16 => f32),
            (DType::U16, DType::F64) => convert_buffer_primitive!(u16 => f64),
            (DType::U16, DType::BOOL) => convert_buffer_to_bool!(u16),
            (DType::U16, DType::U8) => convert_buffer_primitive!(u16 => u8),
            (DType::U16, DType::U32) => convert_buffer_primitive!(u16 => u32),
            (DType::U16, DType::U64) => convert_buffer_primitive!(u16 => u64),
            (DType::U16, DType::I8) => convert_buffer_primitive!(u16 => i8),
            (DType::U16, DType::I16) => convert_buffer_primitive!(u16 => i16),
            (DType::U16, DType::I32) => convert_buffer_primitive!(u16 => i32),
            (DType::U16, DType::I64) => convert_buffer_primitive!(u16 => i64),

            // From U32
            (DType::U32, DType::BF16) => convert_buffer_through_from!(u32 => u8 => half::bf16),
            (DType::U32, DType::F16) => convert_buffer_through_from!(u32 => u8 => half::f16),
            (DType::U32, DType::F32) => convert_buffer_primitive!(u32 => f32),
            (DType::U32, DType::F64) => convert_buffer_primitive!(u32 => f64),
            (DType::U32, DType::BOOL) => convert_buffer_to_bool!(u32),
            (DType::U32, DType::U8) => convert_buffer_primitive!(u32 => u8),
            (DType::U32, DType::U16) => convert_buffer_primitive!(u32 => u16),
            (DType::U32, DType::U64) => convert_buffer_primitive!(u32 => u64),
            (DType::U32, DType::I8) => convert_buffer_primitive!(u32 => i8),
            (DType::U32, DType::I16) => convert_buffer_primitive!(u32 => i16),
            (DType::U32, DType::I32) => convert_buffer_primitive!(u32 => i32),
            (DType::U32, DType::I64) => convert_buffer_primitive!(u32 => i64),

            // From U64
            (DType::U64, DType::BF16) => convert_buffer_through_from!(u64 => u8 => half::bf16),
            (DType::U64, DType::F16) => convert_buffer_through_from!(u64 => u8 => half::f16),
            (DType::U64, DType::F32) => convert_buffer_primitive!(u64 => f32),
            (DType::U64, DType::F64) => convert_buffer_primitive!(u64 => f64),
            (DType::U64, DType::BOOL) => convert_buffer_to_bool!(u64),
            (DType::U64, DType::U8) => convert_buffer_primitive!(u64 => u8),
            (DType::U64, DType::U16) => convert_buffer_primitive!(u64 => u16),
            (DType::U64, DType::U32) => convert_buffer_primitive!(u64 => u32),
            (DType::U64, DType::I8) => convert_buffer_primitive!(u64 => i8),
            (DType::U64, DType::I16) => convert_buffer_primitive!(u64 => i16),
            (DType::U64, DType::I32) => convert_buffer_primitive!(u64 => i32),
            (DType::U64, DType::I64) => convert_buffer_primitive!(u64 => i64),

            // From I8
            (DType::I8, DType::BF16) => convert_buffer_from!(i8 => half::bf16),
            (DType::I8, DType::F16) => convert_buffer_from!(i8 => half::f16),
            (DType::I8, DType::F32) => convert_buffer_primitive!(i8 => f32),
            (DType::I8, DType::F64) => convert_buffer_primitive!(i8 => f64),
            (DType::I8, DType::BOOL) => convert_buffer_to_bool!(i8),
            (DType::I8, DType::U8) => convert_buffer_primitive!(i8 => u8),
            (DType::I8, DType::U16) => convert_buffer_primitive!(i8 => u16),
            (DType::I8, DType::U32) => convert_buffer_primitive!(i8 => u32),
            (DType::I8, DType::U64) => convert_buffer_primitive!(i8 => u64),
            (DType::I8, DType::I16) => convert_buffer_primitive!(i8 => i16),
            (DType::I8, DType::I32) => convert_buffer_primitive!(i8 => i32),
            (DType::I8, DType::I64) => convert_buffer_primitive!(i8 => i64),

            // From I16
            (DType::I16, DType::BF16) => convert_buffer_through_from!(i16 => i8 => half::bf16),
            (DType::I16, DType::F16) => convert_buffer_through_from!(i16 => i8 => half::f16),
            (DType::I16, DType::F32) => convert_buffer_primitive!(i16 => f32),
            (DType::I16, DType::F64) => convert_buffer_primitive!(i16 => f64),
            (DType::I16, DType::BOOL) => convert_buffer_to_bool!(i16),
            (DType::I16, DType::U8) => convert_buffer_primitive!(i16 => u8),
            (DType::I16, DType::U16) => convert_buffer_primitive!(i16 => u16),
            (DType::I16, DType::U32) => convert_buffer_primitive!(i16 => u32),
            (DType::I16, DType::U64) => convert_buffer_primitive!(i16 => u64),
            (DType::I16, DType::I8) => convert_buffer_primitive!(i16 => i8),
            (DType::I16, DType::I32) => convert_buffer_primitive!(i16 => i32),
            (DType::I16, DType::I64) => convert_buffer_primitive!(i16 => i64),

            // From I32
            (DType::I32, DType::BF16) => convert_buffer_through_from!(i32 => i8 => half::bf16),
            (DType::I32, DType::F16) => convert_buffer_through_from!(i32 => i8 => half::f16),
            (DType::I32, DType::F32) => convert_buffer_primitive!(i32 => f32),
            (DType::I32, DType::F64) => convert_buffer_primitive!(i32 => f64),
            (DType::I32, DType::BOOL) => convert_buffer_to_bool!(i32),
            (DType::I32, DType::U8) => convert_buffer_primitive!(i32 => u8),
            (DType::I32, DType::U16) => convert_buffer_primitive!(i32 => u16),
            (DType::I32, DType::U32) => convert_buffer_primitive!(i32 => u32),
            (DType::I32, DType::U64) => convert_buffer_primitive!(i32 => u64),
            (DType::I32, DType::I8) => convert_buffer_primitive!(i32 => i8),
            (DType::I32, DType::I16) => convert_buffer_primitive!(i32 => i16),
            (DType::I32, DType::I64) => convert_buffer_primitive!(i32 => i64),

            // From I64
            (DType::I64, DType::BF16) => convert_buffer_through_from!(i64 => i8 => half::bf16),
            (DType::I64, DType::F16) => convert_buffer_through_from!(i64 => i8 => half::f16),
            (DType::I64, DType::F32) => convert_buffer_primitive!(i64 => f32),
            (DType::I64, DType::F64) => convert_buffer_primitive!(i64 => f64),
            (DType::I64, DType::BOOL) => convert_buffer_to_bool!(i64),
            (DType::I64, DType::U8) => convert_buffer_primitive!(i64 => u8),
            (DType::I64, DType::U16) => convert_buffer_primitive!(i64 => u16),
            (DType::I64, DType::U32) => convert_buffer_primitive!(i64 => u32),
            (DType::I64, DType::U64) => convert_buffer_primitive!(i64 => u64),
            (DType::I64, DType::I8) => convert_buffer_primitive!(i64 => i8),
            (DType::I64, DType::I16) => convert_buffer_primitive!(i64 => i16),
            (DType::I64, DType::I32) => convert_buffer_primitive!(i64 => i32),

            _ => Err(Error::InvalidArgument(format!(
                "Unsupported dtype conversion from {:?} to {:?}",
                from_dtype, to_dtype
            ))),
        }
    }

    /// Read a scalar value at the specified index
    fn read_scalar(&self, index: usize) -> Result<Scalar> {
        if index >= self.len() {
            return Err(Error::InvalidArgument(format!("Index out of bounds: {} >= {}", index, self.len())));
        }

        match self.device() {
            Device::CPU => {
                // For CPU buffers, read directly from memory
                let offset = index * self.dtype().size_in_bytes();
                let ptr = unsafe { (self.as_ptr() as *const u8).add(offset) };

                // Use DType's read_scalar method to safely read the value
                Ok(unsafe { self.dtype().read_scalar(ptr) })
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // For CUDA buffers, we need to copy the data to host first
                let dtype_size = self.dtype().size_in_bytes();
                // Only copy the single value we need
                let mut temp_buffer = vec![0u8; dtype_size];

                unsafe {
                    // Copy just the single value from the device
                    self.copy_to_host(
                        temp_buffer.as_mut_ptr() as *mut c_void,
                        dtype_size,
                        index, // src_offset
                        0,     // dst_offset
                    )?;

                    // Read the scalar from the temporary buffer
                    Ok(self.dtype().read_scalar(temp_buffer.as_ptr()))
                }
            }
            #[cfg(feature = "mps")]
            Device::MPS => {
                let dtype_size = self.dtype().size_in_bytes();
                // Only copy the single value we need
                let mut temp_buffer = vec![0u8; dtype_size];

                unsafe {
                    self.copy_to_host(
                        temp_buffer.as_mut_ptr() as *mut c_void,
                        dtype_size,
                        index, // src_offset
                        0,     // dst_offset
                    )?;

                    Ok(self.dtype().read_scalar(temp_buffer.as_ptr()))
                }
            }
        }
    }

    /// Write a scalar value at the specified index
    fn write_scalar(&mut self, index: usize, value: Scalar) -> Result<()> {
        if index >= self.len() {
            return Err(Error::InvalidArgument(format!("Index out of bounds: {} >= {}", index, self.len())));
        }

        match self.device() {
            Device::CPU => {
                // For CPU buffers, write directly to memory
                let offset = index * self.dtype().size_in_bytes();
                let ptr = unsafe { (self.as_mut_ptr() as *mut u8).add(offset) };

                // Use DType's write_scalar method to safely write the value
                unsafe { self.dtype().write_scalar(ptr, value) };
                Ok(())
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // For CUDA buffers, create a temporary buffer for just the single value
                let dtype_size = self.dtype().size_in_bytes();
                let mut temp_buffer = vec![0u8; dtype_size];

                unsafe {
                    // Write the scalar value to the CPU buffer
                    self.dtype().write_scalar(temp_buffer.as_mut_ptr(), value);

                    // Copy just the single value to the device at the specified index
                    self.copy_from_host(
                        temp_buffer.as_ptr() as *const c_void,
                        dtype_size,
                        0,     // src_offset
                        index, // dst_offset
                    )?;

                    Ok(())
                }
            }
            #[cfg(feature = "mps")]
            Device::MPS => {
                let dtype_size = self.dtype().size_in_bytes();
                let mut temp_buffer = vec![0u8; dtype_size];

                unsafe {
                    // Write the scalar value to the CPU buffer
                    self.dtype().write_scalar(temp_buffer.as_mut_ptr(), value);

                    // Copy just the single value to the device at the specified index
                    self.copy_from_host(
                        temp_buffer.as_ptr() as *const c_void,
                        dtype_size,
                        0,     // src_offset
                        index, // dst_offset
                    )?;

                    Ok(())
                }
            }
        }
    }
}
