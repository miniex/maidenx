#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod buffer;
pub mod error;
#[cfg(feature = "nn")]
pub mod nn_activations;
#[cfg(feature = "nn")]
pub mod nn_layers;
pub mod tensor_ops;

use error::{CudaError, CudaResult};
use std::ffi::c_void;
use std::os::raw::c_char;

// Memory copy kinds
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

// Device attributes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeviceAttribute {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MaxSharedMemoryPerBlock = 8,
}

// Device properties
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DeviceProperties {
    pub name: [i8; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub total_const_mem: usize,
    pub major: i32,
    pub minor: i32,
    pub texture_alignment: usize,
    pub texture_pitch_alignment: usize,
    pub compute_mode: i32,
}

// Stream type with Drop implementation
pub struct CudaStream(cudaStream_t);

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Creates a new CUDA stream
    pub fn new() -> CudaResult<Self> {
        create_stream()
    }

    /// Returns the raw stream handle
    pub fn as_raw(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamDestroy(self.0);
        }
    }
}

// Safe wrapper functions
pub fn malloc<T>(count: usize) -> CudaResult<*mut T> {
    let mut ptr = std::ptr::null_mut();
    unsafe {
        check_cuda_error(cudaMalloc(
            &mut ptr as *mut *mut c_void,
            count * std::mem::size_of::<T>(),
        ))?;
    }
    Ok(ptr as *mut T)
}

pub fn free(ptr: std::ptr::NonNull<c_void>) -> CudaResult<()> {
    unsafe { check_cuda_error(cudaFree(ptr.as_ptr())) }
}

pub fn memcpy<T>(dst: *mut T, src: *const T, count: usize, kind: MemcpyKind) -> CudaResult<()> {
    unsafe {
        check_cuda_error(cudaMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            count * std::mem::size_of::<T>(),
            kind,
        ))
    }
}

pub fn memset<T>(ptr: *mut T, value: i32, count: usize) -> CudaResult<()> {
    unsafe {
        check_cuda_error(cudaMemset(
            ptr as *mut c_void,
            value,
            count * std::mem::size_of::<T>(),
        ))
    }
}

pub fn get_device_count() -> CudaResult<i32> {
    let mut count = 0;
    unsafe {
        check_cuda_error(cudaGetDeviceCount(&mut count))?;
    }
    Ok(count)
}

pub fn set_device(device: i32) -> CudaResult<()> {
    unsafe { check_cuda_error(cudaSetDevice(device)) }
}

// TODO: change to get_best_device
pub fn get_device() -> CudaResult<i32> {
    let mut device = 0;
    unsafe {
        check_cuda_error(cudaGetDevice(&mut device))?;
    }
    Ok(device)
}

pub fn is_device_available(device: i32) -> CudaResult<bool> {
    let count = get_device_count()?;

    if device < 0 || device >= count {
        return Ok(false);
    }

    match get_device_properties(device) {
        Ok(_) => match set_device(device) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        },
        Err(_) => Ok(false),
    }
}

pub fn device_synchronize() -> CudaResult<()> {
    unsafe { check_cuda_error(cudaDeviceSynchronize()) }
}

pub fn get_device_properties(device: i32) -> CudaResult<DeviceProperties> {
    let mut prop = std::mem::MaybeUninit::uninit();
    unsafe {
        check_cuda_error(cudaGetDeviceProperties(prop.as_mut_ptr(), device))?;
        Ok(prop.assume_init())
    }
}

pub fn get_device_attribute(attr: DeviceAttribute, device: i32) -> CudaResult<i32> {
    let mut value = 0;
    unsafe {
        check_cuda_error(cudaDeviceGetAttribute(&mut value, attr, device))?;
    }
    Ok(value)
}

pub fn create_stream() -> CudaResult<CudaStream> {
    let mut stream = std::ptr::null_mut();
    unsafe {
        check_cuda_error(cudaStreamCreate(&mut stream))?;
    }
    Ok(CudaStream(stream))
}

pub fn stream_synchronize(stream: &CudaStream) -> CudaResult<()> {
    unsafe { check_cuda_error(cudaStreamSynchronize(stream.0)) }
}

// FFI declarations
#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: MemcpyKind) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetDeviceProperties(prop: *mut DeviceProperties, device: i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: DeviceAttribute, device: i32) -> i32;
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> i32;
    fn cudaStreamDestroy(stream: cudaStream_t) -> i32;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
}

type cudaStream_t = *mut c_void;

const CUDA_SUCCESS: i32 = 0;

fn check_cuda_error(error: i32) -> CudaResult<()> {
    if error == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::from_code(error))
    }
}
