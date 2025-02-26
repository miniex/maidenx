#![allow(non_snake_case)]
pub mod nn;
pub mod ops;

use std::ffi::{c_char, CStr};

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn cudaPointerGetAttributes(attributes: *mut CudaPointerAttributes, ptr: *const std::ffi::c_void) -> i32;
}

#[repr(C)]
pub struct CudaPointerAttributes {
    pub type_: i32,
    pub device: i32,
    pub devicePointer: *mut std::ffi::c_void,
    pub hostPointer: *mut std::ffi::c_void,
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `ptr` must be valid for writes
/// * The allocated memory must be properly aligned for any type
/// * The allocated memory must not overlap with any other allocated object
/// * The size must not be zero
#[no_mangle]
pub unsafe extern "C" fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32 {
    cudaMalloc(ptr, size)
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `ptr` must have been allocated by `cuda_malloc`
/// * The pointer must not be used after being freed
#[no_mangle]
pub unsafe extern "C" fn cuda_free(ptr: *mut std::ffi::c_void) -> i32 {
    cudaFree(ptr)
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `dst` must be a valid CUDA device pointer allocated by `cuda_malloc`
/// * `src` must be a valid host pointer
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
#[no_mangle]
pub unsafe extern "C" fn cuda_memcpy_h2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize) -> i32 {
    cudaMemcpy(dst, src, size, 1) // cudaMemcpyHostToDevice = 1
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `dst` must be a valid host pointer
/// * `src` must be a valid CUDA device pointer allocated by `cuda_malloc`
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
#[no_mangle]
pub unsafe extern "C" fn cuda_memcpy_d2h(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize) -> i32 {
    cudaMemcpy(dst, src, size, 2) // cudaMemcpyDeviceToHost = 2
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * Both `dst` and `src` must be valid CUDA device pointers allocated by `cuda_malloc`
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
#[no_mangle]
pub unsafe extern "C" fn cuda_memcpy_d2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize) -> i32 {
    cudaMemcpy(dst, src, size, 3) // cudaMemcpyDeviceToDevice = 3
}

/// # Safety
///
/// This function converts a CUDA error code into a human-readable string
pub fn cuda_error(error_code: i32) -> String {
    unsafe {
        let c_str = cudaGetErrorString(error_code);
        if c_str.is_null() {
            format!("Unknown CUDA error: {}", error_code)
        } else {
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }
}

/// # Safety
///
/// This function sets the active CUDA device.
/// - `device_id` must be a valid device index.
#[no_mangle]
pub unsafe extern "C" fn cuda_set_device(device_id: i32) -> i32 {
    cudaSetDevice(device_id)
}

/// # Safety
///
/// This function gets the current active CUDA device.
#[no_mangle]
pub unsafe extern "C" fn cuda_get_device(device_id: *mut i32) -> i32 {
    cudaGetDevice(device_id)
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `stream` must be valid for writes
/// * The stream must be properly destroyed using `cuda_stream_destroy`
#[no_mangle]
pub unsafe extern "C" fn cuda_stream_create(stream: *mut *mut std::ffi::c_void) -> i32 {
    cudaStreamCreate(stream)
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `stream` must have been created by `cuda_stream_create`
/// * The stream must not be used after being destroyed
#[no_mangle]
pub unsafe extern "C" fn cuda_stream_destroy(stream: *mut std::ffi::c_void) -> i32 {
    cudaStreamDestroy(stream)
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `stream` must be a valid CUDA stream created by `cuda_stream_create`
#[no_mangle]
pub unsafe extern "C" fn cuda_stream_synchronize(stream: *mut std::ffi::c_void) -> i32 {
    cudaStreamSynchronize(stream)
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `attributes` must be valid for writes and properly aligned
/// * `ptr` must be either a valid CUDA device pointer or host pointer
/// * The memory referenced by `attributes` must be properly sized to hold CudaPointerAttributes
///
/// Returns:
/// * `0` if successful
/// * CUDA error code if the pointer type cannot be determined or if any other error occurs
#[no_mangle]
pub unsafe extern "C" fn cuda_pointer_get_attributes(attributes: *mut CudaPointerAttributes, ptr: *const std::ffi::c_void) -> i32 {
    cudaPointerGetAttributes(attributes, ptr)
}

/// # Safety
///
/// This function is unsafe because it performs CUDA device memory operations and requires:
/// * `dims_and_strides` must be a valid slice with properly aligned data
/// * The size of the input slice must not be zero
/// * The caller must ensure the returned device memory pointer is properly freed using `cuda_free`
/// * The caller must ensure synchronization if the device memory is used in CUDA operations
///
/// Returns:
/// * A tuple containing the device memory pointer and its size in bytes
/// * Will panic if CUDA memory allocation or copy operations fail
pub unsafe fn cuda_alloc_and_copy_dims(dims_and_strides: &[usize]) -> (*mut std::ffi::c_void, usize) {
    let size = std::mem::size_of_val(dims_and_strides);
    let mut device_ptr = std::ptr::null_mut();

    let status = cuda_malloc(&mut device_ptr, size);
    if status != 0 {
        panic!("CUDA malloc failed");
    }

    let status = cuda_memcpy_h2d(device_ptr, dims_and_strides.as_ptr() as *const std::ffi::c_void, size);
    if status != 0 {
        cuda_free(device_ptr);
        panic!("CUDA memcpy failed");
    }

    (device_ptr, size)
}
