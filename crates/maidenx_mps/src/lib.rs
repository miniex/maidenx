use metal::{Device, MTLResourceOptions};
use std::ffi::c_void;
use std::ptr;

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `ptr` must be valid for writes
/// * The allocated memory must be properly aligned for any type
/// * The allocated memory must not overlap with any other allocated object
/// * The size must not be zero
#[no_mangle]
pub unsafe extern "C" fn mps_malloc(ptr: *mut *mut c_void, size: usize) -> i32 {
    if size == 0 {
        return -1; // Invalid size
    }

    let device = match Device::system_default() {
        Some(device) => device,
        None => return -2, // No Metal device available
    };

    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeManaged);
    let contents_ptr = buffer.contents();

    *ptr = contents_ptr;

    0 // Success
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `ptr` must have been allocated by `mps_malloc`
/// * The pointer must not be used after being freed
#[no_mangle]
pub unsafe extern "C" fn mps_free(ptr: *mut c_void) -> i32 {
    if ptr.is_null() {
        return -1; // Invalid pointer
    }

    0 // Success
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `dst` must be a valid Metal device pointer allocated by `mps_malloc`
/// * `src` must be a valid host pointer
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
#[no_mangle]
pub unsafe extern "C" fn mps_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize) -> i32 {
    if dst.is_null() || src.is_null() || size == 0 {
        return -1; // Invalid parameters
    }

    ptr::copy_nonoverlapping(src, dst, size);

    let device = match Device::system_default() {
        Some(device) => device,
        None => return -2, // No Metal device available
    };

    let temp_buffer = device.new_buffer_with_data(dst as *const std::ffi::c_void, size as u64, MTLResourceOptions::StorageModeManaged);
    temp_buffer.did_modify_range(metal::NSRange::new(0, size as u64));

    0 // Success
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `dst` must be a valid host pointer
/// * `src` must be a valid Metal device pointer allocated by `mps_malloc`
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
#[no_mangle]
pub unsafe extern "C" fn mps_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize) -> i32 {
    if dst.is_null() || src.is_null() || size == 0 {
        return -1; // Invalid parameters
    }

    ptr::copy_nonoverlapping(src, dst, size);

    0 // Success
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * Both `dst` and `src` must be valid Metal device pointers allocated by `mps_malloc`
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
#[no_mangle]
pub unsafe extern "C" fn mps_memcpy_d2d(dst: *mut c_void, src: *const c_void, size: usize) -> i32 {
    if dst.is_null() || src.is_null() || size == 0 {
        return -1; // Invalid parameters
    }

    ptr::copy_nonoverlapping(src, dst, size);

    0 // Success
}

pub fn mps_error(error_code: i32) -> String {
    match error_code {
        0 => String::from("Success"),
        -1 => String::from("Invalid parameters"),
        -2 => String::from("No Metal device available"),
        _ => format!("Unknown Metal error: {}", error_code),
    }
}
