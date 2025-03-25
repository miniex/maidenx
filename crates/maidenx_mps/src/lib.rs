use metal::{Device, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;
use std::sync::Once;

// Initialize static variables for buffer management
static INIT: Once = Once::new();
static mut BUFFER_MAP: Option<Mutex<HashMap<*mut c_void, metal::Buffer>>> = None;

// Initialize the buffer map for storing Metal buffers
fn init_buffer_map() {
    unsafe {
        BUFFER_MAP = Some(Mutex::new(HashMap::new()));
    }
}

// Add a buffer to the map with its pointer as key
unsafe fn add_to_buffer_map(ptr: *mut c_void, buffer: metal::Buffer) {
    INIT.call_once(|| {
        init_buffer_map();
    });

    let buffer_map = &raw const BUFFER_MAP;
    if let Some(map) = (*buffer_map).as_ref() {
        let mut map_guard = map.lock().unwrap();
        map_guard.insert(ptr, buffer);
    }
}

// Retrieve a buffer from the map using its pointer
unsafe fn get_buffer_from_map(ptr: *mut c_void) -> Option<metal::Buffer> {
    let buffer_map = &raw const BUFFER_MAP;
    if let Some(map) = (*buffer_map).as_ref() {
        let map_guard = map.lock().unwrap();
        map_guard.get(&ptr).cloned()
    } else {
        None
    }
}

// Remove a buffer from the map
unsafe fn remove_from_buffer_map(ptr: *mut c_void) {
    let buffer_map = &raw const BUFFER_MAP;
    if let Some(map) = (*buffer_map).as_ref() {
        let mut map_guard = map.lock().unwrap();
        map_guard.remove(&ptr);
    }
}

/// Allocates memory on the Metal device
///
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

    add_to_buffer_map(contents_ptr, buffer);

    *ptr = contents_ptr;

    0 // Success
}

/// Frees memory allocated on the Metal device
///
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

    remove_from_buffer_map(ptr);

    0 // Success
}

/// Copies memory from host to device
///
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

    let buffer = match get_buffer_from_map(dst) {
        Some(buffer) => buffer,
        None => return -1,
    };

    buffer.did_modify_range(metal::NSRange::new(0, size as u64));

    0 // Success
}

/// Copies memory from device to host
///
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

    // Copy data
    ptr::copy_nonoverlapping(src, dst, size);

    0 // Success
}

/// Copies memory from device to device
///
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

    let dst_buffer = match get_buffer_from_map(dst) {
        Some(buffer) => buffer,
        None => return -1,
    };

    ptr::copy_nonoverlapping(src, dst, size);

    dst_buffer.did_modify_range(metal::NSRange::new(0, size as u64));

    0 // Success
}

/// Returns a descriptive error message for MPS error codes
pub fn mps_error(error_code: i32) -> String {
    match error_code {
        0 => String::from("Success"),
        -1 => String::from("Invalid parameters"),
        -2 => String::from("No Metal device available"),
        _ => format!("Unknown Metal error: {}", error_code),
    }
}
