pub mod metal_context;
#[cfg(feature = "nn")]
pub mod nn;
pub mod ops;

pub use metal;

use metal::{Device, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Mutex, Once};

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
/// # Safety
///
/// This function is unsafe because it:
/// * Dereferences raw pointers to access global static state
/// * Assumes the BUFFER_MAP will be properly initialized
/// * The provided pointer must be valid and associated with the given Metal buffer
/// * May cause undefined behavior if called concurrently with other buffer map operations
pub unsafe fn add_to_buffer_map(ptr: *mut c_void, buffer: metal::Buffer) {
    INIT.call_once(|| {
        init_buffer_map();
    });

    let buffer_map = &raw const BUFFER_MAP;
    if let Some(map) = (*buffer_map).as_ref() {
        let mut map_guard = map.lock().unwrap();
        map_guard.insert(ptr, buffer);
    }
}

// Retrieve a buffer from the map using its base pointer
/// # Safety
///
/// This function is unsafe because it:
/// * Dereferences raw pointers to access global static state
/// * Assumes the BUFFER_MAP has been properly initialized
/// * The provided pointer must be valid and previously registered with add_to_buffer_map
/// * Returns None if the pointer is not found in the map
pub unsafe fn get_buffer_from_map(ptr: *mut c_void) -> Option<metal::Buffer> {
    let buffer_map = &raw const BUFFER_MAP;
    if let Some(map) = (*buffer_map).as_ref() {
        let map_guard = map.lock().unwrap();
        map_guard.get(&ptr).cloned()
    } else {
        None
    }
}

// Find the base pointer and offset for a potentially offset pointer
/// # Safety
///
/// This function is unsafe because it:
/// * Dereferences raw pointers to access global static state
/// * Assumes the BUFFER_MAP has been properly initialized
/// * Tries to find the base pointer by checking pointer alignments
pub unsafe fn find_base_ptr_and_offset(ptr: *mut c_void) -> Option<(*mut c_void, usize)> {
    let buffer_map = &raw const BUFFER_MAP;
    if let Some(map) = (*buffer_map).as_ref() {
        let map_guard = map.lock().unwrap();

        // First try to find the exact pointer
        if map_guard.contains_key(&ptr) {
            return Some((ptr, 0));
        }

        // If not found directly, search through all pointers
        for (&base_ptr, buffer) in map_guard.iter() {
            let buffer_size = buffer.length() as usize;
            let ptr_value = ptr as usize;
            let base_value = base_ptr as usize;

            // Check if ptr is within the range of this buffer
            if ptr_value >= base_value && ptr_value < base_value + buffer_size {
                let offset = ptr_value - base_value;
                return Some((base_ptr, offset));
            }
        }
    }

    None
}

// Remove a buffer from the map
/// # Safety
///
/// This function is unsafe because it:
/// * Dereferences raw pointers to access global static state
/// * Assumes the BUFFER_MAP has been properly initialized
/// * The provided pointer must be valid and previously registered with add_to_buffer_map
/// * After removal, the pointer should not be used with Metal operations
pub unsafe fn remove_from_buffer_map(ptr: *mut c_void) {
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

    // Find the base pointer if this is an offset pointer
    let base_ptr = match find_base_ptr_and_offset(ptr) {
        Some((base_ptr, _)) => base_ptr,
        None => return -1, // Buffer not found
    };

    remove_from_buffer_map(base_ptr);

    0 // Success
}

/// Copies memory from host to device with explicit offset support
///
/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `dst` must be a valid Metal device pointer allocated by `mps_malloc`
/// * `src` must be a valid host pointer
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
/// * dst_offset and src_offset must be valid byte offsets
#[no_mangle]
pub unsafe extern "C" fn mps_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize, dst_offset: usize, src_offset: usize) -> i32 {
    if dst.is_null() || src.is_null() || size == 0 {
        return -1; // Invalid parameters
    }

    // Find the base pointer for destination
    let (base_ptr, base_offset) = match find_base_ptr_and_offset(dst) {
        Some(result) => result,
        None => return -1, // Buffer not found
    };

    // Get the Metal buffer
    let buffer = match get_buffer_from_map(base_ptr) {
        Some(buffer) => buffer,
        None => return -1, // Buffer not found
    };

    // Calculate the total destination offset
    let total_dst_offset = base_offset + dst_offset;

    // Calculate the total source offset
    let total_src_offset = src_offset;

    // Calculate the actual destination pointer
    let dst_ptr = (base_ptr as *mut u8).add(total_dst_offset);

    // Calculate the actual source pointer
    let src_ptr = (src as *const u8).add(total_src_offset);

    // Copy the data
    ptr::copy_nonoverlapping(src_ptr as *const c_void, dst_ptr as *mut c_void, size);

    // Notify Metal that we modified the buffer, specifying the range that was modified
    buffer.did_modify_range(metal::NSRange::new(total_dst_offset as u64, size as u64));

    0 // Success
}

/// Copies memory from device to host with explicit offset support
///
/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * `dst` must be a valid host pointer
/// * `src` must be a valid Metal device pointer allocated by `mps_malloc`
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
/// * dst_offset and src_offset must be valid byte offsets
#[no_mangle]
pub unsafe extern "C" fn mps_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize, dst_offset: usize, src_offset: usize) -> i32 {
    if dst.is_null() || src.is_null() || size == 0 {
        return -1; // Invalid parameters
    }

    // Find the base pointer for source
    let (base_ptr, base_offset) = match find_base_ptr_and_offset(src as *mut c_void) {
        Some(result) => result,
        None => return -1, // Buffer not found
    };

    // Get the Metal buffer
    let _buffer = match get_buffer_from_map(base_ptr) {
        Some(buffer) => buffer,
        None => return -1, // Buffer not found
    };

    // Calculate the total source offset
    let total_src_offset = base_offset + src_offset;

    // Calculate the total destination offset
    let total_dst_offset = dst_offset;

    // Calculate the actual source pointer
    let src_ptr = (base_ptr as *const u8).add(total_src_offset);

    // Calculate the actual destination pointer
    let dst_ptr = (dst as *mut u8).add(total_dst_offset);

    // Copy data
    ptr::copy_nonoverlapping(src_ptr as *const c_void, dst_ptr as *mut c_void, size);

    0 // Success
}

/// Copies memory from device to device with explicit offset support
///
/// # Safety
///
/// This function is unsafe because it dereferences raw pointers and requires:
/// * Both `dst` and `src` must be valid Metal device pointers allocated by `mps_malloc`
/// * `size` must not exceed the allocated size of both buffers
/// * The memory regions must not overlap
/// * dst_offset and src_offset must be valid byte offsets
#[no_mangle]
pub unsafe extern "C" fn mps_memcpy_d2d(dst: *mut c_void, src: *const c_void, size: usize, dst_offset: usize, src_offset: usize) -> i32 {
    if dst.is_null() || src.is_null() || size == 0 {
        return -1; // Invalid parameters
    }

    // Find the base pointers
    let (dst_base, dst_base_offset) = match find_base_ptr_and_offset(dst) {
        Some(result) => result,
        None => return -1, // Destination buffer not found
    };

    let (src_base, src_base_offset) = match find_base_ptr_and_offset(src as *mut c_void) {
        Some(result) => result,
        None => return -1, // Source buffer not found
    };

    // Get the Metal buffers
    let dst_buffer = match get_buffer_from_map(dst_base) {
        Some(buffer) => buffer,
        None => return -1, // Destination buffer not found
    };

    let _src_buffer = match get_buffer_from_map(src_base) {
        Some(buffer) => buffer,
        None => return -1, // Source buffer not found
    };

    // Calculate the total destination offset
    let total_dst_offset = dst_base_offset + dst_offset;

    // Calculate the total source offset
    let total_src_offset = src_base_offset + src_offset;

    // Calculate the actual pointers
    let dst_ptr = (dst_base as *mut u8).add(total_dst_offset);
    let src_ptr = (src_base as *const u8).add(total_src_offset);

    // Copy data
    ptr::copy_nonoverlapping(src_ptr as *const c_void, dst_ptr as *mut c_void, size);

    // Notify Metal that we modified the destination buffer
    dst_buffer.did_modify_range(metal::NSRange::new(total_dst_offset as u64, size as u64));

    0 // Success
}

/// Allocates memory on the Metal device and copies dimension array data to it
///
/// # Arguments
///
/// * `dims` - A slice containing the dimensions to be copied to the device
///
/// # Returns
///
/// A tuple containing:
/// * A pointer to the allocated memory on the Metal device
/// * The number of dimensions (length of the dims array)
///
/// # Safety
///
/// This function is unsafe because it:
/// * Dereferences raw pointers
/// * Performs memory allocation on the Metal device
/// * Requires the input slice to be valid for reads
/// * Returns a raw pointer that must be properly managed and eventually freed with `mps_free`
#[no_mangle]
pub unsafe fn mps_alloc_and_copy_dims(dims_and_strides: &[usize]) -> (*mut std::ffi::c_void, usize) {
    let size = std::mem::size_of_val(dims_and_strides);
    if size == 0 {
        return (ptr::null_mut(), 0);
    }

    let mut ptr: *mut c_void = ptr::null_mut();
    let result = mps_malloc(&mut ptr, size);
    if result != 0 || ptr.is_null() {
        return (ptr::null_mut(), 0);
    }

    // Use the updated function signature with explicit offsets
    let result = mps_memcpy_h2d(ptr, dims_and_strides.as_ptr() as *const c_void, size, 0, 0);
    if result != 0 {
        mps_free(ptr);
        return (ptr::null_mut(), 0);
    }

    (ptr, dims_and_strides.len())
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
