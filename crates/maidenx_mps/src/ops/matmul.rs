use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_matmul {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for matrix multiplication and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els correctly describes the output dimensions
        /// - metadata buffer contains valid shape and stride information for both matrices
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els: usize,
            metadata: *const c_void,
            a: *const c_void,
            b: *const c_void,
            c: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize matmul ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let a_buffer = match get_buffer_from_map(a as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for 'a' pointer");
                    return;
                },
            };

            let b_buffer = match get_buffer_from_map(b as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for 'b' pointer");
                    return;
                },
            };

            let c_buffer = match get_buffer_from_map(c as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for 'c' pointer");
                    return;
                },
            };

            // Get metadata buffer if provided
            let metadata_buffer = if !metadata.is_null() {
                match get_buffer_from_map(metadata as *mut c_void) {
                    Some(buffer) => Some(buffer),
                    None => {
                        eprintln!("Failed to get Metal buffer for metadata pointer");
                        return;
                    },
                }
            } else {
                None
            };

            let function_name = format!("metal_{}_kernel", $kernel_name);

            let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                let command_buffer = command_queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&c_buffer), 0);
                compute_encoder.set_buffer(1, Some(&a_buffer), 0);
                compute_encoder.set_buffer(2, Some(&b_buffer), 0);

                let num_els_buffer = device.new_buffer_with_data(
                    &num_els as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(4, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(4, None, 0);
                }

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_els as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_matmul_backward {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for matrix multiplication backward pass and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els_a and num_els_b correctly describe the gradient dimensions
        /// - metadata buffer contains valid shape and stride information
        /// - All pointers and memory access are valid for the entire operation
        /// - Either grad_a or grad_b may be NULL, in which case that part of the gradient is not computed
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $fn_name(
            num_els_a: usize,
            num_els_b: usize,
            metadata: *const c_void,
            grad_output: *const c_void,
            a: *const c_void,
            b: *const c_void,
            grad_a: *const c_void,
            grad_b: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize matmul backward ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let grad_output_buffer = match get_buffer_from_map(grad_output as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for grad_output pointer");
                    return;
                },
            };

            let a_buffer = match get_buffer_from_map(a as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for 'a' pointer");
                    return;
                },
            };

            let b_buffer = match get_buffer_from_map(b as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for 'b' pointer");
                    return;
                },
            };

            let grad_a_buffer = if !grad_a.is_null() {
                match get_buffer_from_map(grad_a as *mut c_void) {
                    Some(buffer) => Some(buffer),
                    None => {
                        eprintln!("Failed to get Metal buffer for grad_a pointer");
                        return;
                    },
                }
            } else {
                None
            };

            let grad_b_buffer = if !grad_b.is_null() {
                match get_buffer_from_map(grad_b as *mut c_void) {
                    Some(buffer) => Some(buffer),
                    None => {
                        eprintln!("Failed to get Metal buffer for grad_b pointer");
                        return;
                    },
                }
            } else {
                None
            };

            // Get metadata buffer if provided
            let metadata_buffer = if !metadata.is_null() {
                match get_buffer_from_map(metadata as *mut c_void) {
                    Some(buffer) => Some(buffer),
                    None => {
                        eprintln!("Failed to get Metal buffer for metadata pointer");
                        return;
                    },
                }
            } else {
                None
            };

            if let Some(ref grad_a_buf) = grad_a_buffer {
                let function_name = format!("metal_{}_grad_a_kernel", $kernel_name);

                let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                    let command_buffer = command_queue.new_command_buffer();
                    let compute_encoder = command_buffer.new_compute_command_encoder();
                    compute_encoder.set_compute_pipeline_state(&pipeline);

                    compute_encoder.set_buffer(0, Some(grad_a_buf), 0);
                    compute_encoder.set_buffer(1, Some(&grad_output_buffer), 0);
                    compute_encoder.set_buffer(2, Some(&b_buffer), 0);

                    let num_els_buffer = device.new_buffer_with_data(
                        &num_els_a as *const usize as *const c_void,
                        std::mem::size_of::<usize>() as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );

                    compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);

                    if let Some(ref meta_buffer) = metadata_buffer {
                        compute_encoder.set_buffer(4, Some(meta_buffer), 0);
                    } else {
                        compute_encoder.set_buffer(4, None, 0);
                    }

                    let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                    let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                    let num_threadgroups = MTLSize::new((num_els_a as u64).div_ceil(threads_per_threadgroup), 1, 1);
                    compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                    compute_encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    Ok(())
                });
            }

            if let Some(ref grad_b_buf) = grad_b_buffer {
                let function_name = format!("metal_{}_grad_b_kernel", $kernel_name);

                let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                    let command_buffer = command_queue.new_command_buffer();
                    let compute_encoder = command_buffer.new_compute_command_encoder();
                    compute_encoder.set_compute_pipeline_state(&pipeline);

                    compute_encoder.set_buffer(0, Some(grad_b_buf), 0);
                    compute_encoder.set_buffer(1, Some(&grad_output_buffer), 0);
                    compute_encoder.set_buffer(2, Some(&a_buffer), 0);

                    let num_els_buffer = device.new_buffer_with_data(
                        &num_els_b as *const usize as *const c_void,
                        std::mem::size_of::<usize>() as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );

                    compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);

                    if let Some(ref meta_buffer) = metadata_buffer {
                        compute_encoder.set_buffer(4, Some(meta_buffer), 0);
                    } else {
                        compute_encoder.set_buffer(4, None, 0);
                    }

                    let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                    let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                    let num_threadgroups = MTLSize::new((num_els_b as u64).div_ceil(threads_per_threadgroup), 1, 1);
                    compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                    compute_encoder.end_encoding();

                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    Ok(())
                });
            }
        }
    };
}

macro_rules! implement_dummy_matmul {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit matrix multiplication in MPS which is not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _metadata: *const c_void,
            _a: *const c_void,
            _b: *const c_void,
            _c: *const c_void,
        ) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_dummy_matmul_backward {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit matrix multiplication backward pass in MPS which is not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $fn_name(
            _num_els_a: usize,
            _num_els_b: usize,
            _metadata: *const c_void,
            _grad_output: *const c_void,
            _a: *const c_void,
            _b: *const c_void,
            _grad_a: *const c_void,
            _grad_b: *const c_void,
        ) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_matmul_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                implement_matmul!([<metal_matmul_ $dtype>], stringify!([<matmul_ $dtype>]), $ty);
                implement_matmul_backward!([<metal_matmul_backward_ $dtype>], stringify!([<matmul_backward_ $dtype>]), $ty);
            )*
        }
    };
}

macro_rules! implement_dummy_matmul_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                implement_dummy_matmul!([<metal_matmul_ $dtype>], $ty);
                implement_dummy_matmul_backward!([<metal_matmul_backward_ $dtype>], $ty);
            )*
        }
    };
}

implement_matmul_ops! {
    bf16 => bf16,
    f16 => f16,
    f32 => f32,
    u8 => u8,
    u16 => u16,
    u32 => u32,
    i8 => i8,
    i16 => i16,
    i32 => i32
}

implement_dummy_matmul_ops! {
    u64 => u64,
    i64 => i64,
    f64 => f64
}
