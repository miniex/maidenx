use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_reduction_op {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for reduction operations and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els, num_dims, and num_red_dims correctly describe the buffer dimensions
        /// - The metadata buffer contains valid shape and stride information
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            num_red_dims: usize,
            metadata: *const c_void,
            inp: *const c_void,
            out: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize reduction ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let inp_buffer = match get_buffer_from_map(inp as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for input pointer");
                    return;
                }
            };

            let out_buffer = match get_buffer_from_map(out as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for out pointer");
                    return;
                }
            };

            // Get metadata buffer if provided
            let metadata_buffer = if !metadata.is_null() {
                match get_buffer_from_map(metadata as *mut c_void) {
                    Some(buffer) => Some(buffer),
                    None => {
                        eprintln!("Failed to get Metal buffer for metadata pointer");
                        return;
                    }
                }
            } else {
                None
            };

            let function_name = format!("metal_{}_kernel", $kernel_name);

            let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                let command_buffer = command_queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&out_buffer), 0);
                compute_encoder.set_buffer(1, Some(&inp_buffer), 0);

                let num_els_buffer = device.new_buffer_with_data(
                    &num_els as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_dims_buffer = device.new_buffer_with_data(
                    &num_dims as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_red_dims_buffer = device.new_buffer_with_data(
                    &num_red_dims as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);
                compute_encoder.set_buffer(4, Some(&num_red_dims_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(5, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(5, None, 0);
                }

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_els as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();

                // Using separate wait_until_completed with timeout would be better,
                // but keeping original implementation for compatibility
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_shape_op {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for shape operations and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els and num_dims correctly describe the buffer dimensions
        /// - The metadata buffer contains valid shape and stride information
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(num_els: usize, num_dims: usize, metadata: *const c_void, inp: *const c_void, out: *const c_void) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize shape ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let inp_buffer = match get_buffer_from_map(inp as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for input pointer");
                    return;
                }
            };

            let out_buffer = match get_buffer_from_map(out as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for out pointer");
                    return;
                }
            };

            // Get metadata buffer if provided
            let metadata_buffer = if !metadata.is_null() {
                match get_buffer_from_map(metadata as *mut c_void) {
                    Some(buffer) => Some(buffer),
                    None => {
                        eprintln!("Failed to get Metal buffer for metadata pointer");
                        return;
                    }
                }
            } else {
                None
            };

            let function_name = format!("metal_{}_kernel", $kernel_name);

            let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                let command_buffer = command_queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&out_buffer), 0);
                compute_encoder.set_buffer(1, Some(&inp_buffer), 0);

                let num_els_buffer = device.new_buffer_with_data(
                    &num_els as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_dims_buffer = device.new_buffer_with_data(
                    &num_dims as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);

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

                // Using separate wait_until_completed with timeout would be better,
                // but keeping original implementation for compatibility
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_dummy_reduction_op {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit reduction operations in MPS which are not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _num_dims: usize,
            _num_red_dims: usize,
            _metadata: *const c_void,
            _inp: *const c_void,
            _out: *const c_void,
        ) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_dummy_shape_op {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit shape operations in MPS which are not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(_num_els: usize, _num_dims: usize, _metadata: *const c_void, _inp: *const c_void, _out: *const c_void) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_reduction_ops {
    ($(
        $dtype:ident => {
            type: $ty:ty,
            standard_ops: [$($std_op:ident),*],
            shape_ops: [$($shape_op:ident),*]
        }
    ),*) => {
        paste::paste! {
            $(
                $(
                    implement_reduction_op!([<metal_ $std_op _ $dtype:lower>], stringify!([<$std_op _ $dtype:lower>]), $ty);
                )*
                $(
                    implement_shape_op!([<metal_ $shape_op _ $dtype:lower>], stringify!([<$shape_op _ $dtype:lower>]), $ty);
                )*
            )*
        }
    }
}

macro_rules! implement_dummy_reduction_ops {
    ($(
        $dtype:ident => {
            type: $ty:ty,
            standard_ops: [$($std_op:ident),*],
            shape_ops: [$($shape_op:ident),*]
        }
    ),*) => {
        paste::paste! {
            $(
                $(
                    implement_dummy_reduction_op!([<metal_ $std_op _ $dtype:lower>], $ty);
                )*
                $(
                    implement_dummy_shape_op!([<metal_ $shape_op _ $dtype:lower>], $ty);
                )*
            )*
        }
    }
}

implement_reduction_ops! {
    BF16 => {
        type: bf16,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    F16 => {
        type: f16,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    F32 => {
        type: f32,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    U8 => {
        type: u8,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    U16 => {
        type: u16,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    U32 => {
        type: u32,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I8 => {
        type: i8,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I16 => {
        type: i16,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I32 => {
        type: i32,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    }
}

implement_dummy_reduction_ops! {
    U64 => {
        type: u64,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    I64 => {
        type: i64,
        standard_ops: [sum, max, min],
        shape_ops: [sum_to_shape, fold]
    },
    F64 => {
        type: f64,
        standard_ops: [sum, mean, max, min],
        shape_ops: [sum_to_shape, fold]
    }
}
