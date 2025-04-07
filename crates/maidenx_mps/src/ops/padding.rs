use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_pad_with_constant {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for constant padding and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els_in, num_els_out, and num_dims correctly describe the tensor dimensions
        /// - The metadata buffer contains valid padding configuration
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const c_void,
            inp: *const c_void,
            out: *const c_void,
            pad_value: $ty,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize padding ops: {:?}", e);
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
                    eprintln!("Failed to get Metal buffer for output pointer");
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

                let num_els_in_buffer = device.new_buffer_with_data(
                    &num_els_in as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_els_out_buffer = device.new_buffer_with_data(
                    &num_els_out as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_dims_buffer = device.new_buffer_with_data(
                    &num_dims as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let pad_value_buffer = device.new_buffer_with_data(
                    &pad_value as *const $ty as *const c_void,
                    std::mem::size_of::<$ty>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(5, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(5, None, 0);
                }

                compute_encoder.set_buffer(6, Some(&pad_value_buffer), 0);

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_els_out as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_pad_with_pattern {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for pattern-based padding and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els_in, num_els_out, and num_dims correctly describe the tensor dimensions
        /// - The metadata buffer contains valid padding configuration
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const c_void,
            inp: *const c_void,
            out: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize padding ops: {:?}", e);
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
                    eprintln!("Failed to get Metal buffer for output pointer");
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

                let num_els_in_buffer = device.new_buffer_with_data(
                    &num_els_in as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_els_out_buffer = device.new_buffer_with_data(
                    &num_els_out as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_dims_buffer = device.new_buffer_with_data(
                    &num_dims as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(5, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(5, None, 0);
                }

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_els_out as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_pad_backward {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for padding backward operations and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els_in, num_els_out, and num_dims correctly describe the tensor dimensions
        /// - The metadata buffer contains valid padding configuration
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els_in: usize,
            num_els_out: usize,
            num_dims: usize,
            metadata: *const c_void,
            grad_out: *const c_void,
            grad_in: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize padding backward ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let grad_out_buffer = match get_buffer_from_map(grad_out as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for grad_out pointer");
                    return;
                }
            };

            let grad_in_buffer = match get_buffer_from_map(grad_in as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for grad_in pointer");
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

                // First, fill grad_in with zeros
                let command_queue = device.new_command_queue();
                let zero_buffer = command_queue.new_command_buffer();
                let blit_encoder = zero_buffer.new_blit_command_encoder();
                let buffer_size = (num_els_in * std::mem::size_of::<$ty>()) as u64;
                blit_encoder.fill_buffer(&grad_in_buffer, metal::NSRange::new(0, buffer_size), 0);
                blit_encoder.end_encoding();
                zero_buffer.commit();
                zero_buffer.wait_until_completed();

                compute_encoder.set_buffer(0, Some(&grad_in_buffer), 0);
                compute_encoder.set_buffer(1, Some(&grad_out_buffer), 0);

                let num_els_in_buffer = device.new_buffer_with_data(
                    &num_els_in as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_els_out_buffer = device.new_buffer_with_data(
                    &num_els_out as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let num_dims_buffer = device.new_buffer_with_data(
                    &num_dims as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(5, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(5, None, 0);
                }

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_els_out as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_dummy_pad_with_constant {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for unsupported types in MPS for constant padding operations.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els_in: usize,
            _num_els_out: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _inp: *const c_void,
            _out: *const c_void,
            _pad_value: $ty,
        ) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_dummy_pad_with_pattern {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for unsupported types in MPS for pattern-based padding operations.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els_in: usize,
            _num_els_out: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _inp: *const c_void,
            _out: *const c_void,
        ) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_dummy_pad_backward {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for unsupported types in MPS for padding backward operations.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els_in: usize,
            _num_els_out: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _grad_out: *const c_void,
            _grad_in: *const c_void,
        ) {
            eprintln!("MPS does not support 64-bit {} operations", stringify!($fn_name));
        }
    };
}

macro_rules! implement_padding_ops {
    ($($dtype:ident: $ty:ty),*) => {
        paste::paste! {
            $(
                implement_pad_with_constant!([<metal_pad_with_constant_ $dtype:lower>], stringify!([<pad_with_constant_ $dtype:lower>]), $ty);
                implement_pad_with_pattern!([<metal_pad_with_reflection_ $dtype:lower>], stringify!([<pad_with_reflection_ $dtype:lower>]), $ty);
                implement_pad_with_pattern!([<metal_pad_with_replication_ $dtype:lower>], stringify!([<pad_with_replication_ $dtype:lower>]), $ty);
                implement_pad_backward!([<metal_pad_with_constant_backward_ $dtype:lower>], stringify!([<pad_with_constant_backward_ $dtype:lower>]), $ty);
                implement_pad_backward!([<metal_pad_with_reflection_backward_ $dtype:lower>], stringify!([<pad_with_reflection_backward_ $dtype:lower>]), $ty);
                implement_pad_backward!([<metal_pad_with_replication_backward_ $dtype:lower>], stringify!([<pad_with_replication_backward_ $dtype:lower>]), $ty);
            )*
        }
    };
}

macro_rules! implement_dummy_padding_ops {
    ($($dtype:ident: $ty:ty),*) => {
        paste::paste! {
            $(
                implement_dummy_pad_with_constant!([<metal_pad_with_constant_ $dtype:lower>], $ty);
                implement_dummy_pad_with_pattern!([<metal_pad_with_reflection_ $dtype:lower>], $ty);
                implement_dummy_pad_with_pattern!([<metal_pad_with_replication_ $dtype:lower>], $ty);
                implement_dummy_pad_backward!([<metal_pad_with_constant_backward_ $dtype:lower>], $ty);
                implement_dummy_pad_backward!([<metal_pad_with_reflection_backward_ $dtype:lower>], $ty);
                implement_dummy_pad_backward!([<metal_pad_with_replication_backward_ $dtype:lower>], $ty);
            )*
        }
    };
}

implement_padding_ops! {
    BF16: bf16,
    F16: f16,
    F32: f32,
    U8: u8,
    U16: u16,
    U32: u32,
    I8: i8,
    I16: i16,
    I32: i32
}

implement_dummy_padding_ops! {
    U64: u64,
    I64: i64,
    F64: f64
}
