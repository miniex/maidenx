use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_softmax {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for softmax operations and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_els, num_dims, and dim correctly describe the tensor dimensions
        /// - If metadata is provided, it contains valid shape information for the tensor
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            dim: usize,
            metadata: *const c_void,
            input: *const c_void,
            output: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize softmax ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let input_buffer = match get_buffer_from_map(input as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for input pointer");
                    return;
                },
            };

            let output_buffer = match get_buffer_from_map(output as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for output pointer");
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

                compute_encoder.set_buffer(0, Some(&input_buffer), 0);
                compute_encoder.set_buffer(1, Some(&output_buffer), 0);

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

                let dim_buffer = device.new_buffer_with_data(
                    &dim as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);
                compute_encoder.set_buffer(4, Some(&dim_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(5, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(5, None, 0);
                }

                // Calculate work distribution
                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);

                // For softmax, calculate total slices
                let (pre_dim_size, post_dim_size) = match &metadata_buffer {
                    Some(meta_buf) => {
                        let meta_ptr = meta_buf.contents() as *const usize;
                        let actual_dim = if dim >= num_dims { num_dims - 1 } else { dim };

                        let mut pre = 1;
                        for i in 0..actual_dim {
                            pre *= *meta_ptr.add(i);
                        }

                        let mut post = 1;
                        for i in (actual_dim + 1)..num_dims {
                            post *= *meta_ptr.add(i);
                        }

                        (pre, post)
                    },
                    None => (1, 1),
                };

                let total_slices = pre_dim_size * post_dim_size;
                let num_threadgroups = MTLSize::new((total_slices as u64).div_ceil(threads_per_threadgroup), 1, 1);

                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_dummy_softmax {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for unsupported types in MPS for softmax operations.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _num_dims: usize,
            _dim: usize,
            _metadata: *const c_void,
            _input: *const c_void,
            _output: *const c_void,
        ) {
            eprintln!("MPS does not support {} for softmax operations", stringify!($ty));
        }
    };
}

macro_rules! implement_softmax_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                implement_softmax!([<metal_softmax_ $dtype>], stringify!([<softmax_ $dtype>]), $ty);
            )*
        }
    };
}

macro_rules! implement_dummy_softmax_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                implement_dummy_softmax!([<metal_softmax_ $dtype>], $ty);
            )*
        }
    };
}

// Implement for supported types
implement_softmax_ops! {
    f32 => f32,
    f16 => f16,
    bf16 => bf16
}

// Implement dummy functions for unsupported types
implement_dummy_softmax_ops! {
    f64 => f64
}
