use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_conv2d_im2col {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for im2col convolution operations and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_col_elements correctly describes the column buffer dimensions
        /// - The metadata buffer contains valid shape, stride, padding, and kernel dimensions
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_col_elements: usize,
            metadata: *const c_void,
            input: *const c_void,
            col: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize conv2d ops: {:?}", e);
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

            let col_buffer = match get_buffer_from_map(col as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for col pointer");
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
                compute_encoder.set_buffer(1, Some(&col_buffer), 0);

                let num_els_buffer = device.new_buffer_with_data(
                    &num_col_elements as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(3, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(3, None, 0);
                }

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_col_elements as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_conv2d_col2im {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal for col2im convolution operations and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected type ($ty)
        /// - num_col_elements correctly describes the column buffer dimensions
        /// - The metadata buffer contains valid shape, stride, padding, and kernel dimensions
        /// - The metadata is formatted as [B, C, H, W, kernel_h, kernel_w, out_h, out_w, pad_h, pad_w, stride_h, stride_w]
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_col_elements: usize,
            metadata: *const c_void,
            col: *const c_void,
            output: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize conv2d ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let col_buffer = match get_buffer_from_map(col as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for col pointer");
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

                // Zero out the output buffer
                let blit_encoder = command_buffer.new_blit_command_encoder();

                // Get dimensions from metadata for output buffer size calculation
                let output_size = if let Some(ref metadata_buff) = metadata_buffer {
                    // This assumes metadata format: [B, C, H, W, kernel_h, kernel_w, out_h, out_w, pad_h, pad_w, stride_h, stride_w]
                    let metadata_ptr = metadata_buff.contents() as *const usize;
                    let b = *metadata_ptr.add(0);
                    let c = *metadata_ptr.add(1);
                    let h = *metadata_ptr.add(2);
                    let w = *metadata_ptr.add(3);
                    b * c * h * w * std::mem::size_of::<$ty>() as usize
                } else {
                    // Fallback sizing logic if metadata is None
                    // You may need to adapt this to your specific requirements
                    num_col_elements * std::mem::size_of::<$ty>() as usize
                };

                blit_encoder.fill_buffer(&output_buffer, metal::NSRange::new(0, output_size as u64), 0);
                blit_encoder.end_encoding();

                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&col_buffer), 0);
                compute_encoder.set_buffer(1, Some(&output_buffer), 0);

                let num_els_buffer = device.new_buffer_with_data(
                    &num_col_elements as *const usize as *const c_void,
                    std::mem::size_of::<usize>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);

                if let Some(ref meta_buffer) = metadata_buffer {
                    compute_encoder.set_buffer(3, Some(meta_buffer), 0);
                } else {
                    compute_encoder.set_buffer(3, None, 0);
                }

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_col_elements as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_dummy_conv2d_im2col {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for unsupported types in MPS for im2col convolution operations.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_col_elements: usize,
            _metadata: *const c_void,
            _input: *const c_void,
            _col: *const c_void,
        ) {
            eprintln!(
                "MPS does not support {} for conv2d_im2col operations",
                stringify!($ty)
            );
        }
    };
}

macro_rules! implement_dummy_conv2d_col2im {
    ($fn_name:ident, $ty:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for unsupported types in MPS for col2im convolution operations.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_col_elements: usize,
            _metadata: *const c_void,
            _col: *const c_void,
            _output: *const c_void,
        ) {
            eprintln!(
                "MPS does not support {} for conv2d_col2im operations",
                stringify!($ty)
            );
        }
    };
}

macro_rules! implement_conv_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                implement_conv2d_im2col!([<metal_conv2d_im2col_ $dtype>], stringify!([<conv2d_im2col_ $dtype>]), $ty);
                implement_conv2d_col2im!([<metal_conv2d_col2im_ $dtype>], stringify!([<conv2d_col2im_ $dtype>]), $ty);
            )*
        }
    };
}

macro_rules! implement_dummy_conv_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                implement_dummy_conv2d_im2col!([<metal_conv2d_im2col_ $dtype>], $ty);
                implement_dummy_conv2d_col2im!([<metal_conv2d_col2im_ $dtype>], $ty);
            )*
        }
    };
}

// Implement for supported types
implement_conv_ops! {
    f32 => f32,
    f16 => f16,
    bf16 => bf16
}

// Implement dummy functions for unsupported types
implement_dummy_conv_ops! {
    f64 => f64
}
