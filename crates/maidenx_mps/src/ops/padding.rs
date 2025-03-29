use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

macro_rules! implement_padding_ops {
    ($($dtype:ident: $ty:ty),*) => {
        paste::paste! {
            $(
                /// # Safety
                ///
                /// This function performs unsafe constant padding operations on Metal GPU:
                /// - Takes raw pointers as input and output
                /// - Requires valid memory allocations for `inp` and `out` with proper alignment
                /// - `num_els_in` must match the size of the input tensor
                /// - `num_els_out` must match the size of the output tensor
                /// - `num_dims` must accurately reflect tensor dimensionality
                /// - `info` must point to valid metadata with proper tensor shape and padding information
                /// - Caller must ensure `out` points to memory with space for `num_els_out` elements
                /// - All pointers must remain valid for the duration of the function call
                /// - `pad_value` specifies the value used to fill padded regions
                pub unsafe fn [<metal_pad_with_constant_ $dtype:lower>](
                    num_els_in: usize,
                    num_els_out: usize,
                    num_dims: usize,
                    info: *const usize,
                    inp: *const $ty,
                    out: *mut $ty,
                    pad_value: $ty
                ) {
                    unsafe {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize padding ops: {:?}", e);
                            return;
                        }

                        let function_name = format!("metal_pad_with_constant_{}_kernel", stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let output_buffer = device.new_buffer(
                                (num_els_out * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let input_buffer = device.new_buffer_with_data(
                                inp as *const c_void,
                                (num_els_in * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_in_buffer = device.new_buffer_with_data(
                                &num_els_in as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_out_buffer = device.new_buffer_with_data(
                                &num_els_out as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (3 * num_dims + 1 + num_dims + 2 * num_dims) * std::mem::size_of::<usize>();

                            let metadata_buffer = device.new_buffer_with_data(
                                info as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let pad_value_buffer = device.new_buffer_with_data(
                                &pad_value as *const $ty as *const c_void,
                                std::mem::size_of::<$ty>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&output_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&input_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);
                            compute_encoder.set_buffer(6, Some(&pad_value_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_out as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = output_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, out, num_els_out);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                }

                /// # Safety
                ///
                /// This function performs unsafe reflection padding operations on Metal GPU:
                /// - Takes raw pointers as input and output
                /// - Requires valid memory allocations for `inp` and `out` with proper alignment
                /// - `num_els_in` must match the size of the input tensor
                /// - `num_els_out` must match the size of the output tensor
                /// - `num_dims` must accurately reflect tensor dimensionality
                /// - `info` must point to valid metadata with proper tensor shape and padding information
                /// - Caller must ensure `out` points to memory with space for `num_els_out` elements
                /// - All pointers must remain valid for the duration of the function call
                /// - Input dimensions must be larger than padding amount for reflection to be valid
                pub unsafe fn [<metal_pad_with_reflection_ $dtype:lower>](
                    num_els_in: usize,
                    num_els_out: usize,
                    num_dims: usize,
                    info: *const usize,
                    inp: *const $ty,
                    out: *mut $ty
                ) {
                    unsafe {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize padding ops: {:?}", e);
                            return;
                        }

                        let function_name = format!("metal_pad_with_reflection_{}_kernel", stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let output_buffer = device.new_buffer(
                                (num_els_out * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let input_buffer = device.new_buffer_with_data(
                                inp as *const c_void,
                                (num_els_in * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_in_buffer = device.new_buffer_with_data(
                                &num_els_in as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_out_buffer = device.new_buffer_with_data(
                                &num_els_out as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (3 * num_dims + 1 + num_dims + 2 * num_dims) * std::mem::size_of::<usize>();

                            let metadata_buffer = device.new_buffer_with_data(
                                info as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&output_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&input_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_out as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = output_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, out, num_els_out);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                }

                /// # Safety
                ///
                /// This function performs unsafe replication padding operations on Metal GPU:
                /// - Takes raw pointers as input and output
                /// - Requires valid memory allocations for `inp` and `out` with proper alignment
                /// - `num_els_in` must match the size of the input tensor
                /// - `num_els_out` must match the size of the output tensor
                /// - `num_dims` must accurately reflect tensor dimensionality
                /// - `info` must point to valid metadata with proper tensor shape and padding information
                /// - Caller must ensure `out` points to memory with space for `num_els_out` elements
                /// - All pointers must remain valid for the duration of the function call
                pub unsafe fn [<metal_pad_with_replication_ $dtype:lower>](
                    num_els_in: usize,
                    num_els_out: usize,
                    num_dims: usize,
                    info: *const usize,
                    inp: *const $ty,
                    out: *mut $ty
                ) {
                    unsafe {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize padding ops: {:?}", e);
                            return;
                        }

                        let function_name = format!("metal_pad_with_replication_{}_kernel", stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let output_buffer = device.new_buffer(
                                (num_els_out * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let input_buffer = device.new_buffer_with_data(
                                inp as *const c_void,
                                (num_els_in * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_in_buffer = device.new_buffer_with_data(
                                &num_els_in as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_out_buffer = device.new_buffer_with_data(
                                &num_els_out as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (3 * num_dims + 1 + num_dims + 2 * num_dims) * std::mem::size_of::<usize>();

                            let metadata_buffer = device.new_buffer_with_data(
                                info as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&output_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&input_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_out as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = output_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, out, num_els_out);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                }

                /// # Safety
                ///
                /// This function performs unsafe backward pass for constant padding on Metal GPU:
                /// - Takes raw pointers for gradient input and output
                /// - Requires valid memory allocations for `grad_out` and `grad_in` with proper alignment
                /// - `num_els_in` must match the size of the input gradient tensor
                /// - `num_els_out` must match the size of the output gradient tensor
                /// - `num_dims` must accurately reflect tensor dimensionality
                /// - `info` must point to valid metadata with the same format as used in the forward pass
                /// - `grad_in` will be zeroed before accumulating gradients
                /// - All pointers must remain valid for the duration of the function call
                pub unsafe fn [<metal_pad_with_constant_backward_ $dtype:lower>](
                    num_els_in: usize,
                    num_els_out: usize,
                    num_dims: usize,
                    info: *const usize,
                    grad_out: *const $ty,
                    grad_in: *mut $ty
                ) {
                    unsafe {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize padding ops: {:?}", e);
                            return;
                        }

                        std::ptr::write_bytes(grad_in, 0, num_els_in);

                        let function_name = format!("metal_pad_with_constant_backward_{}_kernel", stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let grad_in_buffer = device.new_buffer_with_data(
                                grad_in as *const c_void,
                                (num_els_in * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let grad_out_buffer = device.new_buffer_with_data(
                                grad_out as *const c_void,
                                (num_els_out * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_in_buffer = device.new_buffer_with_data(
                                &num_els_in as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_out_buffer = device.new_buffer_with_data(
                                &num_els_out as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (3 * num_dims + 1 + num_dims + 2 * num_dims) * std::mem::size_of::<usize>();

                            let metadata_buffer = device.new_buffer_with_data(
                                info as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&grad_in_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&grad_out_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_out as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let grad_in_ptr = grad_in_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(grad_in_ptr, grad_in, num_els_in);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                }

                /// # Safety
                ///
                /// This function performs unsafe backward pass for reflection padding on Metal GPU:
                /// - Takes raw pointers for gradient input and output
                /// - Requires valid memory allocations for `grad_out` and `grad_in` with proper alignment
                /// - `num_els_in` must match the size of the input gradient tensor
                /// - `num_els_out` must match the size of the output gradient tensor
                /// - `num_dims` must accurately reflect tensor dimensionality
                /// - `info` must point to valid metadata with the same format as used in the forward pass
                /// - `grad_in` will be zeroed before accumulating gradients
                /// - All pointers must remain valid for the duration of the function call
                pub unsafe fn [<metal_pad_with_reflection_backward_ $dtype:lower>](
                    num_els_in: usize,
                    num_els_out: usize,
                    num_dims: usize,
                    info: *const usize,
                    grad_out: *const $ty,
                    grad_in: *mut $ty
                ) {
                    unsafe {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize padding ops: {:?}", e);
                            return;
                        }

                        std::ptr::write_bytes(grad_in, 0, num_els_in);

                        let function_name = format!("metal_pad_with_reflection_backward_{}_kernel", stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let grad_in_buffer = device.new_buffer_with_data(
                                grad_in as *const c_void,
                                (num_els_in * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let grad_out_buffer = device.new_buffer_with_data(
                                grad_out as *const c_void,
                                (num_els_out * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_in_buffer = device.new_buffer_with_data(
                                &num_els_in as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_out_buffer = device.new_buffer_with_data(
                                &num_els_out as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (3 * num_dims + 1 + num_dims + 2 * num_dims) * std::mem::size_of::<usize>();

                            let metadata_buffer = device.new_buffer_with_data(
                                info as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&grad_in_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&grad_out_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_out as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let grad_in_ptr = grad_in_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(grad_in_ptr, grad_in, num_els_in);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                }

                /// # Safety
                ///
                /// This function performs unsafe backward pass for replication padding on Metal GPU:
                /// - Takes raw pointers for gradient input and output
                /// - Requires valid memory allocations for `grad_out` and `grad_in` with proper alignment
                /// - `num_els_in` must match the size of the input gradient tensor
                /// - `num_els_out` must match the size of the output gradient tensor
                /// - `num_dims` must accurately reflect tensor dimensionality
                /// - `info` must point to valid metadata with the same format as used in the forward pass
                /// - `grad_in` will be zeroed before accumulating gradients
                /// - All pointers must remain valid for the duration of the function call
                pub unsafe fn [<metal_pad_with_replication_backward_ $dtype:lower>](
                    num_els_in: usize,
                    num_els_out: usize,
                    num_dims: usize,
                    info: *const usize,
                    grad_out: *const $ty,
                    grad_in: *mut $ty
                ) {
                    unsafe {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize padding ops: {:?}", e);
                            return;
                        }

                        std::ptr::write_bytes(grad_in, 0, num_els_in);

                        let function_name = format!("metal_pad_with_replication_backward_{}_kernel", stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let grad_in_buffer = device.new_buffer_with_data(
                                grad_in as *const c_void,
                                (num_els_in * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let grad_out_buffer = device.new_buffer_with_data(
                                grad_out as *const c_void,
                                (num_els_out * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_in_buffer = device.new_buffer_with_data(
                                &num_els_in as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_out_buffer = device.new_buffer_with_data(
                                &num_els_out as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (3 * num_dims + 1 + num_dims + 2 * num_dims) * std::mem::size_of::<usize>();

                            let metadata_buffer = device.new_buffer_with_data(
                                info as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&grad_in_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&grad_out_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_in_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_out_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_out as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let grad_in_ptr = grad_in_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(grad_in_ptr, grad_in, num_els_in);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                }
            )*
        }
    }
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
