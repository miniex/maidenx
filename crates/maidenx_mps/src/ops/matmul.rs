use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

#[macro_export]
macro_rules! implement_matmul_ops {
    ($($dtype:ident => $ty:ty),*) => {
        paste::paste! {
            $(
                /// # Safety
                ///
                /// This function performs unsafe operations for matmul on Metal GPU:
                /// - Takes raw pointers as input and output
                /// - Requires valid memory allocations for `a`, `b`, and `c` with proper alignment
                /// - `num_els` must match the size of the output tensor `c`
                /// - `metadata` must point to a valid memory location with proper tensor shape information
                /// - Caller must ensure memory pointed to by `c` has enough space for `num_els` elements
                /// - All pointers must remain valid for the duration of the function call
                /// - Concurrent access to the same memory may lead to undefined behavior
                pub unsafe fn [<metal_matmul_ $dtype>](
                    num_els: usize,
                    metadata: *const usize,
                    a: *const $ty,
                    b: *const $ty,
                    c: *mut $ty,
                ) {
                    if let Err(e) = initialize_ops() {
                        eprintln!("Failed to initialize matmul ops: {:?}", e);
                        return;
                    }

                    let function_name = format!("metal_matmul_{}_kernel", stringify!($dtype));

                    let result = execute_function(&function_name, |pipeline, command_queue, device| {
                        let command_buffer = command_queue.new_command_buffer();
                        let compute_encoder = command_buffer.new_compute_command_encoder();
                        compute_encoder.set_compute_pipeline_state(&pipeline);

                        let c_buffer = device.new_buffer(
                            (num_els * std::mem::size_of::<$ty>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        let out_ndim = *metadata;
                        let a_ndim = *(metadata.add(1));
                        let b_ndim = *(metadata.add(2));

                        let metadata_size =
                            (3 + out_ndim + a_ndim + b_ndim + a_ndim + b_ndim + 2) * std::mem::size_of::<usize>();

                        let a_size = {
                            let a_shape_start = 3 + out_ndim;
                            let mut size = 1;
                            for i in 0..a_ndim {
                                size *= *(metadata.add(a_shape_start + i as usize));
                            }
                            size
                        };

                        let b_size = {
                            let b_shape_start = 3 + out_ndim + a_ndim;
                            let mut size = 1;
                            for i in 0..b_ndim {
                                size *= *(metadata.add(b_shape_start + i as usize));
                            }
                            size
                        };

                        let a_buffer = device.new_buffer_with_data(
                            a as *const c_void,
                            (a_size * std::mem::size_of::<$ty>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        let b_buffer = device.new_buffer_with_data(
                            b as *const c_void,
                            (b_size * std::mem::size_of::<$ty>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        let num_els_buffer = device.new_buffer_with_data(
                            &num_els as *const usize as *const c_void,
                            std::mem::size_of::<usize>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        let metadata_buffer = device.new_buffer_with_data(
                            metadata as *const c_void,
                            metadata_size as u64,
                            MTLResourceOptions::StorageModeShared,
                        );

                        compute_encoder.set_buffer(0, Some(&c_buffer), 0);
                        compute_encoder.set_buffer(1, Some(&a_buffer), 0);
                        compute_encoder.set_buffer(2, Some(&b_buffer), 0);
                        compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);
                        compute_encoder.set_buffer(4, Some(&metadata_buffer), 0);

                        let thread_group_size = MTLSize::new(256, 1, 1);
                        let grid_size = MTLSize::new((num_els as u64 + 255) / 256, 1, 1);

                        compute_encoder.dispatch_threads(grid_size, thread_group_size);

                        compute_encoder.end_encoding();

                        command_buffer.commit();
                        command_buffer.wait_until_completed();

                        let output_ptr = c_buffer.contents() as *const $ty;
                        std::ptr::copy_nonoverlapping(output_ptr, c, num_els);

                        Ok(())
                    });

                    if let Err(e) = result {
                        eprintln!("Failed to execute matmul: {:?}", e);
                    }
                }

                /// # Safety
                ///
                /// This function performs unsafe backward pass operations for matmul on Metal GPU:
                /// - Takes raw pointers for input tensors and gradient outputs
                /// - Requires valid memory allocations for all pointers
                /// - `num_els_a` must match the size of tensor A and `grad_a`
                /// - `num_els_b` must match the size of tensor B and `grad_b`
                /// - `metadata` must point to valid memory with proper tensor shape information
                /// - Caller must ensure memory pointed to by `grad_a` and `grad_b` has enough space
                /// - All pointers must remain valid for the duration of the function call
                /// - `grad_a` and/or `grad_b` can be null if those gradients aren't needed
                /// - Concurrent access to the same memory may lead to undefined behavior
                #[allow(clippy::too_many_arguments)]
                pub unsafe fn [<metal_matmul_backward_ $dtype>](
                    num_els_a: usize,
                    num_els_b: usize,
                    metadata: *const usize,
                    grad_output: *const $ty,
                    a: *const $ty,
                    b: *const $ty,
                    grad_a: *mut $ty,
                    grad_b: *mut $ty,
                ) {
                    if let Err(e) = initialize_ops() {
                        eprintln!("Failed to initialize matmul backward ops: {:?}", e);
                        return;
                    }

                    let out_ndim = *metadata;
                    let a_ndim = *(metadata.add(1));
                    let b_ndim = *(metadata.add(2));

                    let metadata_size =
                        (3 + out_ndim + a_ndim + b_ndim + a_ndim + b_ndim + 2) * std::mem::size_of::<usize>();

                    let grad_output_size = {
                        let out_shape_start = 3;
                        let mut size = 1;
                        if out_ndim > 0 {
                            for i in 0..out_ndim {
                                size *= *(metadata.add(out_shape_start + i as usize));
                            }
                        } else {
                            size = 1;
                        }
                        size
                    };

                    let a_size = {
                        let a_shape_start = 3 + out_ndim;
                        let mut size = 1;
                        for i in 0..a_ndim {
                            size *= *(metadata.add(a_shape_start + i as usize));
                        }
                        size
                    };

                    let b_size = {
                        let b_shape_start = 3 + out_ndim + a_ndim;
                        let mut size = 1;
                        for i in 0..b_ndim {
                            size *= *(metadata.add(b_shape_start + i as usize));
                        }
                        size
                    };

                    if !grad_a.is_null() {
                        let function_name = format!("metal_matmul_backward_{}_grad_a_kernel", stringify!($dtype));

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let grad_a_buffer = device.new_buffer(
                                (num_els_a * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let grad_output_buffer = device.new_buffer_with_data(
                                grad_output as *const c_void,
                                (grad_output_size * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let b_buffer = device.new_buffer_with_data(
                                b as *const c_void,
                                (b_size * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_buffer = device.new_buffer_with_data(
                                &num_els_a as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_buffer = device.new_buffer_with_data(
                                metadata as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&grad_a_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&grad_output_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&b_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_a as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = grad_a_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, grad_a, num_els_a);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute backward grad_a: {:?}", e);
                        }
                    }

                    if !grad_b.is_null() {
                        let function_name = format!("metal_matmul_backward_{}_grad_b_kernel", stringify!($dtype));

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let grad_b_buffer = device.new_buffer(
                                (num_els_b * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let grad_output_buffer = device.new_buffer_with_data(
                                grad_output as *const c_void,
                                (grad_output_size * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let a_buffer = device.new_buffer_with_data(
                                a as *const c_void,
                                (a_size * std::mem::size_of::<$ty>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_buffer = device.new_buffer_with_data(
                                &num_els_b as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_buffer = device.new_buffer_with_data(
                                metadata as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&grad_b_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&grad_output_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&a_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els_b as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = grad_b_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, grad_b, num_els_b);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute backward grad_b: {:?}", e);
                        }
                    }
                }
            )*
        }
    }
}

implement_matmul_ops! {
    bf16 => bf16,
    f16 => f16,
    f32 => f32,
    bool => bool,
    u8 => u8,
    u16 => u16,
    u32 => u32,
    i8 => i8,
    i16 => i16,
    i32 => i32
}

