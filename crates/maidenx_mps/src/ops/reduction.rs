use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

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
                    /// # Safety
                    ///
                    /// This function is unsafe because:
                    /// - It dereferences raw pointers (`inp`, `out`, `metadata`)
                    /// - Assumes memory regions are properly allocated with sufficient size
                    /// - Assumes `metadata` points to a valid memory region with the expected format
                    /// - Caller must ensure `num_els` matches the actual size of input/output buffers
                    pub unsafe fn [<metal_ $std_op _ $dtype:lower>](
                        num_els: usize,
                        num_dims: usize,
                        num_red_dims: usize,
                        metadata: *const usize,
                        inp: *const $ty,
                        out: *mut $ty,
                    ) {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize reduction ops: {:?}", e);
                            return;
                        }

                        let function_name = format!("metal_{}_{}_kernel", stringify!($std_op), stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let output_size = num_els * std::mem::size_of::<$ty>();
                            let output_buffer = device.new_buffer(
                                output_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let input_size = num_els * std::mem::size_of::<$ty>();
                            let input_buffer = device.new_buffer_with_data(
                                inp as *const c_void,
                                input_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_buffer = device.new_buffer_with_data(
                                &num_els as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_red_dims_buffer = device.new_buffer_with_data(
                                &num_red_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = (num_dims * 2 + num_red_dims * 2 + 1) * std::mem::size_of::<usize>();
                            let metadata_buffer = device.new_buffer_with_data(
                                metadata as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&output_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&input_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&num_red_dims_buffer), 0);
                            compute_encoder.set_buffer(5, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = output_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, out, num_els);

                            Ok(())
                        });

                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
                )*
                $(
                    /// # Safety
                    ///
                    /// This function is unsafe because:
                    /// - It dereferences raw pointers (`inp`, `out`, `metadata`)
                    /// - Assumes memory regions are properly allocated with sufficient size
                    /// - Assumes `metadata` points to a valid memory region with the expected format
                    /// - Caller must ensure `num_els` matches the actual size of input/output buffers
                    /// - Different shape operations require different metadata sizes and formats
                    pub unsafe fn [<metal_ $shape_op _ $dtype:lower>](
                        num_els: usize,
                        num_dims: usize,
                        metadata: *const usize,
                        inp: *const $ty,
                        out: *mut $ty,
                    ) {
                        if let Err(e) = initialize_ops() {
                            eprintln!("Failed to initialize reduction ops: {:?}", e);
                            return;
                        }

                        let function_name = format!("metal_{}_{}_kernel", stringify!($shape_op), stringify!($dtype).to_lowercase());

                        let result = execute_function(&function_name, |pipeline, command_queue, device| {
                            let command_buffer = command_queue.new_command_buffer();
                            let compute_encoder = command_buffer.new_compute_command_encoder();
                            compute_encoder.set_compute_pipeline_state(&pipeline);

                            let output_size = num_els * std::mem::size_of::<$ty>();
                            let output_buffer = device.new_buffer(
                                output_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let input_size = num_els * std::mem::size_of::<$ty>();
                            let input_buffer = device.new_buffer_with_data(
                                inp as *const c_void,
                                input_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_els_buffer = device.new_buffer_with_data(
                                &num_els as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let num_dims_buffer = device.new_buffer_with_data(
                                &num_dims as *const usize as *const c_void,
                                std::mem::size_of::<usize>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let metadata_size = match stringify!($shape_op) {
                                "sum_to_shape" => (3 * num_dims + 1) * std::mem::size_of::<usize>(),
                                "fold" => (2 * num_dims + 6) * std::mem::size_of::<usize>(),
                                _ => (2 * num_dims) * std::mem::size_of::<usize>(),
                            };

                            let metadata_buffer = device.new_buffer_with_data(
                                metadata as *const c_void,
                                metadata_size as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            compute_encoder.set_buffer(0, Some(&output_buffer), 0);
                            compute_encoder.set_buffer(1, Some(&input_buffer), 0);
                            compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                            compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);
                            compute_encoder.set_buffer(4, Some(&metadata_buffer), 0);

                            let thread_group_size = MTLSize::new(256, 1, 1);
                            let grid_size = MTLSize::new((num_els as u64 + 255) / 256, 1, 1);

                            compute_encoder.dispatch_threads(grid_size, thread_group_size);

                            compute_encoder.end_encoding();

                            command_buffer.commit();
                            command_buffer.wait_until_completed();

                            let output_ptr = output_buffer.contents() as *const $ty;
                            std::ptr::copy_nonoverlapping(output_ptr, out, num_els);

                            Ok(())
                        });


                        if let Err(e) = result {
                            eprintln!("Failed to execute {}: {:?}", function_name, e);
                        }
                    }
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
