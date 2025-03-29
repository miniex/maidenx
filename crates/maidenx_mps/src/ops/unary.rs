use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

macro_rules! implement_unary_op {
    ($fn_name:ident, $in_type:ty, $out_type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function is unsafe because it dereferences raw pointers and requires:
        /// * `input` must be a valid pointer to an array of at least `num_els` elements of type `$in_type`.
        /// * `output` must be a valid pointer to an array of at least `num_els` elements of type `$out_type`.
        /// * If `metadata` is not null, it must point to a valid memory location containing dimension and
        ///   stride information formatted as expected by the Metal kernel.
        /// * The memory regions of `input` and `output` must not overlap.
        /// * All pointers must be properly aligned for their respective types.
        pub unsafe fn $fn_name(num_els: usize, num_dims: usize, metadata: *const usize, input: *const $in_type, output: *mut $out_type) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize unary ops: {:?}", e);
                return;
            }

            let function_name = format!("metal_{}_kernel", $kernel_name);

            let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                let command_buffer = command_queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();

                let input_buffer = device.new_buffer_with_data(
                    input as *const c_void,
                    (num_els * std::mem::size_of::<$in_type>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let output_buffer = device.new_buffer(
                    (num_els * std::mem::size_of::<$out_type>()) as u64,
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

                let metadata_buffer = if !metadata.is_null() {
                    // For unary ops, metadata contains [dims, strides, offset]
                    let metadata_size = (2 * num_dims + 1) * std::mem::size_of::<usize>();
                    Some(device.new_buffer_with_data(
                        metadata as *const c_void,
                        metadata_size as u64,
                        MTLResourceOptions::StorageModeShared,
                    ))
                } else {
                    None
                };

                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&input_buffer), 0);
                compute_encoder.set_buffer(1, Some(&output_buffer), 0);
                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);

                if let Some(ref meta_buf) = metadata_buffer {
                    compute_encoder.set_buffer(4, Some(meta_buf), 0);
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

                unsafe {
                    let out_ptr = output_buffer.contents() as *const $out_type;
                    std::ptr::copy_nonoverlapping(out_ptr, output, num_els);
                }

                Ok(())
            });
        }
    };
}

macro_rules! implement_unary_op_with_constant {
    ($fn_name:ident, $in_type:ty, $out_type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function is unsafe because it dereferences raw pointers and requires:
        /// * `input` must be a valid pointer to an array of at least `num_els` elements of type `$in_type`.
        /// * `output` must be a valid pointer to an array of at least `num_els` elements of type `$out_type`.
        /// * If `metadata` is not null, it must point to a valid memory location containing dimension and
        ///   stride information formatted as expected by the Metal kernel.
        /// * The memory regions of `input` and `output` must not overlap.
        /// * All pointers must be properly aligned for their respective types.
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            input: *const $in_type,
            constant: $in_type,
            output: *mut $out_type,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize unary ops with constant: {:?}", e);
                return;
            }

            let function_name = format!("metal_{}_kernel", $kernel_name);

            let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                let command_buffer = command_queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();

                let input_buffer = device.new_buffer_with_data(
                    input as *const c_void,
                    (num_els * std::mem::size_of::<$in_type>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let output_buffer = device.new_buffer(
                    (num_els * std::mem::size_of::<$out_type>()) as u64,
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

                let metadata_buffer = if !metadata.is_null() {
                    // For unary ops, metadata contains [dims, strides, offset]
                    let metadata_size = (2 * num_dims + 1) * std::mem::size_of::<usize>();
                    Some(device.new_buffer_with_data(
                        metadata as *const c_void,
                        metadata_size as u64,
                        MTLResourceOptions::StorageModeShared,
                    ))
                } else {
                    None
                };

                let constant_buffer = device.new_buffer_with_data(
                    &constant as *const $in_type as *const c_void,
                    std::mem::size_of::<$in_type>() as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&input_buffer), 0);
                compute_encoder.set_buffer(1, Some(&output_buffer), 0);
                compute_encoder.set_buffer(2, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_dims_buffer), 0);

                if let Some(ref meta_buf) = metadata_buffer {
                    compute_encoder.set_buffer(4, Some(meta_buf), 0);
                } else {
                    compute_encoder.set_buffer(4, None, 0);
                }

                compute_encoder.set_buffer(5, Some(&constant_buffer), 0);

                let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup() as u64;
                let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
                let num_threadgroups = MTLSize::new((num_els as u64).div_ceil(threads_per_threadgroup), 1, 1);
                compute_encoder.dispatch_thread_groups(num_threadgroups, threadgroup_size);
                compute_encoder.end_encoding();

                command_buffer.commit();
                command_buffer.wait_until_completed();

                unsafe {
                    let out_ptr = output_buffer.contents() as *const $out_type;
                    std::ptr::copy_nonoverlapping(out_ptr, output, num_els);
                }

                Ok(())
            });
        }
    };
}

macro_rules! declare_extern_unary_ops {
    ($($dtype:ident => {
        type: $ty:ty,
        ops: [$($op:ident),+ $(,)?; $($bool_op:ident),* $(,)?]
    }),+ $(,)?) => {
        $(
            paste::paste! {
                $(
                    implement_unary_op!([<metal_ $op _ $dtype:lower>], $ty, $ty, stringify!([<$op _ $dtype:lower>]));
                )+
                $(
                    implement_unary_op!([<metal_ $bool_op _ $dtype:lower>], $ty, bool, stringify!([<$bool_op _ $dtype:lower>]));
                )*
            }
        )+
    };
}

macro_rules! declare_extern_unary_ops_with_constant {
    ($($dtype:ident => {
        type: $ty:ty,
        ops: [$($op:ident),+ $(,)?; $($bool_op:ident),* $(,)?]
    }),+ $(,)?) => {
        $(
            paste::paste! {
                $(
                    implement_unary_op_with_constant!([<metal_ $op _ $dtype:lower>], $ty, $ty, stringify!([<$op _ $dtype:lower>]));
                )+
                $(
                    implement_unary_op_with_constant!([<metal_ $bool_op _ $dtype:lower>], $ty, bool, stringify!([<$bool_op _ $dtype:lower>]));
                )*
            }
        )+
    };
}

declare_extern_unary_ops! {
    BF16 => {
        type: bf16,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    F16 => {
        type: f16,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    F32 => {
        type: f32,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    BOOL => {
        type: bool,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh; logical_not]
    },
    U8 => {
        type: u8,
        ops: [sign, square, sqrt; logical_not]
    },
    U16 => {
        type: u16,
        ops: [sign, square, sqrt; logical_not]
    },
    U32 => {
        type: u32,
        ops: [sign, square, sqrt; logical_not]
    },
    I8 => {
        type: i8,
        ops: [neg, abs, sign, square, sqrt; logical_not]
    },
    I16 => {
        type: i16,
        ops: [neg, abs, sign, square, sqrt; logical_not]
    },
    I32 => {
        type: i32,
        ops: [neg, abs, sign, square, sqrt; logical_not]
    },
}

declare_extern_unary_ops_with_constant! {
    BF16 => {
        type: bf16,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    F16 => {
        type: f16,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    F32 => {
        type: f32,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    BOOL => {
        type: bool,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    U8 => {
        type: u8,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    U16 => {
        type: u16,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    U32 => {
        type: u32,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I8 => {
        type: i8,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I16 => {
        type: i16,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I32 => {
        type: i32,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    }
}
