use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_unary_op {
    ($fn_name:ident, $in_type:ty, $out_type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal and requires that:
        /// - The input and output pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected types ($in_type for input, $out_type for output)
        /// - num_els and num_dims correctly describe the buffer dimensions
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            metadata: *const c_void,
            input: *const c_void,
            out: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize unary ops: {:?}", e);
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

            let out_buffer = match get_buffer_from_map(out as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for out pointer");
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
                compute_encoder.set_buffer(1, Some(&out_buffer), 0);

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
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_unary_op_with_constant {
    ($fn_name:ident, $in_type:ty, $out_type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal with a constant value and requires that:
        /// - The input and output pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected types ($in_type for input, $out_type for output)
        /// - The constant value is a valid $in_type
        /// - num_els and num_dims correctly describe the buffer dimensions
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            metadata: *const c_void,
            input: *const c_void,
            constant: $in_type,
            out: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize unary ops with constant: {:?}", e);
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

            let out_buffer = match get_buffer_from_map(out as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for out pointer");
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
                compute_encoder.set_buffer(1, Some(&out_buffer), 0);

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

                let constant_buffer = device.new_buffer_with_data(
                    &constant as *const $in_type as *const c_void,
                    std::mem::size_of::<$in_type>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                compute_encoder.set_buffer(5, Some(&constant_buffer), 0);

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

macro_rules! implement_dummy_unary_op {
    ($fn_name:ident, $in_type:ty, $out_type:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit types in MPS which are not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _input: *const c_void,
            _out: *const c_void,
        ) {
            eprintln!(
                "MPS does not support 64-bit unary operations for {}",
                stringify!($fn_name)
            );
        }
    };
}

macro_rules! implement_dummy_unary_op_with_constant {
    ($fn_name:ident, $in_type:ty, $out_type:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit types with constants in MPS which are not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _input: *const c_void,
            _constant: $in_type,
            _out: *const c_void,
        ) {
            eprintln!(
                "MPS does not support 64-bit unary operations with constant for {}",
                stringify!($fn_name)
            );
        }
    };
}

macro_rules! declare_unary_ops {
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

macro_rules! declare_unary_ops_with_constant {
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

macro_rules! declare_dummy_unary_ops {
    ($($dtype:ident => {
        type: $ty:ty,
        ops: [$($op:ident),+ $(,)?; $($bool_op:ident),* $(,)?]
    }),+ $(,)?) => {
        $(
            paste::paste! {
                $(
                    implement_dummy_unary_op!([<metal_ $op _ $dtype:lower>], $ty, $ty);
                )+
                $(
                    implement_dummy_unary_op!([<metal_ $bool_op _ $dtype:lower>], $ty, bool);
                )*
            }
        )+
    };
}

macro_rules! declare_dummy_unary_ops_with_constant {
    ($($dtype:ident => {
        type: $ty:ty,
        ops: [$($op:ident),+ $(,)?; $($bool_op:ident),* $(,)?]
    }),+ $(,)?) => {
        $(
            paste::paste! {
                $(
                    implement_dummy_unary_op_with_constant!([<metal_ $op _ $dtype:lower>], $ty, $ty);
                )+
                $(
                    implement_dummy_unary_op_with_constant!([<metal_ $bool_op _ $dtype:lower>], $ty, bool);
                )*
            }
        )+
    };
}

declare_unary_ops! {
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
        ops: [sign, square, sqrt, relu; logical_not]
    },
    U16 => {
        type: u16,
        ops: [sign, square, sqrt, relu; logical_not]
    },
    U32 => {
        type: u32,
        ops: [sign, square, sqrt, relu; logical_not]
    },
    I8 => {
        type: i8,
        ops: [neg, abs, sign, square, sqrt, relu; logical_not]
    },
    I16 => {
        type: i16,
        ops: [neg, abs, sign, square, sqrt, relu; logical_not]
    },
    I32 => {
        type: i32,
        ops: [neg, abs, sign, square, sqrt, relu; logical_not]
    },
}

declare_unary_ops_with_constant! {
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

declare_dummy_unary_ops! {
    F64 => {
        type: f64,
        ops: [neg, abs, sign, square, sqrt, relu, sigmoid, tanh, gelu, sin, cos, tan, ln, log10, log2, exp, exp10, exp2, softplus, recip; logical_not]
    },
    U64 => {
        type: u64,
        ops: [sign, square, sqrt, relu; logical_not]
    },
    I64 => {
        type: i64,
        ops: [neg, abs, sign, square, sqrt, relu; logical_not]
    }
}

declare_dummy_unary_ops_with_constant! {
    F64 => {
        type: f64,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow, leaky_relu, elu; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    U64 => {
        type: u64,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    },
    I64 => {
        type: i64,
        ops: [add_scalar, sub_scalar, mul_scalar, div_scalar, maximum_scalar, minimum_scalar, pow; eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar]
    }
}
