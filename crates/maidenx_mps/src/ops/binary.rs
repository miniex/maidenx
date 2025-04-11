use crate::get_buffer_from_map;
use crate::metal_context::{execute_function, initialize_ops};
use metal::MTLSize;
use std::ffi::c_void;

macro_rules! implement_binary_op {
    ($fn_name:ident, $in_type:ty, $out_type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function performs raw buffer operations using Metal and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected types ($in_type for inputs, $out_type for output)
        /// - num_els and num_dims correctly describe the buffer dimensions
        /// - All pointers and memory access are valid for the entire operation
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            metadata: *const c_void,
            lhs: *const c_void,
            rhs: *const c_void,
            out: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize binary ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let lhs_buffer = match get_buffer_from_map(lhs as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for lhs pointer");
                    return;
                },
            };

            let rhs_buffer = match get_buffer_from_map(rhs as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for rhs pointer");
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

                compute_encoder.set_buffer(0, Some(&lhs_buffer), 0);
                compute_encoder.set_buffer(1, Some(&rhs_buffer), 0);
                compute_encoder.set_buffer(2, Some(&out_buffer), 0);

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

                compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);

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
                command_buffer.wait_until_completed();

                Ok(())
            });
        }
    };
}

macro_rules! implement_dummy_binary_op {
    ($fn_name:ident, $in_type:ty, $out_type:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit types in MPS which are not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _lhs: *const c_void,
            _rhs: *const c_void,
            _out: *const c_void,
        ) {
            // MPS doesn't support 64-bit types, this is a dummy function
            // that will never actually be called because the tensor layer
            // will prevent execution reaching this point.
            eprintln!("MPS does not support 64-bit operations");
        }
    };
}

macro_rules! implement_binary_ops {
    ($dtype:ident => $ty:ty) => {
        paste::paste! {
            implement_binary_op!([<metal_add_ $dtype>], $ty, $ty, stringify!([<add_ $dtype>]));
            implement_binary_op!([<metal_sub_ $dtype>], $ty, $ty, stringify!([<sub_ $dtype>]));
            implement_binary_op!([<metal_mul_ $dtype>], $ty, $ty, stringify!([<mul_ $dtype>]));
            implement_binary_op!([<metal_div_ $dtype>], $ty, $ty, stringify!([<div_ $dtype>]));
            implement_binary_op!([<metal_maximum_ $dtype>], $ty, $ty, stringify!([<maximum_ $dtype>]));
            implement_binary_op!([<metal_minimum_ $dtype>], $ty, $ty, stringify!([<minimum_ $dtype>]));

            implement_binary_op!([<metal_logical_and_ $dtype>], $ty, bool, stringify!([<logical_and_ $dtype>]));
            implement_binary_op!([<metal_logical_or_ $dtype>], $ty, bool, stringify!([<logical_or_ $dtype>]));
            implement_binary_op!([<metal_logical_xor_ $dtype>], $ty, bool, stringify!([<logical_xor_ $dtype>]));

            implement_binary_op!([<metal_eq_ $dtype>], $ty, bool, stringify!([<eq_ $dtype>]));
            implement_binary_op!([<metal_ne_ $dtype>], $ty, bool, stringify!([<ne_ $dtype>]));
            implement_binary_op!([<metal_lt_ $dtype>], $ty, bool, stringify!([<lt_ $dtype>]));
            implement_binary_op!([<metal_le_ $dtype>], $ty, bool, stringify!([<le_ $dtype>]));
            implement_binary_op!([<metal_gt_ $dtype>], $ty, bool, stringify!([<gt_ $dtype>]));
            implement_binary_op!([<metal_ge_ $dtype>], $ty, bool, stringify!([<ge_ $dtype>]));
        }
    };
}

macro_rules! implement_dummy_binary_ops {
    ($dtype:ident => $ty:ty) => {
        paste::paste! {
            implement_dummy_binary_op!([<metal_add_ $dtype>], $ty, $ty);
            implement_dummy_binary_op!([<metal_sub_ $dtype>], $ty, $ty);
            implement_dummy_binary_op!([<metal_mul_ $dtype>], $ty, $ty);
            implement_dummy_binary_op!([<metal_div_ $dtype>], $ty, $ty);
            implement_dummy_binary_op!([<metal_maximum_ $dtype>], $ty, $ty);
            implement_dummy_binary_op!([<metal_minimum_ $dtype>], $ty, $ty);

            implement_dummy_binary_op!([<metal_logical_and_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_logical_or_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_logical_xor_ $dtype>], $ty, bool);

            implement_dummy_binary_op!([<metal_eq_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_ne_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_lt_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_le_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_gt_ $dtype>], $ty, bool);
            implement_dummy_binary_op!([<metal_ge_ $dtype>], $ty, bool);
        }
    };
}

macro_rules! implement_binary_op_inplace {
    ($fn_name:ident, $type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function performs raw in-place buffer operations using Metal and requires that:
        /// - All provided pointers map to valid Metal buffers registered with add_to_buffer_map
        /// - Buffer contents match the expected types ($type for inputs and output)
        /// - num_els and num_dims correctly describe the buffer dimensions
        /// - All pointers and memory access are valid for the entire operation
        /// - The operation is performed in-place on lhs, the out parameter is not used but must be provided
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            metadata: *const c_void,
            lhs: *const c_void,
            rhs: *const c_void,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize binary ops: {:?}", e);
                return;
            }

            // Find base pointers and get the Metal buffers
            let lhs_buffer = match get_buffer_from_map(lhs as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for lhs pointer");
                    return;
                },
            };

            let rhs_buffer = match get_buffer_from_map(rhs as *mut c_void) {
                Some(buffer) => buffer,
                None => {
                    eprintln!("Failed to get Metal buffer for rhs pointer");
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

                compute_encoder.set_buffer(0, Some(&lhs_buffer), 0);
                compute_encoder.set_buffer(1, Some(&rhs_buffer), 0);

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

macro_rules! implement_dummy_binary_op_inplace {
    ($fn_name:ident, $type:ty) => {
        /// # Safety
        ///
        /// This is a dummy function for 64-bit types in MPS which are not supported.
        /// It's unsafe to maintain the same signature as the actual implementation.
        pub unsafe fn $fn_name(
            _num_els: usize,
            _num_dims: usize,
            _metadata: *const c_void,
            _lhs: *const c_void,
            _rhs: *const c_void,
        ) {
            // MPS doesn't support 64-bit types, this is a dummy function
            // that will never actually be called because the tensor layer
            // will prevent execution reaching this point.
            eprintln!("MPS does not support 64-bit operations");
        }
    };
}

macro_rules! implement_binary_ops_inplace {
    ($dtype:ident => $ty:ty) => {
        paste::paste! {
            implement_binary_op_inplace!([<metal_add_inplace_ $dtype>], $ty, stringify!([<add_inplace_ $dtype>]));
            implement_binary_op_inplace!([<metal_sub_inplace_ $dtype>], $ty, stringify!([<sub_inplace_ $dtype>]));
            implement_binary_op_inplace!([<metal_mul_inplace_ $dtype>], $ty, stringify!([<mul_inplace_ $dtype>]));
            implement_binary_op_inplace!([<metal_div_inplace_ $dtype>], $ty, stringify!([<div_inplace_ $dtype>]));
            implement_binary_op_inplace!([<metal_maximum_inplace_ $dtype>], $ty, stringify!([<maximum_inplace_ $dtype>]));
            implement_binary_op_inplace!([<metal_minimum_inplace_ $dtype>], $ty, stringify!([<minimum_inplace_ $dtype>]));
        }
    };
}

macro_rules! implement_dummy_binary_ops_inplace {
    ($dtype:ident => $ty:ty) => {
        paste::paste! {
            implement_dummy_binary_op_inplace!([<metal_add_inplace_ $dtype>], $ty);
            implement_dummy_binary_op_inplace!([<metal_sub_inplace_ $dtype>], $ty);
            implement_dummy_binary_op_inplace!([<metal_mul_inplace_ $dtype>], $ty);
            implement_dummy_binary_op_inplace!([<metal_div_inplace_ $dtype>], $ty);
            implement_dummy_binary_op_inplace!([<metal_maximum_inplace_ $dtype>], $ty);
            implement_dummy_binary_op_inplace!([<metal_minimum_inplace_ $dtype>], $ty);
        }
    };
}

implement_binary_ops!(bf16 => bf16);
implement_binary_ops!(f16 => f16);
implement_binary_ops!(f32 => f32);
implement_binary_ops!(bool => bool);
implement_binary_ops!(u8 => u8);
implement_binary_ops!(u16 => u16);
implement_binary_ops!(u32 => u32);
implement_binary_ops!(i8 => i8);
implement_binary_ops!(i16 => i16);
implement_binary_ops!(i32 => i32);

implement_dummy_binary_ops!(u64 => u64);
implement_dummy_binary_ops!(i64 => i64);
implement_dummy_binary_ops!(f64 => f64);

implement_binary_ops_inplace!(bf16 => bf16);
implement_binary_ops_inplace!(f16 => f16);
implement_binary_ops_inplace!(f32 => f32);
implement_binary_ops_inplace!(bool => bool);
implement_binary_ops_inplace!(u8 => u8);
implement_binary_ops_inplace!(u16 => u16);
implement_binary_ops_inplace!(u32 => u32);
implement_binary_ops_inplace!(i8 => i8);
implement_binary_ops_inplace!(i16 => i16);
implement_binary_ops_inplace!(i32 => i32);

implement_dummy_binary_ops_inplace!(u64 => u64);
implement_dummy_binary_ops_inplace!(i64 => i64);
implement_dummy_binary_ops_inplace!(f64 => f64);
