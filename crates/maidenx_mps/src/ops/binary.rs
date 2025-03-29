use crate::metal_context::{execute_function, initialize_ops};
use half::{bf16, f16};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

macro_rules! implement_binary_op {
    ($fn_name:ident, $in_type:ty, $out_type:ty, $kernel_name:expr) => {
        /// # Safety
        ///
        /// This function is unsafe because it dereferences raw pointers and requires:
        /// * `lhs` must be a valid pointer to an array of at least `num_els` elements of type `$in_type`.
        /// * `rhs` must be a valid pointer to an array of at least `num_els` elements of type `$in_type`.
        /// * `out` must be a valid pointer to an array of at least `num_els` elements of type `$out_type`.
        /// * If `metadata` is not null, it must point to a valid memory location containing dimension and
        ///   stride information formatted as expected by the Metal kernel.
        /// * The memory regions of `lhs`, `rhs`, and `out` must not overlap.
        /// * All pointers must be properly aligned for their respective types.
        pub unsafe fn $fn_name(
            num_els: usize,
            num_dims: usize,
            metadata: *const usize,
            lhs: *const $in_type,
            rhs: *const $in_type,
            out: *mut $out_type,
        ) {
            if let Err(e) = initialize_ops() {
                eprintln!("Failed to initialize binary ops: {:?}", e);
                return;
            }

            let function_name = format!("metal_{}_kernel", $kernel_name);

            let _ = execute_function(&function_name, |pipeline, command_queue, device| {
                let command_buffer = command_queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();

                let lhs_buffer = device.new_buffer_with_data(
                    lhs as *const c_void,
                    (num_els * std::mem::size_of::<$in_type>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let rhs_buffer = device.new_buffer_with_data(
                    rhs as *const c_void,
                    (num_els * std::mem::size_of::<$in_type>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                let out_buffer = device.new_buffer(
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
                    let metadata_size = (3 * num_dims + 2) * std::mem::size_of::<usize>();
                    Some(device.new_buffer_with_data(
                        metadata as *const c_void,
                        metadata_size as u64,
                        MTLResourceOptions::StorageModeShared,
                    ))
                } else {
                    None
                };

                compute_encoder.set_compute_pipeline_state(&pipeline);

                compute_encoder.set_buffer(0, Some(&lhs_buffer), 0);
                compute_encoder.set_buffer(1, Some(&rhs_buffer), 0);
                compute_encoder.set_buffer(2, Some(&out_buffer), 0);
                compute_encoder.set_buffer(3, Some(&num_els_buffer), 0);
                compute_encoder.set_buffer(4, Some(&num_dims_buffer), 0);

                if let Some(ref meta_buf) = metadata_buffer {
                    compute_encoder.set_buffer(5, Some(meta_buf), 0);
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

                unsafe {
                    let out_ptr = out_buffer.contents() as *const $out_type;
                    std::ptr::copy_nonoverlapping(out_ptr, out, num_els);
                }

                Ok(())
            });
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
