use crate::{
    be::CleanupFn,
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
use half::{bf16, f16};
use maidenx_cpu::ops::reduction::*;
#[cfg(feature = "cuda")]
use maidenx_cuda::{cuda_alloc_and_copy_dims, cuda_free, cuda_set_device, ops::reduction::*};

#[macro_export]
macro_rules! declare_reduction_op {
    ($name:ident: standard, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els: usize,
                num_dims: usize,
                num_red_dims: usize,
                dims_and_strides: Option<&[usize]>,
            ) -> Result<()> {
                let (dims_and_strides, cleanup_fn): (*const usize, CleanupFn) = match output.device() {
                    Device::CPU => (
                        dims_and_strides.map_or(std::ptr::null(), |d| d.as_ptr()),
                        None
                    ),
                    #[cfg(feature = "cuda")]
                    Device::CUDA(device_id) => {
                        if cuda_set_device(device_id as i32) != 0 {
                            return Err(Error::CudaError("Failed to set CUDA device".to_string()));
                        }
                        let (ptr, _) = dims_and_strides
                            .map_or((std::ptr::null(), None), |dims| {
                                let (p, _) = cuda_alloc_and_copy_dims(dims);
                                (p as *const usize, Some(dims.len()))
                            });
                        (
                            ptr,
                            Some(Box::new(move || {
                                if !ptr.is_null() {
                                    cuda_free(ptr as *mut std::ffi::c_void);
                                }
                            }) as Box<dyn FnOnce()>)
                        )
                    },
                };

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<$name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        num_red_dims,
                                        dims_and_strides,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_ $name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        num_red_dims,
                                        dims_and_strides,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                }

                if let Some(cleanup) = cleanup_fn {
                    cleanup();
                }

                Ok(())
            }
        }
    };
    ($name:ident: shape, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els: usize,
                num_dims: usize,
                dims_and_strides: Option<&[usize]>,
            ) -> Result<()> {
                let (dims_and_strides, cleanup_fn): (*const usize, CleanupFn) = match output.device() {
                    Device::CPU => (
                        dims_and_strides.map_or(std::ptr::null(), |d| d.as_ptr()),
                        None
                    ),
                    #[cfg(feature = "cuda")]
                    Device::CUDA(device_id) => {
                        if cuda_set_device(device_id as i32) != 0 {
                            return Err(Error::CudaError("Failed to set CUDA device".to_string()));
                        }
                        let (ptr, _) = dims_and_strides
                            .map_or((std::ptr::null(), None), |dims| {
                                let (p, _) = cuda_alloc_and_copy_dims(dims);
                                (p as *const usize, Some(dims.len()))
                            });
                        (
                            ptr,
                            Some(Box::new(move || {
                                if !ptr.is_null() {
                                    cuda_free(ptr as *mut std::ffi::c_void);
                                }
                            }) as Box<dyn FnOnce()>)
                        )
                    },
                };

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<$name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        dims_and_strides,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_ $name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        dims_and_strides,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                }

                if let Some(cleanup) = cleanup_fn {
                    cleanup();
                }

                Ok(())
            }
        }
    };
}

declare_reduction_op!(sum: standard, [BF16, F16, F32, F64, U8, U32, I8, I32, I64]);
declare_reduction_op!(sum_to_shape: shape, [BF16, F16, F32, F64, U8, U32, I8, I32, I64]);
declare_reduction_op!(mean: standard, [BF16, F16, F32, F64]);
