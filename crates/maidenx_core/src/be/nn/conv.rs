#![allow(unused_imports)]
#![allow(unreachable_patterns)]

use crate::{
    be::CleanupFn,
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
use half::{bf16, f16};
use maidenx_cpu::nn::conv::*;
#[cfg(feature = "cuda")]
use maidenx_cuda::{cuda_alloc_and_copy_dims, cuda_free, cuda_set_device, nn::conv::*};

#[macro_export]
macro_rules! declare_conv2d_op {
    ([$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn im2col(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els: usize,
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
                                    [<conv2d_im2col_ $dtype:lower>](
                                        num_els,
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
                                    [<cuda_conv2d_im2col_ $dtype:lower>](
                                        num_els,
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

            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn col2im(
                output: &dyn Buffer,
                col: &dyn Buffer,
                num_els: usize,
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

                match output.device() {
                    Device::CPU => {
                        match output.dtype() {
                            $(
                                DType::$dtype => {
                                    [<conv2d_col2im_ $dtype:lower>](
                                        num_els,
                                        dims_and_strides,
                                        col.as_ptr() as *const [<$dtype:lower>],
                                        output.as_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match output.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_conv2d_col2im_ $dtype:lower>](
                                        num_els,
                                        dims_and_strides,
                                        col.as_ptr() as *const [<$dtype:lower>],
                                        output.as_ptr() as *mut [<$dtype:lower>],
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

declare_conv2d_op!([BF16, F16, F32, F64]);
