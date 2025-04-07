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
use maidenx_cpu::ops::matmul::*;
#[cfg(feature = "cuda")]
use maidenx_cuda::{cuda_alloc_and_copy_dims, cuda_free, cuda_set_device, ops::matmul::*};
#[cfg(feature = "mps")]
use maidenx_mps::{mps_alloc_and_copy_dims, mps_free, ops::matmul::*};

#[macro_export]
macro_rules! declare_matmul_op {
    ([$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn matmul(
                output: &mut dyn Buffer,
                lhs: &dyn Buffer,
                rhs: &dyn Buffer,
                num_els: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
                assert_eq!(lhs.dtype(), rhs.dtype(), concat!("DType mismatch in ", stringify!($name)));

                let (metadata, cleanup_fn): (*const usize, CleanupFn) = match output.device() {
                    Device::CPU => (
                        metadata.map_or(std::ptr::null(), |d| d.as_ptr()),
                        None
                    ),
                    #[cfg(feature = "cuda")]
                    Device::CUDA(device_id) => {
                        if cuda_set_device(device_id as i32) != 0 {
                            return Err(Error::CudaError("Failed to set CUDA device".to_string()));
                        }
                        let (ptr, _) = metadata
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
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        let (ptr, _) = metadata
                            .map_or((std::ptr::null(), None), |dims| {
                                let (p, len) = mps_alloc_and_copy_dims(dims);
                                (p as *const usize, Some(len))
                            });
                        (
                            ptr,
                            Some(Box::new(move || {
                                if !ptr.is_null() {
                                    mps_free(ptr as *mut std::ffi::c_void);
                                }
                            }) as Box<dyn FnOnce()>)
                        )
                    },
                };

                match lhs.device() {
                    Device::CPU => {
                        match lhs.dtype() {
                            $(
                                DType::$dtype => {
                                    [<matmul_ $dtype:lower>](
                                        num_els,
                                        metadata,
                                        lhs.as_ptr() as *const [<$dtype:lower>],
                                        rhs.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match lhs.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_matmul_ $dtype:lower>](
                                        num_els,
                                        metadata,
                                        lhs.as_ptr() as *const [<$dtype:lower>],
                                        rhs.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match lhs.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_matmul_ $dtype:lower>](
                                        num_els,
                                        metadata as *const std::ffi::c_void,
                                        lhs.as_ptr(),
                                        rhs.as_ptr(),
                                        output.as_ptr(),
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
            pub unsafe fn matmul_backward(
                grad_lhs: Option<&mut dyn Buffer>,
                grad_rhs: Option<&mut dyn Buffer>,
                grad_output: &dyn Buffer,
                lhs: &dyn Buffer,
                rhs: &dyn Buffer,
                num_els_a: usize,
                num_els_b: usize,
                metadata: Option<&[usize]>,

            ) -> Result<()> {
                assert_eq!(lhs.dtype(), rhs.dtype(), concat!("DType mismatch in ", stringify!($name)));

                let (metadata, cleanup_fn): (*const usize, CleanupFn) = match grad_output.device() {
                    Device::CPU => (
                        metadata.map_or(std::ptr::null(), |d| d.as_ptr()),
                        None
                    ),
                    #[cfg(feature = "cuda")]
                    Device::CUDA(device_id) => {
                        if cuda_set_device(device_id as i32) != 0 {
                            return Err(Error::CudaError("Failed to set CUDA device".to_string()));
                        }
                        let (ptr, _) = metadata
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
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        let (ptr, _) = metadata
                            .map_or((std::ptr::null(), None), |dims| {
                                let (p, len) = mps_alloc_and_copy_dims(dims);
                                (p as *const usize, Some(len))
                            });
                        (
                            ptr,
                            Some(Box::new(move || {
                                if !ptr.is_null() {
                                    mps_free(ptr as *mut std::ffi::c_void);
                                }
                            }) as Box<dyn FnOnce()>)
                        )
                    },
                };

                match lhs.device() {
                    Device::CPU => {
                        match lhs.dtype() {
                            $(
                                DType::$dtype => {
                                    [<matmul_backward_ $dtype:lower>](
                                        num_els_a,
                                        num_els_b,
                                        metadata,
                                        grad_output.as_ptr() as *const [<$dtype:lower>],
                                        lhs.as_ptr() as *const [<$dtype:lower>],
                                        rhs.as_ptr() as *const [<$dtype:lower>],
                                        grad_lhs.map_or(std::ptr::null_mut(), |buf| buf.as_mut_ptr() as *mut [<$dtype:lower>]),
                                        grad_rhs.map_or(std::ptr::null_mut(), |buf| buf.as_mut_ptr() as *mut [<$dtype:lower>]),
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match lhs.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_matmul_backward_ $dtype:lower>](
                                        num_els_a,
                                        num_els_b,
                                        metadata,
                                        grad_output.as_ptr() as *const [<$dtype:lower>],
                                        lhs.as_ptr() as *const [<$dtype:lower>],
                                        rhs.as_ptr() as *const [<$dtype:lower>],
                                        grad_lhs.map_or(std::ptr::null_mut(), |buf| buf.as_mut_ptr() as *mut [<$dtype:lower>]),
                                        grad_rhs.map_or(std::ptr::null_mut(), |buf| buf.as_mut_ptr() as *mut [<$dtype:lower>]),
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match lhs.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_matmul_backward_ $dtype:lower>](
                                        num_els_a,
                                        num_els_b,
                                        metadata as *const std::ffi::c_void,
                                        grad_output.as_ptr(),
                                        lhs.as_ptr(),
                                        rhs.as_ptr(),
                                        grad_lhs.map_or(std::ptr::null_mut(), |buf| buf.as_ptr()),
                                        grad_rhs.map_or(std::ptr::null_mut(), |buf| buf.as_ptr()),
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

declare_matmul_op!([BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
