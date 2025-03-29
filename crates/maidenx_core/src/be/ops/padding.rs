use crate::{
    be::CleanupFn,
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};
use half::{bf16, f16};
use maidenx_cpu::ops::padding::*;
#[cfg(feature = "cuda")]
use maidenx_cuda::{cuda_alloc_and_copy_dims, cuda_free, cuda_set_device, ops::padding::*};
#[cfg(feature = "mps")]
use maidenx_mps::{mps_alloc_and_copy_dims, mps_free, ops::padding::*};

#[macro_export]
macro_rules! declare_padding_op {
    ($name:ident: constant, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els_in: usize,
                num_els_out: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
                pad_value: Scalar,
            ) -> Result<()> {
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

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<pad_with_constant_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                         pad_value.[<as_ $dtype:lower>](),
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
                                    [<cuda_pad_with_constant_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                        pad_value.[<as_ $dtype:lower>](),
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_pad_with_constant_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                        pad_value.[<as_ $dtype:lower>](),
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
    ($name:ident: reflection, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els_in: usize,
                num_els_out: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
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

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<pad_with_reflection_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
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
                                    [<cuda_pad_with_reflection_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_pad_with_reflection_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
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
    ($name:ident: replication, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els_in: usize,
                num_els_out: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
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

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<pad_with_replication_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
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
                                    [<cuda_pad_with_replication_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_pad_with_replication_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
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
    ($name:ident: constant_backward, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                grad_in: &mut dyn Buffer,
                grad_out: &dyn Buffer,
                num_els_in: usize,
                num_els_out: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
                let (metadata, cleanup_fn): (*const usize, CleanupFn) = match grad_in.device() {
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

                match grad_out.device() {
                    Device::CPU => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<pad_with_constant_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_pad_with_constant_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_pad_with_constant_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
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
    ($name:ident: reflection_backward, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                grad_in: &mut dyn Buffer,
                grad_out: &dyn Buffer,
                num_els_in: usize,
                num_els_out: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
                let (metadata, cleanup_fn): (*const usize, CleanupFn) = match grad_in.device() {
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

                match grad_out.device() {
                    Device::CPU => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<pad_with_reflection_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_pad_with_reflection_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_pad_with_reflection_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
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
    ($name:ident: replication_backward, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                grad_in: &mut dyn Buffer,
                grad_out: &dyn Buffer,
                num_els_in: usize,
                num_els_out: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
                let (metadata, cleanup_fn): (*const usize, CleanupFn) = match grad_in.device() {
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

                match grad_out.device() {
                    Device::CPU => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<pad_with_replication_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Device::CUDA(_) => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<cuda_pad_with_replication_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
                                    )
                                }
                            )*
                            _ => return Err(Error::UnsupportedDType)
                        }
                    },
                    #[cfg(feature = "mps")]
                    Device::MPS => {
                        match grad_out.dtype() {
                            $(
                                DType::$dtype => {
                                    [<metal_pad_with_replication_backward_ $dtype:lower>](
                                        num_els_in,
                                        num_els_out,
                                        num_dims,
                                        metadata,
                                        grad_out.as_ptr() as *const [<$dtype:lower>],
                                        grad_in.as_mut_ptr() as *mut [<$dtype:lower>],
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

declare_padding_op!(pad_with_constant: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_padding_op!(pad_with_constant_backward: constant_backward, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);

declare_padding_op!(pad_with_reflection: reflection, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_padding_op!(pad_with_reflection_backward: reflection_backward, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);

declare_padding_op!(pad_with_replication: replication, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_padding_op!(pad_with_replication_backward: replication_backward, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
