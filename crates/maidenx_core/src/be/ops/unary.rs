#![allow(unreachable_patterns)]

use crate::{
    be::CleanupFn,
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
    scalar::Scalar,
};
use half::{bf16, f16};
use maidenx_cpu::ops::unary::*;
#[cfg(feature = "cuda")]
use maidenx_cuda::{cuda_alloc_and_copy_dims, cuda_free, cuda_set_device, ops::unary::*};

#[macro_export]
macro_rules! declare_unary_op {
    ($name:ident: standard, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els: usize,
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
                };

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<$name _ $dtype:lower>](
                                        num_els,
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
                                    [<cuda_ $name _ $dtype:lower>](
                                        num_els,
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
    ($name:ident: to_bool, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                num_els: usize,
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
                };

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<$name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut bool,
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
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        output.as_mut_ptr() as *mut bool,
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
    ($name:ident: constant, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                constant: Scalar,
                num_els: usize,
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
                };

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<$name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        constant.[<as_ $dtype:lower>](),
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
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        constant.[<as_ $dtype:lower>](),
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
    ($name:ident: constant_to_bool, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                input: &dyn Buffer,
                constant: Scalar,
                num_els: usize,
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
                };

                match input.device() {
                    Device::CPU => {
                        match input.dtype() {
                            $(
                                DType::$dtype => {
                                    [<$name _ $dtype:lower>](
                                        num_els,
                                        num_dims,
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        constant.[<as_ $dtype:lower>](),
                                        output.as_mut_ptr() as *mut bool,
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
                                        metadata,
                                        input.as_ptr() as *const [<$dtype:lower>],
                                        constant.[<as_ $dtype:lower>](),
                                        output.as_mut_ptr() as *mut bool,
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

// Regular unary operations
declare_unary_op!(neg: standard, [BF16, F16, F32, F64, I8, I16, I32, I64]);
declare_unary_op!(abs: standard, [BF16, F16, F32, F64, I8, I16, I32, I64]);
declare_unary_op!(sign: standard, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(square: standard, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(sqrt: standard, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(relu: standard, [BF16, F16, F32, F64]);
declare_unary_op!(sigmoid: standard, [BF16, F16, F32, F64]);
declare_unary_op!(tanh: standard, [BF16, F16, F32, F64]);
declare_unary_op!(gelu: standard, [BF16, F16, F32, F64]);
declare_unary_op!(sin: standard, [BF16, F16, F32, F64]);
declare_unary_op!(cos: standard, [BF16, F16, F32, F64]);
declare_unary_op!(tan: standard, [BF16, F16, F32, F64]);
declare_unary_op!(ln: standard, [BF16, F16, F32, F64]);
declare_unary_op!(log10: standard, [BF16, F16, F32, F64]);
declare_unary_op!(log2: standard, [BF16, F16, F32, F64]);
declare_unary_op!(exp: standard, [BF16, F16, F32, F64]);
declare_unary_op!(exp10: standard, [BF16, F16, F32, F64]);
declare_unary_op!(exp2: standard, [BF16, F16, F32, F64]);
declare_unary_op!(softplus: standard, [BF16, F16, F32, F64]);
declare_unary_op!(recip: standard, [BF16, F16, F32, F64]);
declare_unary_op!(logical_not: to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);

// Operations with constant
declare_unary_op!(add_scalar: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(sub_scalar: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(mul_scalar: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(div_scalar: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(maximum_scalar: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(minimum_scalar: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(pow: constant, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(leaky_relu: constant, [BF16, F16, F32, F64]);
declare_unary_op!(elu: constant, [BF16, F16, F32, F64]);

// Comparison ops with constant
declare_unary_op!(eq_scalar: constant_to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(ne_scalar: constant_to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(lt_scalar: constant_to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(le_scalar: constant_to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(gt_scalar: constant_to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_unary_op!(ge_scalar: constant_to_bool, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
