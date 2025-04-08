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
use maidenx_cpu::ops::binary::*;
#[cfg(feature = "cuda")]
use maidenx_cuda::{cuda_alloc_and_copy_dims, cuda_free, cuda_set_device, ops::binary::*};
#[cfg(feature = "mps")]
use maidenx_mps::{mps_alloc_and_copy_dims, mps_free, ops::binary::*};

macro_rules! impl_for_type {
    ($name:ident, $size:expr, $num_dims:expr, $metadata:expr, $lhs:expr, $rhs:expr, $out:expr, $type:ty, true, $device:expr) => {
        match $device {
            Device::CPU => paste::paste! {
                [<$name _ $type>]($size, $num_dims, $metadata,
                    $lhs.as_ptr() as *const $type,
                    $rhs.as_ptr() as *const $type,
                    $out.as_mut_ptr() as *mut bool)
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => paste::paste! {
                [<cuda_ $name _ $type>]($size, $num_dims, $metadata,
                    $lhs.as_ptr() as *const $type,
                    $rhs.as_ptr() as *const $type,
                    $out.as_mut_ptr() as *mut bool)
            },
            #[cfg(feature = "mps")]
            Device::MPS => paste::paste! {
                [<metal_ $name _ $type>]($size, $num_dims, $metadata as *const std::ffi::c_void,
                    $lhs.as_ptr(),
                    $rhs.as_ptr(),
                    $out.as_ptr())
            },
        }
    };
    ($name:ident, $size:expr, $num_dims:expr, $metadata:expr, $lhs:expr, $rhs:expr, $out:expr, $type:ty, false, $device:expr) => {
        match $device {
            Device::CPU => paste::paste! {
                [<$name _ $type>]($size, $num_dims, $metadata,
                    $lhs.as_ptr() as *const $type,
                    $rhs.as_ptr() as *const $type,
                    $out.as_mut_ptr() as *mut $type)
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => paste::paste! {
                [<cuda_ $name _ $type>]($size, $num_dims, $metadata,
                    $lhs.as_ptr() as *const $type,
                    $rhs.as_ptr() as *const $type,
                    $out.as_mut_ptr() as *mut $type)
            },
            #[cfg(feature = "mps")]
            Device::MPS => paste::paste! {
                [<metal_ $name _ $type>]($size, $num_dims, $metadata as *const std::ffi::c_void,
                    $lhs.as_ptr(),
                    $rhs.as_ptr(),
                    $out.as_ptr())
            },
        }
    };
    ($name:ident, $size:expr, $num_dims:expr, $metadata:expr, $lhs:expr, $rhs:expr, $type:ty, $device:expr) => {
        match $device {
            Device::CPU => paste::paste! {
                [<$name _ $type>]($size, $num_dims, $metadata,
                    $lhs.as_mut_ptr() as *mut $type,
                    $rhs.as_ptr() as *const $type)
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => paste::paste! {
                [<cuda_ $name _ $type>]($size, $num_dims, $metadata,
                    $lhs.as_mut_ptr() as *mut $type,
                    $rhs.as_ptr() as *const $type)
            },
            #[cfg(feature = "mps")]
            Device::MPS => paste::paste! {
                [<metal_ $name _ $type>]($size, $num_dims, $metadata as *const std::ffi::c_void,
                    $lhs.as_ptr(),
                    $rhs.as_ptr())
            },
        }
    };
}

#[macro_export]
macro_rules! declare_binary_op {
    ($name:ident, $compare:expr, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                output: &mut dyn Buffer,
                lhs: &dyn Buffer,
                rhs: &dyn Buffer,
                size: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
                assert_eq!(lhs.dtype(), rhs.dtype(), concat!("DType mismatch in ", stringify!($name)));

                if $compare {
                    assert_eq!(output.dtype(), DType::BOOL, "Output must be BOOL for comparison operations");
                } else {
                    assert_eq!(output.dtype(), lhs.dtype(), "Output dtype must match input dtype");
                }

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

                match lhs.dtype() {
                    $(
                        DType::$dtype => impl_for_type!($name, size, num_dims, metadata, lhs, rhs, output, [<$dtype:lower>], $compare, output.device()),
                    )*
                    _ => return Err(Error::UnsupportedDType)
                };

                if let Some(cleanup) = cleanup_fn {
                    cleanup();
                }

                Ok(())
            }
        }
    };
}

#[macro_export]
macro_rules! declare_binary_op_inplace {
    ($name:ident, [$($dtype:ident),* $(,)?]) => {
        paste::paste! {
            /// # Safety
            /// This function is unsafe because it performs raw pointer operations.
            pub unsafe fn $name(
                lhs: &mut dyn Buffer,
                rhs: &dyn Buffer,
                size: usize,
                num_dims: usize,
                metadata: Option<&[usize]>,
            ) -> Result<()> {
                assert_eq!(lhs.dtype(), rhs.dtype(), concat!("DType mismatch in ", stringify!($name)));

                let (metadata, cleanup_fn): (*const usize, CleanupFn) = match lhs.device() {
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

                match lhs.dtype() {
                    $(
                        DType::$dtype => impl_for_type!($name, size, num_dims, metadata, lhs, rhs, [<$dtype:lower>], lhs.device()),
                    )*
                    _ => return Err(Error::UnsupportedDType)
                };

                if let Some(cleanup) = cleanup_fn {
                    cleanup();
                }

                Ok(())
            }
        }
    };
}

declare_binary_op!(add, false, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(sub, false, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(mul, false, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(div, false, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(maximum, false, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(minimum, false, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);

declare_binary_op!(logical_and, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(logical_or, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(logical_xor, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);

declare_binary_op!(eq, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(ne, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(lt, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(le, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(gt, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op!(ge, true, [BF16, F16, F32, F64, BOOL, U8, U16, U32, U64, I8, I16, I32, I64]);

declare_binary_op_inplace!(add_inplace, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op_inplace!(sub_inplace, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op_inplace!(mul_inplace, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
declare_binary_op_inplace!(div_inplace, [BF16, F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64]);
