use crate::{adapter::TensorAdapter, Tensor, TensorData, TensorMetadata};
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use maidenx_core::buffer::cuda::CudaBuffer;
use maidenx_core::{
    buffer::{cpu::CpuBuffer, Buffer},
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};
use rand::distributions::Distribution;
use std::sync::Arc;

impl Tensor {
    pub fn new<T>(data: T) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::new_with_spec(data, device, dtype)
    }

    pub fn new_with_spec<T>(data: T, device: Device, dtype: DType) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let shape = data.to_shape();
        let layout = Layout::from_shape(&shape);
        let size = layout.size();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        let src_dtype = data.dtype();
        let src_data = data.to_flat_vec()?;

        {
            if src_dtype == dtype {
                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(src_data.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
                }
            } else {
                let mut converted_data = vec![0u8; size * dtype.size_in_bytes()];

                for i in 0..src_data.len() {
                    let scalar = unsafe { src_dtype.read_scalar((src_data.as_ptr() as *const u8).add(i * src_dtype.size_in_bytes())) };

                    unsafe {
                        dtype.write_scalar(converted_data.as_mut_ptr().add(i * dtype.size_in_bytes()), scalar);
                    }
                }

                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(converted_data.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
                }
            }
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: false,
            },
            node: None,
        })
    }

    pub fn from_tensor(target: &Tensor) -> Result<Self> {
        let mut result = Self::empty_with_spec(target.shape(), target.device(), target.dtype())?;

        if target.is_contiguous() {
            unsafe {
                result.with_buffer_mut(|buf| {
                    buf.copy_from(target.buffer())?;
                    Ok(())
                })?;
            }
        } else {
            let contiguous_target = target.contiguous()?;
            unsafe {
                result.with_buffer_mut(|buf| {
                    buf.copy_from(contiguous_target.buffer())?;
                    Ok(())
                })?;
            }
        }

        Ok(result)
    }

    pub fn share_data(target: &Tensor) -> Result<Self> {
        let tensor = Self {
            data: TensorData {
                buffer: Arc::clone(&target.data.buffer),
                grad: None,
            },
            metadata: TensorMetadata {
                device: target.device(),
                dtype: target.dtype(),
                layout: target.layout().clone(),
                requires_grad: false,
            },
            node: None,
        };

        Ok(tensor)
    }

    pub fn empty(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::empty_with_spec(shape, device, dtype)
    }

    pub fn empty_like(src: &Tensor) -> Result<Self> {
        Self::empty_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    pub fn empty_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: false,
            },
            node: None,
        })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::zeros_with_spec(shape, device, dtype)
    }

    pub fn zeros_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        let elem_size = dtype.size_in_bytes();
        let total_bytes = size * elem_size;
        let zero_buf = vec![0u8; total_bytes];

        {
            unsafe {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                buffer_mut.copy_from_host(zero_buf.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: false,
            },
            node: None,
        })
    }

    pub fn zeros_like(src: &Tensor) -> Result<Self> {
        Self::zeros_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    pub fn ones(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::ones_with_spec(shape, device, dtype)
    }

    pub fn ones_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        let one_bytes = match dtype {
            DType::BF16 => bf16::ONE.to_ne_bytes().to_vec(),
            DType::F16 => f16::ONE.to_ne_bytes().to_vec(),
            DType::F32 => 1.0f32.to_ne_bytes().to_vec(),
            DType::F64 => 1.0f64.to_ne_bytes().to_vec(),
            DType::BOOL => vec![1u8],
            DType::U8 => vec![1u8],
            DType::U32 => 1u32.to_ne_bytes().to_vec(),
            DType::I8 => 1i8.to_ne_bytes().to_vec(),
            DType::I32 => 1i32.to_ne_bytes().to_vec(),
            DType::I64 => 1i64.to_ne_bytes().to_vec(),
        };
        let elem_size = dtype.size_in_bytes();
        let total_bytes = size * elem_size;

        let mut host_buf = Vec::with_capacity(total_bytes);
        for _ in 0..size {
            host_buf.extend_from_slice(&one_bytes);
        }

        {
            unsafe {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                buffer_mut.copy_from_host(host_buf.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: false,
            },
            node: None,
        })
    }

    pub fn ones_like(src: &Tensor) -> Result<Self> {
        Self::ones_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    pub fn fill<T: Into<Scalar>>(shape: &[usize], value: T) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::fill_with_spec(shape, value, device, dtype)
    }

    pub fn fill_like<T: Into<Scalar>>(src: &Tensor, value: T) -> Result<Self> {
        Self::fill_with_spec(src.layout().shape(), value, src.device(), src.dtype())
    }

    pub fn fill_with_spec<T: Into<Scalar>>(shape: &[usize], value: T, device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();
        let scalar_value = value.into();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        let value_bytes = match dtype {
            DType::F32 => scalar_value.as_f32().to_ne_bytes().to_vec(),
            DType::F64 => scalar_value.as_f64().to_ne_bytes().to_vec(),
            DType::BF16 => bf16::from_f32(scalar_value.as_f32()).to_ne_bytes().to_vec(),
            DType::F16 => f16::from_f32(scalar_value.as_f32()).to_ne_bytes().to_vec(),
            DType::I32 => scalar_value.as_i32().to_ne_bytes().to_vec(),
            DType::I64 => scalar_value.as_i64().to_ne_bytes().to_vec(),
            DType::U32 => scalar_value.as_u32().to_ne_bytes().to_vec(),
            DType::I8 => (scalar_value.as_i32() as i8).to_ne_bytes().to_vec(),
            DType::U8 => (scalar_value.as_u32() as u8).to_ne_bytes().to_vec(),
            DType::BOOL => vec![if scalar_value.as_bool() { 1u8 } else { 0u8 }],
        };

        let elem_size = dtype.size_in_bytes();
        let mut host_buf = Vec::with_capacity(size * elem_size);

        for _ in 0..size {
            host_buf.extend_from_slice(&value_bytes);
        }

        {
            unsafe {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                buffer_mut.copy_from_host(host_buf.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: false,
            },
            node: None,
        })
    }

    pub fn randn(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::randn_with_spec(shape, device, dtype)
    }

    pub fn randn_like(src: &Tensor) -> Result<Self> {
        Self::randn_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    pub fn randn_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).map_err(|_e| Error::External {
            message: "Failed to create normal distribution with mean=0.0 and std=1.0".to_string(),
        })?;
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();

        let mut result = Self::new_with_spec(data, device, dtype)?;
        result.with_dtype(dtype)?;
        result.with_shape(shape)?;

        Ok(result)
    }
}
