use crate::{adapter::TensorAdapter, Tensor, TensorData};
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use maidenx_core::buffer::cuda::CudaBuffer;
use maidenx_core::{
    buffer::{cpu::CpuBuffer, Buffer},
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::{Error, Result},
    layout::Layout,
};
use rand::distributions::Distribution;
use std::sync::{Arc, RwLock};

impl Tensor {
    pub fn new<T>(data: T) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let device = get_default_device();

        Self::new_with_spec(data, device)
    }

    pub fn new_with_spec<T>(data: T, device: Device) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let dtype = data.dtype();
        let shape = data.to_shape();
        let layout = Layout::from_shape(&shape);
        let size = layout.size();

        let buffer: Arc<RwLock<dyn Buffer>> = match device {
            Device::CPU => Arc::new(RwLock::new(CpuBuffer::new(size, dtype)?)),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(RwLock::new(CudaBuffer::new(size, dtype, id)?)),
        };

        let flat_data = data.to_flat_vec()?;
        {
            let mut guard = buffer.write().map_err(|_| Error::BufferLocked)?;
            unsafe {
                guard.copy_from_host(flat_data.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, layout, grad: None },
            node: None,
            device,
            dtype,
            requires_grad: false,
        })
    }

    pub fn from_tensor(target: &Tensor) -> Result<Self> {
        let result = Self::empty_with_spec(target.shape(), target.device(), target.dtype())?;

        unsafe {
            result.with_buffer_mut(|buf| {
                buf.copy_from(&*target.buffer()?)?;

                Ok(())
            })?;
        }

        Ok(result)
    }

    pub fn empty(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::empty_with_spec(shape, device, dtype)
    }

    pub fn empty_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let buffer: Arc<RwLock<dyn Buffer>> = match device {
            Device::CPU => Arc::new(RwLock::new(CpuBuffer::new(size, dtype)?)),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(RwLock::new(CudaBuffer::new(size, dtype, id)?)),
        };

        Ok(Self {
            data: TensorData { buffer, layout, grad: None },
            node: None,
            device,
            dtype,
            requires_grad: false,
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

        let buffer: Arc<RwLock<dyn Buffer>> = match device {
            Device::CPU => Arc::new(RwLock::new(CpuBuffer::new(size, dtype)?)),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(RwLock::new(CudaBuffer::new(size, dtype, id)?)),
        };

        let elem_size = dtype.size_in_bytes();
        let total_bytes = size * elem_size;
        let zero_buf = vec![0u8; total_bytes];

        {
            let mut guard = buffer.write().map_err(|_| Error::BufferLocked)?;
            unsafe {
                guard.copy_from_host(zero_buf.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, layout, grad: None },
            node: None,
            device,
            dtype,
            requires_grad: false,
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

        let buffer: Arc<RwLock<dyn Buffer>> = match device {
            Device::CPU => Arc::new(RwLock::new(CpuBuffer::new(size, dtype)?)),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(RwLock::new(CudaBuffer::new(size, dtype, id)?)),
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
            let mut guard = buffer.write().map_err(|_| Error::BufferLocked)?;
            unsafe {
                guard.copy_from_host(host_buf.as_ptr() as *const std::ffi::c_void, size * dtype.size_in_bytes())?;
            }
        }

        Ok(Self {
            data: TensorData { buffer, layout, grad: None },
            device,
            dtype,
            node: None,
            requires_grad: false,
        })
    }

    pub fn ones_like(src: &Tensor) -> Result<Self> {
        Self::ones_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    pub fn randn(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::randn_with_spec(shape, device, dtype)
    }

    pub fn randn_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).map_err(|_e| Error::External {
            message: "Failed to create normal distribution with mean=0.0 and std=1.0".to_string(),
        })?;
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();

        let mut result = Self::new_with_spec(data, device)?;
        result.with_dtype(dtype)?;
        result.with_shape(shape)?;

        Ok(result)
    }
}
