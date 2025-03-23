use crate::{adapter::TensorAdapter, Tensor, TensorData, TensorMetadata};
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use maidenx_core::buffer::cuda::CudaBuffer;
#[cfg(feature = "mps")]
use maidenx_core::buffer::mps::MpsBuffer;
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
        let dtype = data.dtype();

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
            #[cfg(feature = "mps")]
            Device::MPS => Arc::new(MpsBuffer::new(size, dtype)?),
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
            #[cfg(feature = "mps")]
            Device::MPS => Arc::new(MpsBuffer::new(size, dtype)?),
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
            #[cfg(feature = "mps")]
            Device::MPS => Arc::new(MpsBuffer::new(size, dtype)?),
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
            #[cfg(feature = "mps")]
            Device::MPS => Arc::new(MpsBuffer::new(size, dtype)?),
        };

        let one_bytes = match dtype {
            DType::BF16 => bf16::ONE.to_ne_bytes().to_vec(),
            DType::F16 => f16::ONE.to_ne_bytes().to_vec(),
            DType::F32 => 1.0f32.to_ne_bytes().to_vec(),
            DType::F64 => 1.0f64.to_ne_bytes().to_vec(),
            DType::BOOL => vec![1u8],
            DType::U8 => vec![1u8],
            DType::U16 => 1u16.to_ne_bytes().to_vec(),
            DType::U32 => 1u32.to_ne_bytes().to_vec(),
            DType::U64 => 1u64.to_ne_bytes().to_vec(),
            DType::I8 => 1i8.to_ne_bytes().to_vec(),
            DType::I16 => 1i16.to_ne_bytes().to_vec(),
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
            #[cfg(feature = "mps")]
            Device::MPS => Arc::new(MpsBuffer::new(size, dtype)?),
        };

        let value_bytes = match dtype {
            DType::BF16 => bf16::from_f32(scalar_value.as_f32()).to_ne_bytes().to_vec(),
            DType::F16 => f16::from_f32(scalar_value.as_f32()).to_ne_bytes().to_vec(),
            DType::F32 => scalar_value.as_f32().to_ne_bytes().to_vec(),
            DType::F64 => scalar_value.as_f64().to_ne_bytes().to_vec(),
            DType::BOOL => vec![if scalar_value.as_bool() { 1u8 } else { 0u8 }],
            DType::U8 => (scalar_value.as_u32() as u8).to_ne_bytes().to_vec(),
            DType::U16 => scalar_value.as_u16().to_ne_bytes().to_vec(),
            DType::U32 => scalar_value.as_u32().to_ne_bytes().to_vec(),
            DType::U64 => scalar_value.as_u64().to_ne_bytes().to_vec(),
            DType::I8 => (scalar_value.as_i32() as i8).to_ne_bytes().to_vec(),
            DType::I16 => scalar_value.as_i16().to_ne_bytes().to_vec(),
            DType::I32 => scalar_value.as_i32().to_ne_bytes().to_vec(),
            DType::I64 => scalar_value.as_i64().to_ne_bytes().to_vec(),
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

    pub fn range(n: usize) -> Result<Self> {
        Self::arange(0, n as i32, 1)
    }

    pub fn range_with_spec(n: usize, device: Device, dtype: DType) -> Result<Self> {
        Self::arange_with_spec(0, n as i32, 1, device, dtype)
    }

    pub fn arange<T: Into<Scalar>>(start: T, end: T, step: T) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::arange_with_spec(start, end, step, device, dtype)
    }

    pub fn arange_with_spec<T: Into<Scalar>>(start: T, end: T, step: T, device: Device, dtype: DType) -> Result<Self> {
        let start_scalar = start.into();
        let end_scalar = end.into();
        let step_scalar = step.into();

        let (start_val, end_val, step_val) = match dtype {
            DType::BF16 | DType::F16 | DType::F32 => (start_scalar.as_f32(), end_scalar.as_f32(), step_scalar.as_f32()),
            DType::F64 => (start_scalar.as_f64() as f32, end_scalar.as_f64() as f32, step_scalar.as_f64() as f32),
            DType::BOOL | DType::I8 | DType::I16 | DType::I32 => {
                (start_scalar.as_i32() as f32, end_scalar.as_i32() as f32, step_scalar.as_i32() as f32)
            }
            DType::I64 => (start_scalar.as_i64() as f32, end_scalar.as_i64() as f32, step_scalar.as_i64() as f32),
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                (start_scalar.as_u32() as f32, end_scalar.as_u32() as f32, step_scalar.as_u32() as f32)
            }
        };

        if step_val == 0.0 {
            return Err(Error::InvalidArgument("arange: step cannot be zero".to_string()));
        }

        let count = ((end_val - start_val) / step_val).ceil() as usize;

        let values: Vec<f32> = (0..count).map(|i| start_val + (i as f32) * step_val).collect();
        match dtype {
            DType::BF16 | DType::F16 => {
                let mut tensor = Self::new_with_spec(values, device, DType::F32)?;
                tensor.with_dtype(dtype)?;
                Ok(tensor)
            }
            DType::F32 => Self::new_with_spec(values, device, dtype),
            DType::F64 => {
                let double_values: Vec<f64> = values.into_iter().map(|v| v as f64).collect();
                Self::new_with_spec(double_values, device, dtype)
            }
            DType::BOOL => {
                let bool_values: Vec<bool> = values.into_iter().map(|v| v != 0.0).collect();
                Self::new_with_spec(bool_values, device, dtype)
            }
            DType::U8 => {
                let uint_values: Vec<u8> = values
                    .into_iter()
                    .map(|v| {
                        if v < 0.0 {
                            0
                        } else if v > u8::MAX as f32 {
                            u8::MAX
                        } else {
                            v as u8
                        }
                    })
                    .collect();
                Self::new_with_spec(uint_values, device, dtype)
            }
            DType::U16 => {
                let uint_values: Vec<u16> = values
                    .into_iter()
                    .map(|v| {
                        if v < 0.0 {
                            0
                        } else if v > u16::MAX as f32 {
                            u16::MAX
                        } else {
                            v as u16
                        }
                    })
                    .collect();
                Self::new_with_spec(uint_values, device, dtype)
            }
            DType::U32 => {
                let uint_values: Vec<u32> = values.into_iter().map(|v| if v < 0.0 { 0 } else { v as u32 }).collect();
                Self::new_with_spec(uint_values, device, dtype)
            }
            DType::U64 => {
                let uint_values: Vec<u64> = values.into_iter().map(|v| if v < 0.0 { 0 } else { v as u64 }).collect();
                Self::new_with_spec(uint_values, device, dtype)
            }
            DType::I8 => {
                let int_values: Vec<i8> = values
                    .into_iter()
                    .map(|v| {
                        if v < i8::MIN as f32 {
                            i8::MIN
                        } else if v > i8::MAX as f32 {
                            i8::MAX
                        } else {
                            v as i8
                        }
                    })
                    .collect();
                Self::new_with_spec(int_values, device, dtype)
            }
            DType::I16 => {
                let int_values: Vec<i16> = values
                    .into_iter()
                    .map(|v| {
                        if v < i16::MIN as f32 {
                            i16::MIN
                        } else if v > i16::MAX as f32 {
                            i16::MAX
                        } else {
                            v as i16
                        }
                    })
                    .collect();
                Self::new_with_spec(int_values, device, dtype)
            }
            DType::I32 => {
                let int_values: Vec<i32> = values.into_iter().map(|v| v as i32).collect();
                Self::new_with_spec(int_values, device, dtype)
            }
            DType::I64 => {
                let int_values: Vec<i64> = values.into_iter().map(|v| v as i64).collect();
                Self::new_with_spec(int_values, device, dtype)
            }
        }
    }
}
