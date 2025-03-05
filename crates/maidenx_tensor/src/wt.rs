use crate::{
    utils::broadcast::{compute_broadcast_shape, pad_shape},
    Tensor, TensorData, TensorMetadata,
};
#[cfg(feature = "cuda")]
use maidenx_core::buffer::cuda::CudaBuffer;
use maidenx_core::{
    buffer::{cpu::CpuBuffer, Buffer},
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::{Arc, Mutex};

impl Tensor {
    pub fn copy(&self) -> Result<Self> {
        let buffer_len = self.buffer().len();
        let layout_size = self.size();
        let size = if buffer_len != layout_size { buffer_len } else { layout_size };

        let layout = self.layout().clone();
        let device = self.device();
        let dtype = self.dtype();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(self.buffer())?;
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: self.requires_grad(),
            },
            node: None,
        })
    }

    pub fn with_shape(&mut self, shape: &[usize]) -> Result<()> {
        if self.size() != Layout::compute_size(shape) {
            return Err(Error::InvalidShape {
                message: format!(
                    "Shape mismatch: expected total size {}, but got {} for shape {:?}",
                    self.size(),
                    Layout::compute_size(shape),
                    shape
                ),
            });
        }

        self.metadata.layout = Layout::from_shape(shape);

        Ok(())
    }

    pub fn to_shape(&mut self, shape: &[usize]) -> Result<()> {
        if self.size() != Layout::compute_size(shape) {
            return Err(Error::InvalidShape {
                message: format!(
                    "Shape mismatch: expected total size {}, but got {} for shape {:?}",
                    self.size(),
                    Layout::compute_size(shape),
                    shape
                ),
            });
        }

        let mut tensor = self.copy()?;
        tensor.metadata.layout = Layout::from_shape(shape);

        Ok(())
    }

    pub fn with_device(&mut self, device: Device) -> Result<()> {
        let cur_device = self.device();
        if cur_device == device {
            return Ok(());
        }

        let size = self.size();
        let dtype = self.dtype();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_device(self.buffer())?;
        }

        self.data.buffer = buffer;
        self.metadata.device = device;

        Ok(())
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        let cur_device = self.device();
        if cur_device == device {
            return Self::from_tensor(self);
        }

        let size = self.size();
        let dtype = self.dtype();
        let layout = self.layout().clone();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_device(self.buffer())?;
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: self.requires_grad(),
            },
            node: None,
        })
    }

    pub fn with_dtype(&mut self, dtype: DType) -> Result<()> {
        let buffer_len = self.buffer().len(); // read lock for length
        let layout_size = self.size();
        let size = if buffer_len != layout_size { buffer_len } else { layout_size };

        let device = self.device();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(self.buffer())?;
        }

        self.data.buffer = buffer;
        self.metadata.dtype = dtype;

        Ok(())
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        let old_dtype = self.dtype();
        if old_dtype == dtype {
            return Self::from_tensor(self);
        }

        let buffer_len = self.buffer().len();
        let layout_size = self.size();
        let size = if buffer_len != layout_size { buffer_len } else { layout_size };

        let layout = self.layout().clone();
        let device = self.device();

        let mut buffer: Arc<dyn Buffer> = match device {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(self.buffer())?;
        }

        Ok(Self {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device,
                dtype,
                layout,
                requires_grad: self.requires_grad(),
            },
            node: None,
        })
    }

    pub fn with_grad(&mut self) -> Result<()> {
        if matches!(self.metadata.dtype, DType::BOOL | DType::U8 | DType::U32) {
            return Err(Error::UnsupportedDType);
        }

        self.metadata.requires_grad = true;
        if self.data.grad.is_none() {
            let grad_storage = Tensor::zeros_like(self)?;
            self.data.grad = Some(Arc::new(Mutex::new(grad_storage)));
        }

        Ok(())
    }

    pub fn with_broadcast(&mut self, target: &Tensor) -> Result<()> {
        use std::ffi::c_void;

        // ---- case 1) scalar -> target shape ----
        if self.shape().is_empty() {
            let target_size = target.size();
            let dtype = self.dtype();
            let mut new_buffer: Arc<dyn Buffer> = match self.device() {
                Device::CPU => Arc::new(CpuBuffer::new(target_size, dtype)?),
                #[cfg(feature = "cuda")]
                Device::CUDA(id) => Arc::new(CudaBuffer::new(target_size, dtype, id)?),
            };

            let mut scalar_buf = vec![0u8; dtype.size_in_bytes()];
            {
                let src_guard = self.buffer();
                unsafe {
                    src_guard.copy_to_host(scalar_buf.as_mut_ptr() as *mut c_void, scalar_buf.len())?;
                }
            }

            let mut result_buf = vec![0u8; target_size * dtype.size_in_bytes()];
            for i in 0..target_size {
                let offset = i * dtype.size_in_bytes();
                result_buf[offset..offset + scalar_buf.len()].copy_from_slice(&scalar_buf);
            }

            unsafe {
                let dst_mut = Arc::get_mut(&mut new_buffer).ok_or(Error::BufferShared)?;
                dst_mut.copy_from_host(result_buf.as_ptr() as *const c_void, result_buf.len())?;
            }

            self.data.buffer = new_buffer;
            self.metadata.layout = target.layout().clone();

            return Ok(());
        }

        // ---- case 2) target is scalar: do nothing ----
        if target.shape().is_empty() {
            return Ok(());
        }

        // ---- case 3) general broadcast ----
        let broadcasted_shape = compute_broadcast_shape(self.shape(), target.shape())?;
        if self.shape() == broadcasted_shape {
            return Ok(());
        }

        let size = Layout::compute_size(&broadcasted_shape);
        let dtype = self.dtype();
        let mut new_buffer: Arc<dyn Buffer> = match self.device() {
            Device::CPU => Arc::new(CpuBuffer::new(size, dtype)?),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(CudaBuffer::new(size, dtype, id)?),
        };

        let src_size = self.size();
        let mut temp_buf = vec![0u8; src_size * dtype.size_in_bytes()];
        {
            let src_guard = self.buffer();
            unsafe {
                src_guard.copy_to_host(temp_buf.as_mut_ptr() as *mut c_void, temp_buf.len())?;
            }
        }

        let element_size = dtype.size_in_bytes();
        let mut result_buf = vec![0u8; size * element_size];

        let src_strides = self.strides();
        let src_shape = self.shape();
        let padded_src_shape = pad_shape(src_shape, broadcasted_shape.len());
        let mut padded_src_strides = pad_shape(src_strides, broadcasted_shape.len());

        for i in 0..broadcasted_shape.len() {
            if padded_src_shape[i] == 1 {
                padded_src_strides[i] = 0;
            }
        }

        let mut pos = vec![0; broadcasted_shape.len()];
        for i in 0..size {
            let mut remainder = i;
            for d in (0..broadcasted_shape.len()).rev() {
                pos[d] = remainder % broadcasted_shape[d];
                remainder /= broadcasted_shape[d];
            }

            let mut src_idx = 0;
            for d in 0..broadcasted_shape.len() {
                src_idx += pos[d] * padded_src_strides[d];
            }

            let src_offset = src_idx * element_size;
            let dst_offset = i * element_size;
            result_buf[dst_offset..dst_offset + element_size].copy_from_slice(&temp_buf[src_offset..src_offset + element_size]);
        }

        unsafe {
            let dst_mut = Arc::get_mut(&mut new_buffer).ok_or(Error::BufferShared)?;
            dst_mut.copy_from_host(result_buf.as_ptr() as *const c_void, result_buf.len())?;
        }

        self.data.buffer = new_buffer;
        self.metadata.layout = Layout::from_shape(&broadcasted_shape);

        Ok(())
    }

    pub fn to_broadcast(&self, target: &Tensor) -> Result<Self> {
        let mut t = self.copy()?;
        t.with_broadcast(target)?;

        Ok(t)
    }
}
