use crate::Tensor;
#[cfg(feature = "cuda")]
use maidenx_core::buffer::cuda::CudaBuffer;
use maidenx_core::{
    buffer::{cpu::CpuBuffer, Buffer},
    device::Device,
    error::{Error, Result},
};
use std::sync::{Arc, RwLock};

impl Tensor {
    pub fn is_contiguous(&self) -> bool {
        let shape = self.shape();
        let strides = self.strides();

        let ndim = shape.len();
        let mut expected_strides = vec![0; ndim];
        let mut acc = 1;
        for i in (0..ndim).rev() {
            expected_strides[i] = acc;
            acc *= shape[i];
        }

        strides == expected_strides
    }

    pub fn contiguous(&mut self) -> Result<()> {
        if self.is_contiguous() {
            return Ok(());
        }

        let shape = self.shape().to_vec();
        let old_strides = self.strides();
        let size = self.size();
        let ndim = shape.len();
        let dtype = self.dtype();

        let byte_size = size * dtype.size_in_bytes();
        let mut host_in = vec![0u8; byte_size];
        unsafe {
            self.buffer()?.copy_to_host(host_in.as_mut_ptr() as *mut std::ffi::c_void, byte_size)?;
        }

        let mut new_strides = vec![0; ndim];
        let mut acc = 1;
        for i in (0..ndim).rev() {
            new_strides[i] = acc;
            acc *= shape[i];
        }

        let mut host_out = vec![0u8; byte_size];

        fn linear_to_nd(mut i: usize, shape: &[usize]) -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                let dim_size = shape[d];
                coords[d] = i % dim_size;
                i /= dim_size;
            }
            coords
        }

        fn nd_to_offset(coords: &[usize], strides: &[usize]) -> usize {
            coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
        }

        let elem_size = dtype.size_in_bytes();
        for i in 0..size {
            let coords = linear_to_nd(i, &shape);
            let old_off = nd_to_offset(&coords, old_strides);
            let new_off = i;

            let src_byte = old_off * elem_size;
            let dst_byte = new_off * elem_size;

            host_out[dst_byte..(dst_byte + elem_size)].copy_from_slice(&host_in[src_byte..(src_byte + elem_size)]);
        }

        let new_buffer: Arc<RwLock<dyn Buffer>> = match self.device() {
            Device::CPU => Arc::new(RwLock::new(CpuBuffer::new(size, dtype)?)),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Arc::new(RwLock::new(CudaBuffer::new(size, dtype, id)?)),
        };

        unsafe {
            new_buffer
                .write()
                .map_err(|_| Error::BufferLocked)?
                .copy_from_host(host_out.as_ptr() as *const std::ffi::c_void, byte_size)?;
        }

        self.data.buffer = new_buffer;
        self.data.layout.set_strides(&new_strides);

        Ok(())
    }

    pub fn detach(&self) -> Result<Self> {
        let mut result = self.clone();
        result.requires_grad = false;
        result.node = None;

        Ok(result)
    }
}
