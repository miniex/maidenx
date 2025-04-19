use crate::{
    adapter::TensorAdapter, insert_info, insert_storage, link_tensor_to_storage, next_storage_id, next_tensor_id,
    Tensor, TensorId, TensorInfo, TensorMetadata, TensorMode, TensorStorage,
};
use half::{bf16, f16};
use maidenx_core::{
    buffer::BufferManager,
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

        let mut buffer = BufferManager::create(size, device, dtype)?;

        let src_dtype = data.dtype();
        let src_data = data.to_flat_vec()?;

        {
            if src_dtype == dtype {
                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(
                        src_data.as_ptr() as *const std::ffi::c_void,
                        size * dtype.size_in_bytes(),
                        0,
                        0,
                    )?;
                }
            } else {
                let mut converted_data = vec![0u8; size * dtype.size_in_bytes()];

                for i in 0..src_data.len() {
                    let scalar = unsafe {
                        src_dtype.read_scalar((src_data.as_ptr() as *const u8).add(i * src_dtype.size_in_bytes()))
                    };

                    unsafe {
                        dtype.write_scalar(converted_data.as_mut_ptr().add(i * dtype.size_in_bytes()), scalar);
                    }
                }

                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(
                        converted_data.as_ptr() as *const std::ffi::c_void,
                        size * dtype.size_in_bytes(),
                        0,
                        0,
                    )?;
                }
            }
        }

        let tid = next_tensor_id();
        let sid = next_storage_id();
        link_tensor_to_storage(tid, sid);
        insert_storage(sid, TensorStorage::new(buffer));
        let metadata = TensorMetadata { device, dtype, layout };
        insert_info(
            tid,
            TensorInfo {
                metadata,
                mode: TensorMode::Created,
            },
        );

        Ok(Tensor { tid, gid: TensorId(0) })
    }
}
