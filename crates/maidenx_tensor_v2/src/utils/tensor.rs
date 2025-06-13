use crate::{
    get_metadata_mut, get_mode, insert_metadata, insert_storage, link_tensor_to_storage, next_storage_id, Tensor, TensorId,
    TensorMetadata, TensorStorage, TensorUpdateStatus,
};
use maidenx_core::{device::Device, dtype::DType, error::{Error, Result}, layout::Layout};
use std::sync::Arc;

/// Helper function to create storage and allocate tensor with given buffer
pub fn create_storage_with_buffer(
    tid: TensorId,
    device: Device,
    dtype: DType,
    layout: Layout,
    buffer: Arc<dyn maidenx_core::buffer::Buffer>,
) -> Result<Tensor> {
    let sid = next_storage_id();
    link_tensor_to_storage(tid, sid);
    insert_storage(sid, TensorStorage::new(buffer));

    let metadata = TensorMetadata {
        device,
        dtype,
        layout,
        mode: get_mode(),
        update_status: TensorUpdateStatus::Materialized,
    };
    insert_metadata(tid, metadata);

    Ok(Tensor {
        tid,
        gtid: TensorId(0),
        gid: None,
    })
}

/// Helper function to update tensor status
pub fn update_tensor_status(tid: TensorId, status: TensorUpdateStatus) -> Result<()> {
    let metadata_ref = get_metadata_mut(tid)
        .ok_or_else(|| Error::InvalidState("tensor metadata not found".into()))?;
    let mut metadata = metadata_ref.write()
        .map_err(|_| Error::InvalidState("failed to acquire metadata lock".into()))?;
    metadata.set_update_status(status);
    Ok(())
}
