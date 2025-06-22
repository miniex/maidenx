use crate::{get_metadata_mut, Tensor, TensorId, TensorUpdateStatus};
use maidenx_core::error::{Error, Result};

pub fn update_tensor_status(tid: TensorId, status: TensorUpdateStatus) -> Result<()> {
    let metadata_ref = get_metadata_mut(tid).ok_or_else(|| Error::InvalidState("tensor metadata not found".into()))?;
    let mut metadata = metadata_ref
        .write()
        .map_err(|_| Error::InvalidState("failed to acquire metadata lock".into()))?;
    metadata.set_update_status(status);
    Ok(())
}

pub fn share_storage_id(source: &Tensor, dest: &Tensor) -> Result<()> {
    if let Some(source_storage_id) = crate::get_storage_id(source.id()) {
        crate::link_tensor_to_storage(dest.id(), source_storage_id);
        Ok(())
    } else {
        Err(maidenx_core::error::Error::InvalidState(
            "source tensor has no storage".into(),
        ))
    }
}
