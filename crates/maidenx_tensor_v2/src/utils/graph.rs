use crate::{
    get_graph_mut, insert_metadata, new_backward_graph, new_forward_graph, next_tensor_id, Tensor, TensorId,
    TensorMetadata, TensorMode, TensorNode, TensorUpdateStatus,
};
use maidenx_core::{
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::Arc;

pub fn add_to_forward_graph<F>(
    op_name: &'static str,
    inputs: &[&Tensor],
    output_devices: &[Device],
    output_dtypes: &[DType],
    output_layouts: &[Layout],
    execute_fn: F,
) -> Result<Vec<Tensor>>
where
    F: Fn(&[TensorId], &[TensorId]) -> Result<Vec<TensorId>> + Send + Sync + 'static,
{
    if inputs.is_empty() {
        return Err(Error::InvalidState("No input tensors provided".into()));
    }
    if output_devices.len() != output_dtypes.len() || output_devices.len() != output_layouts.len() {
        return Err(Error::InvalidState(
            "Mismatch between output devices, dtypes, and layouts".into(),
        ));
    }

    let gid = {
        let mut found_gid = None;
        for input in inputs {
            if let Some(input_gid) = input.graph_id() {
                found_gid = Some(input_gid);
                break;
            }
        }

        match found_gid {
            Some(existing_gid) => existing_gid,
            None => new_forward_graph(),
        }
    };

    let mut output_tids = Vec::new();
    let mut output_tensors = Vec::new();

    for ((&device, &dtype), layout) in output_devices
        .iter()
        .zip(output_dtypes.iter())
        .zip(output_layouts.iter())
    {
        let output_tid = next_tensor_id();

        let metadata = TensorMetadata {
            device,
            dtype,
            layout: layout.clone(),
            grad_tensor_id: None,
            graph_id: Some(gid),
            mode: TensorMode::Lazy,
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        output_tids.push(output_tid);
        output_tensors.push(Tensor(output_tid));
    }

    let input_tids: Vec<TensorId> = inputs.iter().map(|t| t.id()).collect();

    let compute_fn = Arc::new(execute_fn);

    let node = TensorNode::new(op_name, input_tids, output_tids, Some(compute_fn));
    let graph_entry = get_graph_mut(gid).ok_or_else(|| Error::InvalidState(format!("graph {:?} not found", gid)))?;
    let mut graph = graph_entry.write().map_err(|_| Error::Lock)?;
    graph.add_node(node);

    Ok(output_tensors)
}

pub fn add_to_backward_graph<F>(
    op_name: &'static str,
    output_grad: &Tensor,
    input_tensor: &Tensor,
    input_original_shape: &[usize],
    execute_fn: F,
) -> Result<()>
where
    F: Fn(&TensorId) -> Result<Tensor> + Send + Sync + 'static,
{
    let input_grad_tid = input_tensor
        .grad_tensor_id()
        .ok_or_else(|| Error::InvalidState("Input tensor does not have gradient tensor".into()))?;

    let backward_gid = {
        // First check if output_grad already has a graph
        if let Some(output_gid) = output_grad.graph_id() {
            output_gid
        }
        // If not, create a new graph and assign it to output_grad
        else {
            let new_gid = new_backward_graph();
            // Set the output_grad's graph_id so subsequent calls will use the same graph
            if let Some(metadata_ref) = crate::get_metadata_mut(output_grad.id()) {
                if let Ok(mut metadata) = metadata_ref.write() {
                    metadata.graph_id = Some(new_gid);
                }
            }
            new_gid
        }
    };

    if input_tensor.requires_grad() && input_tensor.grad().graph_id().is_none() {
        if let Some(metadata_ref) = crate::get_metadata_mut(input_grad_tid) {
            if let Ok(mut metadata) = metadata_ref.write() {
                metadata.graph_id = Some(backward_gid);
            }
        }
    }

    let original_shape = input_original_shape.to_vec();
    let output_tids = vec![next_tensor_id()];

    let metadata = TensorMetadata {
        device: input_tensor.device(),
        dtype: input_tensor.dtype(),
        layout: Layout::from_shape(&original_shape),
        grad_tensor_id: None,
        graph_id: Some(backward_gid),
        mode: TensorMode::Lazy,
        update_status: TensorUpdateStatus::Pending,
    };
    insert_metadata(output_tids[0], metadata);

    let input_tids = vec![output_grad.id()];
    let compute_fn = Arc::new(
        move |input_tids: &[TensorId], output_tids: &[TensorId]| -> Result<Vec<TensorId>> {
            let grad_output_tid = input_tids[0];

            let grad_input = execute_fn(&grad_output_tid)?;

            if let Some(storage_id) = crate::get_storage_id(grad_input.id()) {
                if let Some(storage_ref) = crate::get_storage(storage_id) {
                    let storage_guard = storage_ref.read().map_err(|_| Error::Lock)?;
                    let buffer = storage_guard.buffer_arc();

                    // Link output tensor
                    let new_sid = crate::next_storage_id();
                    crate::link_tensor_to_storage(output_tids[0], new_sid);
                    crate::insert_storage(new_sid, crate::TensorStorage::new(buffer.clone()));
                    crate::utils::tensor::update_tensor_status(output_tids[0], TensorUpdateStatus::Materialized)?;

                    // Link input gradient tensor
                    let input_sid = crate::next_storage_id();
                    crate::link_tensor_to_storage(input_grad_tid, input_sid);
                    crate::insert_storage(input_sid, crate::TensorStorage::new(buffer));
                    crate::utils::tensor::update_tensor_status(input_grad_tid, TensorUpdateStatus::Materialized)?;
                }
            }

            Ok(vec![output_tids[0]])
        },
    );

    let node = TensorNode::new(op_name, input_tids.clone(), output_tids.clone(), Some(compute_fn));
    let graph_entry = get_graph_mut(backward_gid)
        .ok_or_else(|| Error::InvalidState(format!("backward graph {:?} not found", backward_gid)))?;
    let mut graph = graph_entry.write().map_err(|_| Error::Lock)?;
    graph.add_node(node);

    Ok(())
}
