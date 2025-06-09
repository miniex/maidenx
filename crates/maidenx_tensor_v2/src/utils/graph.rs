use crate::{
    get_graph_mut, insert_metadata, insert_storage, link_tensor_to_storage, new_graph, next_storage_id,
    next_tensor_id, Tensor, TensorId, TensorMetadata, TensorMode, TensorNode, TensorStorage, TensorUpdateStatus,
};
use maidenx_core::{device::Device, dtype::DType, error::{Error, Result}};
use std::sync::Arc;

/// Adds a tensor operation to the computation graph.
/// 
/// This is a generic helper function for adding unary operations (operations with one input 
/// and one output) to the computation graph in lazy mode.
/// 
/// # Parameters
/// 
/// * `tensor` - The input tensor
/// * `op_name` - Name of the operation (e.g., "device_transfer", "dtype_conversion")
/// * `device` - Target device for the output tensor
/// * `dtype` - Target dtype for the output tensor  
/// * `execute_fn` - Function that executes the operation in eager mode
/// 
/// # Returns
/// 
/// A new tensor representing the operation result in the computation graph
pub fn add_to_graph<F>(
    tensor: &Tensor,
    op_name: &'static str, 
    device: Device, 
    dtype: DType, 
    execute_fn: F
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
{
    let gid = tensor.gid().unwrap_or_else(new_graph);
    let output_tid = next_tensor_id();
    let layout = tensor.layout();

    let metadata = TensorMetadata {
        device,
        dtype,
        layout,
        mode: TensorMode::Lazy,
        update_status: TensorUpdateStatus::Pending,
    };
    insert_metadata(output_tid, metadata);

    let compute_fn = Arc::new(
        move |inputs: &[TensorId], outputs: &[TensorId]| -> Result<Vec<TensorId>> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::InvalidState(
                    format!("{} expects 1 input and 1 output", op_name).into(),
                ));
            }

            let input_tensor = Tensor {
                tid: inputs[0],
                gtid: TensorId(0),
                gid: Some(gid),
            };

            let result = execute_fn(&input_tensor)?;
            let sid = next_storage_id();
            link_tensor_to_storage(outputs[0], sid);
            insert_storage(sid, TensorStorage::new(result.storage()?.buffer_arc()));

            Ok(outputs.to_vec())
        },
    );

    let node = TensorNode::new(op_name, vec![tensor.tid()], vec![output_tid], Some(compute_fn));
    if let Some(graph_entry) = get_graph_mut(gid) {
        let mut graph = graph_entry.write().map_err(|_| Error::Lock)?;
        graph.add_node(node);
    }

    Ok(Tensor {
        tid: output_tid,
        gtid: TensorId(0),
        gid: Some(gid),
    })
}