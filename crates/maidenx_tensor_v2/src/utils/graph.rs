use crate::{
    get_current_graph_id, get_graph_mut, utils::tensor::share_storage_id, with_mode, Tensor, TensorId, TensorMode,
    TensorNode,
};
use maidenx_core::error::{Error, Result};
use std::sync::Arc;

pub fn add_to_graph<F>(op_name: &'static str, inputs: &[&Tensor], outputs: &[&Tensor], execute_fn: F) -> Result<()>
where
    F: Fn(&[Tensor], &[Tensor]) -> Result<()> + Send + Sync + 'static,
{
    if inputs.is_empty() {
        return Err(Error::InvalidState("No input tensors provided".into()));
    }

    let input_tids: Vec<TensorId> = inputs.iter().map(|t| t.id()).collect();
    let output_tids: Vec<TensorId> = outputs.iter().map(|t| t.id()).collect();

    let compute_fn = Arc::new(execute_fn);

    let node = TensorNode::new(op_name, input_tids, output_tids, Some(compute_fn));
    let graph_id = get_current_graph_id();
    let graph_entry =
        get_graph_mut(graph_id).ok_or_else(|| Error::InvalidState(format!("graph {:?} not found", graph_id)))?;
    let mut graph = graph_entry.write().map_err(|_| Error::Lock)?;
    graph.add_node(node);

    Ok(())
}

pub fn accumulate(from: &Tensor, to: &Tensor) -> Result<()> {
    add_to_graph("accumulate", &[from], &[to], move |inputs, outputs| {
        with_mode(TensorMode::Eager, || {
            if outputs[0].is_storaged() {
                let temp = outputs[0].try_add(&inputs[0])?;
                share_storage_id(&temp, &outputs[0])?;
            } else {
                share_storage_id(&inputs[0], &outputs[0])?;
            }
            Ok(())
        })
    })
}
