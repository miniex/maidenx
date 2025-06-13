//! ## Graph construction utilities
//!
//! This module provides helper functions for building computation graphs in lazy mode.
//! These utilities handle the creation of tensor nodes, management of tensor metadata,
//! and orchestration of deferred execution.
//!
//! **Key concepts:**
//! * **Computation graphs**: DAG structures representing tensor operations
//! * **Deferred execution**: Operations are recorded but not executed until `forward()` is called
//! * **Storage management**: Efficient memory allocation and linking for graph outputs
//! * **Multi-tensor support**: Handles operations with multiple inputs and outputs
//!
//! **Usage patterns:**
//! * Called internally by tensor operations in lazy mode
//! * Manages tensor ID allocation and metadata creation
//! * Links execution functions to graph nodes for later evaluation

use crate::{
    get_graph_mut, insert_metadata, new_graph, next_tensor_id, Tensor, TensorId, TensorMetadata, TensorMode,
    TensorNode, TensorUpdateStatus,
};
use maidenx_core::{
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::Arc;

/// Adds a tensor operation to the computation graph with custom layout specification.
///
/// This function adds operations to the computation graph in lazy mode with automatic
/// dependency edge management. It supports custom layout specification for operations
/// that change tensor layout (view, transpose, etc.).
///
/// # Parameters
///
/// * `inputs` - Slice of input tensors that will be consumed by the operation
/// * `op_name` - Name of the operation for debugging and graph visualization
/// * `output_devices` - Target devices for each output tensor
/// * `output_dtypes` - Target data types for each output tensor
/// * `output_layouts` - Target layouts for each output tensor
/// * `execute_fn` - Function that executes the operation, receiving input tensors and target tensor IDs
///
/// # Returns
///
/// A vector of new tensors representing the operation results in the computation graph.
/// These tensors are in pending state and will be materialized when `forward()` is called.
///
/// # Examples
///
/// ```
/// // View operation with custom layout
/// let mut new_layout = tensor.layout();
/// new_layout.view(&new_shape)?;
/// let result = add_to_graph(
///     &[&tensor],
///     "view",
///     &[tensor.device()],
///     &[tensor.dtype()],
///     &[new_layout],
///     |inputs, target_ids| {
///         Ok(vec![inputs[0].execute_view(&new_shape, target_ids[0])?])
///     }
/// )?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// * No input tensors are provided
/// * Number of output devices, dtypes, and layouts don't match
/// * Graph storage allocation fails
/// * Tensor metadata creation fails
/// * The execution function returns wrong number of results during computation
///
/// # Performance Notes
///
/// * Output tensor IDs are pre-allocated for memory efficiency
/// * Metadata is created immediately with correct layout information
/// * Dependency edges are automatically added based on tensor relationships
/// * The execution function is stored as a closure and called during `forward()`
pub fn add_to_graph<F>(
    inputs: &[&Tensor],
    op_name: &'static str,
    output_devices: &[Device],
    output_dtypes: &[DType],
    output_layouts: &[Layout],
    execute_fn: F,
) -> Result<Vec<Tensor>>
where
    F: Fn(&[Tensor], &[TensorId]) -> Result<Vec<Tensor>> + Send + Sync + 'static,
{
    if inputs.is_empty() {
        return Err(Error::InvalidState("No input tensors provided".into()));
    }
    if output_devices.len() != output_dtypes.len() || output_devices.len() != output_layouts.len() {
        return Err(Error::InvalidState("Mismatch between output devices, dtypes, and layouts".into()));
    }

    let gid = inputs[0].gid().unwrap_or_else(new_graph);

    // Create output tensors
    let mut output_tids = Vec::new();
    let mut output_tensors = Vec::new();

    for ((&device, &dtype), layout) in output_devices.iter().zip(output_dtypes.iter()).zip(output_layouts.iter()) {
        let output_tid = next_tensor_id();

        let metadata = TensorMetadata {
            device,
            dtype,
            layout: layout.clone(),
            mode: TensorMode::Lazy,
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(output_tid, metadata);

        output_tids.push(output_tid);
        output_tensors.push(Tensor {
            tid: output_tid,
            gtid: TensorId(0),
            gid: Some(gid),
        });
    }

    let input_tids: Vec<TensorId> = inputs.iter().map(|t| t.tid()).collect();

    let compute_fn = Arc::new(
        move |inputs: &[TensorId], outputs: &[TensorId]| -> Result<Vec<TensorId>> {
            let input_tensors: Vec<Tensor> = inputs
                .iter()
                .map(|&tid| Tensor {
                    tid,
                    gtid: TensorId(0),
                    gid: Some(gid),
                })
                .collect();

            let results = execute_fn(&input_tensors, outputs)?;

            if results.len() != outputs.len() {
                return Err(Error::InvalidState(
                    format!(
                        "{} returned {} results but expected {}",
                        op_name,
                        results.len(),
                        outputs.len()
                    )
                    .into(),
                ));
            }

            Ok(outputs.to_vec())
        },
    );

    let node = TensorNode::new(op_name, input_tids, output_tids, Some(compute_fn));
    let graph_entry = get_graph_mut(gid)
        .ok_or_else(|| Error::InvalidState(format!("graph {} not found", gid.0)))?;
    let mut graph = graph_entry.write().map_err(|_| Error::Lock)?;
    graph.add_node(node); // This will automatically add dependency edges

    Ok(output_tensors)
}
