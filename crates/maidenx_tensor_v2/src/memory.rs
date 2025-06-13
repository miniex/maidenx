use crate::{get_mode, next_tensor_id, utils::graph::add_to_graph, Tensor, TensorId, TensorMode};
use maidenx_core::{
    buffer::BufferManager,
    error::{Error, Result},
    layout::Layout,
};
use std::sync::Arc;

/// ## Memory layout conversion helpers
///
/// This `impl` block provides methods for converting tensors to contiguous memory layout.
/// There are two main categories:
///
/// * **New tensor creation**: `contiguous`, `try_contiguous` - creates new contiguous tensors
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct memory reordering
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **Performance notes:**
/// * Contiguous tensors have better cache locality and enable vectorized operations
/// * If tensor is already contiguous, data is simply copied without reordering
/// * Non-contiguous tensors require element-by-element reordering which is slower
impl Tensor {
    /// Runs [`try_contiguous`](Self::try_contiguous) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let contiguous_tensor = tensor.contiguous();
    /// assert!(contiguous_tensor.is_contiguous());
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer creation fails
    /// * When memory reordering fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn contiguous(&self) -> Self {
        self.try_contiguous().expect("failed to create contiguous tensor")
    }

    /// Attempts to convert tensor to contiguous memory layout, returning a new tensor.
    ///
    /// This operation creates a new tensor with contiguous memory layout. If the tensor
    /// is already contiguous, the data is simply copied. If not, elements are reordered
    /// to achieve contiguous layout.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn make_contiguous() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[2, 3]);
    ///     let contiguous_tensor = tensor.try_contiguous()?;
    ///     assert!(contiguous_tensor.is_contiguous());
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer creation fails
    /// - Memory copying or reordering fails
    /// - Graph operations fail in lazy mode
    ///
    /// # Performance Notes
    ///
    /// * If tensor is already contiguous, this is a simple memory copy
    /// * For non-contiguous tensors, requires element-by-element reordering
    /// * Contiguous tensors enable more efficient subsequent operations
    pub fn try_contiguous(&self) -> Result<Self> {
        let shape = self.shape();
        let contiguous_layout = Layout::from_shape(&shape);

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = crate::TensorMetadata {
                    device: self.device(),
                    dtype: self.dtype(),
                    layout: contiguous_layout,
                    mode: get_mode(),
                    update_status: crate::TensorUpdateStatus::Pending,
                };
                crate::insert_metadata(target_tid, metadata);
                self.execute_contiguous(target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_graph(
                    &[self],
                    "contiguous",
                    &[self.device()],
                    &[self.dtype()],
                    &[contiguous_layout],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_contiguous(target_tids[0])?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    fn execute_contiguous(&self, target_tid: TensorId) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype();
        let layout = self.layout();
        let shape = layout.shape();

        let contiguous_layout = Layout::from_shape(shape);
        let total_elements = contiguous_layout.size();

        let mut buffer = BufferManager::create(total_elements, device, dtype)?;

        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;

            if layout.is_contiguous() {
                buffer_mut.copy_from_with_device(self.storage()?.buffer(), 0, 0, total_elements)?;
            } else {
                let elem_size = dtype.size_in_bytes();
                let storage = self.storage()?;
                let src_buffer = storage.buffer();

                let mut host_data = vec![0u8; total_elements * elem_size];

                let mut src_host_data = vec![0u8; src_buffer.len() * elem_size];
                unsafe {
                    src_buffer.copy_to_host(
                        src_host_data.as_mut_ptr() as *mut std::ffi::c_void,
                        src_buffer.len() * elem_size,
                        0,
                        0,
                    )?;
                }

                let mut dst_idx = 0;
                self.iter_indices(|indices| {
                    let mut src_offset = layout.offset();
                    for (idx, &stride) in indices.iter().zip(layout.strides().iter()) {
                        src_offset += idx * stride;
                    }
                    let src_offset = src_offset * elem_size;
                    let dst_offset = dst_idx * elem_size;

                    if src_offset + elem_size <= src_host_data.len() && dst_offset + elem_size <= host_data.len() {
                        host_data[dst_offset..dst_offset + elem_size]
                            .copy_from_slice(&src_host_data[src_offset..src_offset + elem_size]);
                    }

                    dst_idx += 1;
                });

                unsafe {
                    buffer_mut.copy_from_host(
                        host_data.as_ptr() as *const std::ffi::c_void,
                        total_elements * elem_size,
                        0,
                        0,
                    )?;
                }
            }
        }

        let sid = crate::next_storage_id();
        crate::link_tensor_to_storage(target_tid, sid);
        crate::insert_storage(sid, crate::TensorStorage::new(buffer));

        crate::utils::tensor::update_tensor_status(target_tid, crate::TensorUpdateStatus::Materialized)?;

        Ok(Tensor {
            tid: target_tid,
            gtid: crate::TensorId(0),
            gid: None,
        })
    }
}
