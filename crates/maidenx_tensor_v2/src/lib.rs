mod adapter;
mod cast;
mod creation;
mod display;
mod iterator;
mod ops;
pub mod prelude;
mod utils;
mod vec;

use dashmap::{
    mapref::one::{Ref, RefMut},
    DashMap,
};
use maidenx_core::{
    buffer::Buffer,
    device::Device,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);
static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(1);
#[inline]
pub(crate) fn next_tensor_id() -> TensorId {
    TensorId(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone)]
pub struct Tensor {
    tid: TensorId,  // a value of 0 is reserved as null; valid IDs start from 1
    gtid: TensorId, // a value of 0 is reserved as null; valid IDs start from 1
    gid: Option<TensorGraphId>,
}

/// ## Identifier & gradient bookkeeping
///
/// This smaller `impl` block deals solely with IDs and simple flags that
/// describe *what* a tensor is rather than *where/what data* it holds.
///
/// * `tid` / `gtid` – globally unique IDs for the value and its gradient  
/// * `gid`           – computation-graph membership (optional)  
/// * `is_const`      – *true* if the tensor is not tracked in a graph  
/// * `requires_grad` – whether automatic differentiation should collect grads  
/// * `grad`          – shallow handle to the gradient tensor (shares storage)  
///
/// No locks or heavy work are involved here; these are pure, branch-only
/// queries suitable for hot code paths.
impl Tensor {
    /// Returns the unique tensor identifier.
    ///
    /// Every tensor is assigned a unique ID at creation time. This ID is used
    /// to locate the tensor's metadata and storage in the global registries.
    ///
    /// # Returns
    ///
    /// The unique TensorId of this tensor.
    #[inline]
    pub fn tid(&self) -> TensorId {
        self.tid
    }

    /// Returns the gradient tensor identifier.
    ///
    /// If this tensor requires gradients, the gtid points to the tensor
    /// that stores the gradient values. A gtid of 0 indicates no gradient.
    ///
    /// # Returns
    ///
    /// The TensorId of the gradient tensor, or TensorId(0) if no gradient exists.
    #[inline]
    pub fn gtid(&self) -> TensorId {
        self.gtid
    }

    /// Returns the graph identifier this tensor belongs to, if any.
    ///
    /// Tensors that are part of a computation graph have an associated graph ID.
    /// This is used for automatic differentiation and lazy evaluation.
    ///
    /// # Returns
    ///
    /// An Option containing the TensorGraphId if the tensor is part of a graph,
    /// or None otherwise.
    #[inline]
    pub fn gid(&self) -> Option<TensorGraphId> {
        self.gid
    }

    /// Checks if the tensor is a constant tensor.
    ///
    /// A tensor is considered constant if it is not part of a computation graph.
    /// Constant tensors are not tracked for automatic differentiation.
    ///
    /// # Returns
    ///
    /// `true` if the tensor is constant (not part of a graph), `false` otherwise.
    #[inline]
    pub fn is_const(&self) -> bool {
        !self.gid.is_some()
    }

    /// Checks if the tensor requires gradient computation.
    ///
    /// A tensor requires gradients if it has a valid gradient tensor ID (gtid).
    /// This is used to determine if the tensor should be tracked during
    /// automatic differentiation.
    ///
    /// # Returns
    ///
    /// `true` if the tensor requires gradient computation, `false` otherwise.
    pub fn requires_grad(&self) -> bool {
        !matches!(self.gtid.0, 0)
    }

    /// Returns the gradient tensor associated with this tensor.
    ///
    /// The returned `Tensor` **shares the same underlying storage and metadata**
    /// with the gradient buffer; it is a *shallow handle*, not a copy.
    /// Its own `gtid` is always `0`, meaning the gradient itself is **not**
    /// tracked for higher-order differentiation.
    ///
    /// # Returns
    ///
    /// A Tensor representing the gradient of this tensor.
    pub fn grad(&self) -> Tensor {
        Tensor {
            tid: self.gtid,
            gtid: TensorId(0),
            gid: self.gid,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Storage registry
// ────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorStorageId(usize);
static STORAGE_COUNTER: AtomicUsize = AtomicUsize::new(1);
#[inline]
pub(crate) fn next_storage_id() -> TensorStorageId {
    TensorStorageId(STORAGE_COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub struct TensorStorage {
    buffer: Arc<dyn Buffer>,
}

impl TensorStorage {
    pub fn new(buffer: Arc<dyn Buffer>) -> Self {
        Self { buffer }
    }

    pub fn buffer(&self) -> &dyn Buffer {
        Arc::as_ref(&self.buffer)
    }

    pub fn buffer_mut(&mut self) -> Result<&mut (dyn Buffer + 'static)> {
        Arc::get_mut(&mut self.buffer).ok_or(Error::BufferShared)
    }

    pub fn buffer_arc(&self) -> Arc<dyn Buffer> {
        self.buffer.clone()
    }

    pub fn with_buffer_mut<F, R>(&mut self, func: F) -> Result<R>
    where
        F: FnOnce(&mut (dyn Buffer + 'static)) -> Result<R>,
    {
        let buf = Arc::get_mut(&mut self.buffer).ok_or(Error::BufferShared)?;
        func(buf)
    }
}

static TENSOR_TO_STORAGE: LazyLock<DashMap<TensorId, TensorStorageId>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 14, 64));

pub fn link_tensor_to_storage(tid: TensorId, sid: TensorStorageId) {
    TENSOR_TO_STORAGE.insert(tid, sid);
}

pub fn get_storage_id(tid: TensorId) -> Option<TensorStorageId> {
    TENSOR_TO_STORAGE.get(&tid).map(|ref_val| *ref_val.value())
}

pub fn get_storage_id_mut(tid: TensorId) -> Option<RefMut<'static, TensorId, TensorStorageId>> {
    TENSOR_TO_STORAGE.get_mut(&tid)
}

static STORAGES: LazyLock<DashMap<TensorStorageId, RwLock<TensorStorage>>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 14, 64));

pub fn insert_storage(sid: TensorStorageId, storage: TensorStorage) {
    STORAGES.insert(sid, RwLock::new(storage));
}

pub fn get_storage(sid: TensorStorageId) -> Option<Ref<'static, TensorStorageId, RwLock<TensorStorage>>> {
    STORAGES.get(&sid)
}

pub fn get_storage_mut(sid: TensorStorageId) -> Option<RefMut<'static, TensorStorageId, RwLock<TensorStorage>>> {
    STORAGES.get_mut(&sid)
}

impl Tensor {
    #[inline]
    fn storage(&self) -> Result<RwLockReadGuard<'static, TensorStorage>> {
        let sid =
            get_storage_id(self.tid()).ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;
        let entry = get_storage(sid).ok_or_else(|| Error::InvalidState("tensor storage not found".into()))?;

        let guard = entry.read().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockReadGuard<'_, TensorStorage>, RwLockReadGuard<'static, TensorStorage>>(guard)
        })
    }

    #[inline]
    fn storage_mut(&self) -> Result<RwLockWriteGuard<'static, TensorStorage>> {
        let sid =
            get_storage_id(self.tid()).ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;
        let entry = get_storage(sid).ok_or_else(|| Error::InvalidState("tensor storage not found".into()))?;

        let guard = entry.write().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockWriteGuard<'_, TensorStorage>, RwLockWriteGuard<'static, TensorStorage>>(guard)
        })
    }

    pub fn is_storaged(&self) -> bool {
        let sid = match get_storage_id(self.tid()) {
            Some(id) => id,
            None => return false,
        };

        get_storage(sid).is_some()
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Metadata registry
// ────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorMode {
    Eager,
    Lazy,
}

thread_local! {
    static GLOBAL_TENSOR_MODE: RefCell<TensorMode> = const {RefCell::new(TensorMode::Eager)};
}

pub struct TensorModeGuard {
    prev_mode: TensorMode,
}

impl TensorModeGuard {
    fn new(mode: TensorMode) -> Self {
        let prev_mode = GLOBAL_TENSOR_MODE.with(|m| {
            let prev = *m.borrow();
            *m.borrow_mut() = mode;
            prev
        });

        Self { prev_mode }
    }
}

impl Drop for TensorModeGuard {
    fn drop(&mut self) {
        GLOBAL_TENSOR_MODE.with(|m| {
            *m.borrow_mut() = self.prev_mode;
        });
    }
}

pub fn get_mode() -> TensorMode {
    GLOBAL_TENSOR_MODE.with(|m| *m.borrow())
}

pub fn eager_mode() -> TensorModeGuard {
    TensorModeGuard::new(TensorMode::Eager)
}

pub fn lazy_mode() -> TensorModeGuard {
    TensorModeGuard::new(TensorMode::Lazy)
}

/// Switches the current lexical scope to **Eager** execution mode.  
/// When the scope ends, the previous mode is automatically restored.
///
/// ### Example
/// ```rust
/// maidenx_tensor::eager!();
/// // Operations inside this scope run immediately.
/// ```
#[macro_export]
macro_rules! eager {
    () => {
        let _guard = eager_mode();
    };
}

/// Switches the current lexical scope to **Lazy** execution mode.  
/// When the scope ends, the previous mode is automatically restored.
///
/// ### Example
/// ```rust
/// maidenx_tensor::lazy!();
/// // Operations inside this scope are recorded to the graph.
/// ```
#[macro_export]
macro_rules! lazy {
    () => {
        let _guard = lazy_mode();
    };
}

pub fn with_mode<F, R>(mode: TensorMode, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = TensorModeGuard::new(mode);
    f()
}

#[derive(Clone)]
pub enum TensorUpdateStatus {
    Pending,
    Materialized,
    Invalidated,
}

#[derive(Clone)]
pub struct TensorMetadata {
    device: Device,
    dtype: DType,
    layout: Layout,
    mode: TensorMode,
    update_status: TensorUpdateStatus,
}

impl TensorMetadata {
    pub fn get_device(&self) -> Device {
        self.device
    }

    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }

    pub fn get_dtype(&self) -> DType {
        self.dtype
    }

    pub fn set_dtype(&mut self, dtype: DType) {
        self.dtype = dtype;
    }

    pub fn get_layout(&self) -> Layout {
        self.layout.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.get_layout().shape().to_vec()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.get_layout().strides().to_vec()
    }

    pub fn offset(&self) -> usize {
        self.get_layout().offset()
    }

    pub fn size(&self) -> usize {
        self.get_layout().size()
    }

    pub fn ndim(&self) -> usize {
        self.get_layout().ndim()
    }

    pub fn dim_size(&self, dim: usize) -> Option<usize> {
        self.get_layout().dim_size(dim)
    }

    pub fn is_contiguous(&self) -> bool {
        self.get_layout().is_contiguous()
    }

    pub fn mode(&self) -> TensorMode {
        self.mode
    }

    pub fn get_update_status(&self) -> TensorUpdateStatus {
        self.update_status.clone()
    }

    pub fn set_update_status(&mut self, update_status: TensorUpdateStatus) {
        self.update_status = update_status;
    }
}

static METADATAS: LazyLock<DashMap<TensorId, RwLock<TensorMetadata>>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 14, 64));

pub fn insert_metadata(tid: TensorId, metadata: TensorMetadata) {
    METADATAS.insert(tid, RwLock::new(metadata));
}

pub fn get_metadata(tid: TensorId) -> Option<Ref<'static, TensorId, RwLock<TensorMetadata>>> {
    METADATAS.get(&tid)
}

pub fn get_metadata_mut(tid: TensorId) -> Option<RefMut<'static, TensorId, RwLock<TensorMetadata>>> {
    METADATAS.get_mut(&tid)
}

/// ## Low-level metadata helpers
///
/// This `impl` block exposes *internal* accessors for metadata and storage:
///
/// * `metadata` / `metadata_mut` – lock-protected access to `TensorMetadata`  
/// * Public read-only views (`device`, `dtype`, `layout`, …)  
///
/// These methods **do not mutate user-visible state** or perform math; they
/// only read bookkeeping structures held in global registries.  Use them for
/// cheap inspection or when implementing new ops.
impl Tensor {
    #[inline]
    fn metadata(&self) -> Result<RwLockReadGuard<'static, TensorMetadata>> {
        let entry = get_metadata(self.tid()).ok_or_else(|| Error::InvalidState("tensor info not found".into()))?;

        let guard = entry.read().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockReadGuard<'_, TensorMetadata>, RwLockReadGuard<'static, TensorMetadata>>(guard)
        })
    }

    #[inline]
    fn metadata_mut(&self) -> Result<RwLockWriteGuard<'static, TensorMetadata>> {
        let entry = get_metadata(self.tid()).ok_or_else(|| Error::InvalidState("tensor info not found".into()))?;

        let guard = entry.write().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockWriteGuard<'_, TensorMetadata>, RwLockWriteGuard<'static, TensorMetadata>>(guard)
        })
    }

    /// Returns the device where this tensor is allocated.
    ///
    /// The device determines where tensor computations will be performed
    /// (e.g., CPU, CUDA, Metal).
    ///
    /// # Returns
    ///
    /// The Device enum value representing the tensor's device.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let device = tensor.device();
    /// ```
    pub fn device(&self) -> Device {
        self.metadata().unwrap().get_device()
    }

    /// Returns the data type of the tensor elements.
    ///
    /// The data type determines the memory layout and precision of each tensor element
    /// (e.g., F32, F64, I32, etc.).
    ///
    /// # Returns
    ///
    /// The DType enum value representing the tensor's element data type.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let dtype = tensor.dtype();
    /// ```
    pub fn dtype(&self) -> DType {
        self.metadata().unwrap().get_dtype()
    }

    /// Returns the memory layout of the tensor.
    ///
    /// The layout contains information about tensor shape, strides, and memory offset.
    /// This is the complete description of how tensor data is arranged in memory.
    ///
    /// # Returns
    ///
    /// A clone of the tensor's Layout.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let layout = tensor.layout();
    /// ```
    pub fn layout(&self) -> Layout {
        self.metadata().unwrap().get_layout().clone()
    }

    /// Returns the shape of the tensor as a vector of dimension sizes.
    ///
    /// The shape describes the size of each dimension of the tensor.
    /// For example, a 2×3 matrix has shape `[2, 3]`.
    ///
    /// # Returns
    ///
    /// A `Vec<usize>` containing the size of each dimension.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert_eq!(tensor.shape(), vec![2, 3]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        self.layout().shape().to_vec()
    }

    /// Returns the strides of the tensor as a vector.
    ///
    /// Strides determine how to index into memory when moving along each dimension.
    /// For contiguous tensors, the stride for a dimension is the product of all
    /// subsequent dimension sizes.
    ///
    /// # Returns
    ///
    /// A `Vec<usize>` containing the stride for each dimension.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let strides = tensor.strides();
    /// ```
    pub fn strides(&self) -> Vec<usize> {
        self.layout().strides().to_vec()
    }

    /// Returns the memory offset of the tensor.
    ///
    /// The offset is the number of elements to skip from the beginning of the
    /// storage buffer to reach the first element of this tensor. This is used
    /// for tensors that view into a portion of a larger storage buffer.
    ///
    /// # Returns
    ///
    /// The offset in number of elements.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let offset = tensor.offset();
    /// ```
    pub fn offset(&self) -> usize {
        self.layout().offset()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This is the product of all dimension sizes in the tensor's shape.
    ///
    /// # Returns
    ///
    /// The total number of elements in the tensor.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert_eq!(tensor.size(), 6); // 2 * 3 = 6 elements
    /// ```
    pub fn size(&self) -> usize {
        self.layout().size()
    }

    /// Returns the number of dimensions in the tensor.
    ///
    /// # Returns
    ///
    /// The number of dimensions (rank) of the tensor.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert_eq!(tensor.ndim(), 2); // 2D tensor
    /// ```
    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    /// Returns the size of the specified dimension.
    ///
    /// # Parameters
    ///
    /// * `dim` - The dimension index to query.
    ///
    /// # Returns
    ///
    /// Some(size) if the dimension exists, None otherwise.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert_eq!(tensor.dim_size(0), Some(2));
    /// assert_eq!(tensor.dim_size(1), Some(3));
    /// assert_eq!(tensor.dim_size(2), None); // Out of bounds
    /// ```
    pub fn dim_size(&self, dim: usize) -> Option<usize> {
        self.layout().dim_size(dim)
    }

    /// Checks if the tensor's memory layout is contiguous.
    ///
    /// A contiguous tensor has elements stored in memory without gaps,
    /// which can make operations more efficient.
    ///
    /// # Returns
    ///
    /// `true` if the tensor is contiguous, `false` otherwise.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert!(tensor.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns the execution mode of the tensor.
    ///
    /// The mode can be either Eager (immediate execution) or Lazy (deferred execution
    /// as part of a computation graph).
    ///
    /// # Returns
    ///
    /// The TensorMode enum value representing the tensor's execution mode.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let mode = tensor.mode();
    /// ```
    pub fn mode(&self) -> TensorMode {
        self.metadata().unwrap().mode()
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Graph
// ────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorNodeId(pub usize);
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(1);
#[inline]
fn next_node_id() -> TensorNodeId {
    TensorNodeId(NODE_COUNTER.fetch_add(1, Ordering::Relaxed))
}

type DynFn = Arc<dyn for<'b> Fn(&'b [TensorId], &'b [TensorId]) -> Result<Vec<TensorId>> + Send + Sync + 'static>;

pub struct TensorNode {
    pub nid: TensorNodeId,
    pub op_name: &'static str,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub compute_fn: Option<DynFn>,
}

impl TensorNode {
    pub fn new(
        op_name: &'static str,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        compute_fn: Option<DynFn>,
    ) -> Self {
        Self {
            nid: next_node_id(),
            op_name,
            inputs,
            outputs,
            compute_fn,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorEdgeId(usize);

#[derive(Clone)]
pub struct TensorEdge {
    from: TensorNodeId,
    to: TensorNodeId,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorGraphId(usize);
static GRAPH_COUNTER: AtomicUsize = AtomicUsize::new(1);
fn next_graph_id() -> TensorGraphId {
    TensorGraphId(GRAPH_COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub struct TensorGraph {
    nodes: HashMap<TensorNodeId, TensorNode>,
    edges: HashMap<TensorEdgeId, (TensorNodeId, TensorNodeId)>,
    topo_order: Vec<TensorNodeId>,
}

impl Default for TensorGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            topo_order: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: TensorNode) {
        self.nodes.insert(node.nid, node);
    }

    pub fn connect(&mut self, from: TensorNodeId, to: TensorNodeId) {
        let eid = TensorEdgeId(self.edges.len() + 1);
        self.edges.insert(eid, (from, to));
    }

    pub fn topo_sort(&mut self) {
        self.topo_order.clear();

        let mut in_degree = HashMap::new();

        for node_id in self.nodes.keys() {
            in_degree.insert(*node_id, 0);
        }

        for (_, to) in self.edges.values() {
            *in_degree.entry(*to).or_insert(0) += 1;
        }

        let mut queue = VecDeque::new();
        for (node_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(*node_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            self.topo_order.push(node_id);

            let outgoing_edges: Vec<_> = self.edges.iter().filter(|(_, (from, _))| *from == node_id).collect();

            for (_, (_, to)) in outgoing_edges {
                if let Some(degree) = in_degree.get_mut(to) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(*to);
                    }
                }
            }
        }

        if self.topo_order.len() != self.nodes.len() {
            self.topo_order.clear();
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Graph registry
// ────────────────────────────────────────────────────────────────────────────

static GRAPHS: LazyLock<DashMap<TensorGraphId, RwLock<TensorGraph>>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 5, 16));

pub fn new_graph() -> TensorGraphId {
    let gid = next_graph_id();
    GRAPHS.insert(gid, RwLock::new(TensorGraph::new()));
    gid
}

pub fn get_graph(gid: TensorGraphId) -> Option<Ref<'static, TensorGraphId, RwLock<TensorGraph>>> {
    GRAPHS.get(&gid)
}

pub fn get_graph_mut(gid: TensorGraphId) -> Option<RefMut<'static, TensorGraphId, RwLock<TensorGraph>>> {
    GRAPHS.get_mut(&gid)
}

// ────────────────────────────────────────────────────────────────────────────
//  Executor
// ────────────────────────────────────────────────────────────────────────────

pub struct TensorGraphExecutor<'a> {
    graph: &'a TensorGraph,
}

impl<'a> TensorGraphExecutor<'a> {
    pub fn new(graph: &'a TensorGraph) -> Self {
        Self { graph }
    }

    pub fn forward(&self) -> Result<()> {
        if self.graph.topo_order.is_empty() {
            return Err(Error::InvalidState("topological order not computed".into()));
        }

        for &nid in &self.graph.topo_order {
            let node = self
                .graph
                .nodes
                .get(&nid)
                .ok_or_else(|| Error::InvalidState(format!("node {} not found", nid.0)))?;

            if let Some(ref f) = node.compute_fn {
                let outs = f(&node.inputs, &node.outputs)?;
                if outs.len() != node.outputs.len() {
                    return Err(Error::InvalidState(format!(
                        "op '{}' returned {} outputs, expected {}",
                        node.op_name,
                        outs.len(),
                        node.outputs.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// ## Execution helpers
///
/// This `impl` block provides runtime helpers that **execute** a tensor’s
/// computation graph.
///
/// * `try_forward`  — fallible forward pass  
/// * `try_backward` — fallible backward pass (autograd)  
/// * `forward`     — thin, infallible wrapper around `try_forward`  
/// * `backward`   — thin, infallible wrapper around `try_backward`  
///
/// All tensors created by the factory helpers are *constant* (`gid == None`);
/// calling these functions on such tensors is therefore an error.
impl Tensor {
    /// Runs [`try_forward`](Self::try_forward) and panics on failure.
    ///
    /// # Panics
    ///
    /// * When the tensor is materialized, constant/detached, or any other
    ///   error occurs during the forward pass.
    pub fn forward(&self) {
        self.try_forward().expect("forward pass failed");
    }

    /// Runs [`try_backward`](Self::try_backward) and panics on failure.
    ///
    /// # Panics
    ///
    /// * When the tensor does not require gradients or any other error
    ///   occurs during the backward pass.
    pub fn backward(&self) {
        self.try_backward().expect("backward pass failed");
    }

    /// Executes the forward pass of the computation graph.
    ///
    /// The graph is topologically sorted on first use and cached afterwards.
    ///
    /// # Errors
    ///
    /// * `InvalidState` – the tensor is constant/detached (no `gid`)  
    /// * `InvalidState` – the tensor is **already materialized**  
    /// * `InvalidState` – the graph cannot be found  
    /// * `Lock`         – the graph’s lock is poisoned  
    pub fn try_forward(&self) -> Result<()> {
        let gid = self.gid.ok_or_else(|| {
            Error::InvalidState("tensor is not connected to a computation graph (constant or detached)".into())
        })?;

        if self.is_storaged() {
            return Err(Error::InvalidState(
                "tensor is already materialized; forward pass is unnecessary".into(),
            ));
        }

        let graph_guard = get_graph(gid).ok_or_else(|| Error::InvalidState("graph not found".into()))?;

        {
            let mut g = graph_guard.write().map_err(|_| Error::Lock)?;
            if g.topo_order.is_empty() {
                g.topo_sort();
            }
        }

        {
            let g = graph_guard.read().map_err(|_| Error::Lock)?;
            TensorGraphExecutor::new(&*g).forward()?;
        }

        Ok(())
    }

    /// Executes the backward pass (automatic differentiation).
    ///
    /// Internally calls `self.grad().try_forward()`, treating the gradient
    /// graph as a normal forward pass.
    ///
    /// # Errors
    ///
    /// * `InvalidState` – the tensor does **not** require gradients  
    /// * Propagates all errors from `try_forward`
    pub fn try_backward(&self) -> Result<()> {
        if !self.requires_grad() {
            return Err(Error::InvalidState(
                "tensor does not require gradients (constant or detached)".into(),
            ));
        }

        self.grad().try_forward()
    }
}
