mod adapter;
mod creation;
mod ops;
mod vec;
mod wt;

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
    tid: TensorId, // a value of 0 is reserved as null; valid IDs start from 1
    gid: TensorId, // a value of 0 is reserved as null; valid IDs start from 1
}

impl Tensor {
    pub fn tid(&self) -> TensorId {
        self.tid
    }

    pub fn gid(&self) -> TensorId {
        self.gid
    }

    pub fn requires_grad(&self) -> bool {
        match self.gid.0 {
            0 => false,
            _ => true,
        }
    }

    pub fn grad(&self) -> Tensor {
        Tensor {
            tid: self.gid(),
            gid: TensorId(0),
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
}

// ────────────────────────────────────────────────────────────────────────────
//  Info registry
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TensorMetadata {
    device: Device,
    dtype: DType,
    layout: Layout,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorMode {
    Created,
    Eager,
    Lazy,
}

pub struct TensorInfo {
    metadata: TensorMetadata,
    mode: TensorMode,
}

impl TensorInfo {
    pub fn device(&self) -> Device {
        self.metadata.device
    }

    pub fn dtype(&self) -> DType {
        self.metadata.dtype
    }

    pub fn layout(&self) -> Layout {
        self.metadata.layout.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.layout().shape().to_vec()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.layout().strides().to_vec()
    }

    pub fn offset(&self) -> usize {
        self.layout().offset()
    }

    pub fn size(&self) -> usize {
        self.layout().size()
    }

    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    pub fn dim_size(&self, dim: usize) -> Option<usize> {
        self.layout().dim_size(dim)
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    pub fn mode(&self) -> TensorMode {
        self.mode
    }
}

static INFOS: LazyLock<DashMap<TensorId, RwLock<TensorInfo>>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 14, 64));

pub fn insert_info(id: TensorId, info: TensorInfo) {
    INFOS.insert(id, RwLock::new(info));
}

pub fn get_info(id: TensorId) -> Option<Ref<'static, TensorId, RwLock<TensorInfo>>> {
    INFOS.get(&id)
}

pub fn get_info_mut(id: TensorId) -> Option<RefMut<'static, TensorId, RwLock<TensorInfo>>> {
    INFOS.get_mut(&id)
}

impl Tensor {
    #[inline]
    fn info(&self) -> Result<RwLockReadGuard<'static, TensorInfo>> {
        let entry = get_info(self.tid()).ok_or_else(|| Error::InvalidState("tensor info not found".into()))?;

        let guard = entry.read().map_err(|_| Error::Lock)?;
        Ok(unsafe { mem::transmute::<RwLockReadGuard<'_, TensorInfo>, RwLockReadGuard<'static, TensorInfo>>(guard) })
    }

    #[inline]
    fn info_mut(&self) -> Result<RwLockWriteGuard<'static, TensorInfo>> {
        let entry = get_info(self.tid()).ok_or_else(|| Error::InvalidState("tensor info not found".into()))?;

        let guard = entry.write().map_err(|_| Error::Lock)?;
        Ok(unsafe { mem::transmute::<RwLockWriteGuard<'_, TensorInfo>, RwLockWriteGuard<'static, TensorInfo>>(guard) })
    }

    pub fn device(&self) -> Device {
        self.info().unwrap().device()
    }

    pub fn dtype(&self) -> DType {
        self.info().unwrap().dtype()
    }

    pub fn layout(&self) -> Layout {
        self.info().unwrap().layout().clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.layout().shape().to_vec()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.layout().strides().to_vec()
    }

    pub fn offset(&self) -> usize {
        self.layout().offset()
    }

    pub fn size(&self) -> usize {
        self.layout().size()
    }

    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    pub fn dim_size(&self, dim: usize) -> Option<usize> {
        self.layout().dim_size(dim)
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    pub fn mode(&self) -> TensorMode {
        self.info().unwrap().mode()
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

type DynFn<'a, T> = Arc<dyn Fn(T) -> Result<Vec<TensorId>> + Send + Sync + 'a>;

pub struct TensorNode<'a> {
    pub nid: TensorNodeId,
    pub op_name: &'static str,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub forward_fn: Option<DynFn<'a, &'a [TensorId]>>,
    pub backward_fn: Option<DynFn<'a, (&'a [TensorId], &'a [TensorId])>>,
}

impl<'a> TensorNode<'a> {
    pub fn new(
        op_name: &'static str,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        forward_fn: Option<DynFn<'a, &'a [TensorId]>>,
        backward_fn: Option<DynFn<'a, (&'a [TensorId], &'a [TensorId])>>,
    ) -> Self {
        Self {
            nid: next_node_id(),
            op_name,
            inputs,
            outputs,
            forward_fn,
            backward_fn,
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

pub struct TensorGraph<'a> {
    nodes: HashMap<TensorNodeId, TensorNode<'a>>,
    fwd_edges: HashMap<TensorEdgeId, (TensorNodeId, TensorNodeId)>,
    bwd_edges: HashMap<TensorEdgeId, (TensorNodeId, TensorNodeId)>,
    fwd_topo_order: Vec<TensorNodeId>,
    bwd_topo_order: Vec<TensorNodeId>,
}

impl Default for TensorGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> TensorGraph<'a> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            fwd_edges: HashMap::new(),
            bwd_edges: HashMap::new(),
            fwd_topo_order: Vec::new(),
            bwd_topo_order: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: TensorNode<'a>) {
        self.nodes.insert(node.nid, node);
    }

    pub fn connect_fwd(&mut self, from: TensorNodeId, to: TensorNodeId) {
        let eid = TensorEdgeId(self.fwd_edges.len() + 1);
        self.fwd_edges.insert(eid, (from, to));
    }

    pub fn connect_bwd(&mut self, from: TensorNodeId, to: TensorNodeId) {
        let eid = TensorEdgeId(self.bwd_edges.len() + 1);
        self.bwd_edges.insert(eid, (from, to));
    }

    pub fn topo_sort_fwd(&mut self) {
        self.fwd_topo_order.clear();

        let mut in_degree = HashMap::new();

        for node_id in self.nodes.keys() {
            in_degree.insert(*node_id, 0);
        }

        for (_, (_, to)) in &self.fwd_edges {
            *in_degree.entry(*to).or_insert(0) += 1;
        }

        let mut queue = VecDeque::new();
        for (node_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(*node_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            self.fwd_topo_order.push(node_id);

            let outgoing_edges: Vec<_> = self
                .fwd_edges
                .iter()
                .filter(|(_, (from, _))| *from == node_id)
                .collect();

            for (_, (_, to)) in outgoing_edges {
                if let Some(degree) = in_degree.get_mut(to) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(*to);
                    }
                }
            }
        }

        if self.fwd_topo_order.len() != self.nodes.len() {
            self.fwd_topo_order.clear();
        }
    }

    pub fn topo_sort_bwd(&mut self) {
        self.bwd_topo_order.clear();

        let mut in_degree = HashMap::new();

        for node_id in self.nodes.keys() {
            in_degree.insert(*node_id, 0);
        }

        for (_, (_, to)) in &self.bwd_edges {
            *in_degree.entry(*to).or_insert(0) += 1;
        }

        let mut queue = VecDeque::new();
        for (node_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(*node_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            self.bwd_topo_order.push(node_id);

            let outgoing_edges: Vec<_> = self
                .bwd_edges
                .iter()
                .filter(|(_, (from, _))| *from == node_id)
                .collect();

            for (_, (_, to)) in outgoing_edges {
                if let Some(degree) = in_degree.get_mut(to) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(*to);
                    }
                }
            }
        }

        if self.bwd_topo_order.len() != self.nodes.len() {
            self.bwd_topo_order.clear();
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Graph registry
// ────────────────────────────────────────────────────────────────────────────

static GRAPHS: LazyLock<DashMap<TensorGraphId, RwLock<TensorGraph<'static>>>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 10, 64));

pub fn new_graph() -> TensorGraphId {
    let gid = next_graph_id();
    GRAPHS.insert(gid, RwLock::new(TensorGraph::new()));
    gid
}

pub fn get_graph(gid: TensorGraphId) -> Option<Ref<'static, TensorGraphId, RwLock<TensorGraph<'static>>>> {
    GRAPHS.get(&gid)
}

pub fn get_graph_mut(gid: TensorGraphId) -> Option<RefMut<'static, TensorGraphId, RwLock<TensorGraph<'static>>>> {
    GRAPHS.get_mut(&gid)
}

// ────────────────────────────────────────────────────────────────────────────
//  Executor
// ────────────────────────────────────────────────────────────────────────────

pub struct TensorGraphExecutor<'a> {
    graph: &'a TensorGraph<'a>,
}

impl<'a> TensorGraphExecutor<'a> {
    pub fn new(graph: &'a TensorGraph<'a>) -> Self {
        Self { graph }
    }

    pub fn forward(&self) -> Result<()> {
        if self.graph.fwd_topo_order.is_empty() {
            return Err(Error::InvalidState("Forward topological order not computed".into()));
        }

        for &nid in &self.graph.fwd_topo_order {
            let node = self
                .graph
                .nodes
                .get(&nid)
                .ok_or_else(|| Error::InvalidState(format!("Node {} not found in graph", nid.0).into()))?;

            if let Some(ref fwd_fn) = node.forward_fn {
                let outputs = fwd_fn(&node.inputs[..])?;

                if outputs.len() != node.outputs.len() {
                    return Err(Error::InvalidState(
                        format!(
                            "Forward function for op '{}' returned {} outputs, expected {}",
                            node.op_name,
                            outputs.len(),
                            node.outputs.len()
                        )
                        .into(),
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn backward(&self, loss_tensor_id: TensorId) -> Result<()> {
        if self.graph.bwd_topo_order.is_empty() {
            return Err(Error::InvalidState("Backward topological order not computed".into()));
        }

        for &nid in &self.graph.bwd_topo_order {
            let node = self
                .graph
                .nodes
                .get(&nid)
                .ok_or_else(|| Error::InvalidState(format!("Node {} not found in graph", nid.0).into()))?;

            if let Some(ref bwd_fn) = node.backward_fn {
                bwd_fn((&node.inputs[..], &node.outputs[..]))?;
            }
        }

        Ok(())
    }
}
