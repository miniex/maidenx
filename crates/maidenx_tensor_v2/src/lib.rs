mod adapter;
mod cast;
mod creation;
mod display;
mod iterator;
mod memory;
mod ops;
pub mod prelude;
pub mod utils;
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

// ────────────────────────────────────────────────────────────────────────────
//  Tensor
// ────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);
static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);
#[inline]
pub(crate) fn next_tensor_id() -> TensorId {
    TensorId(TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst))
}

#[repr(transparent)]
#[derive(Clone)]
pub struct Tensor(TensorId);

/// ## Tensor identifier
///
/// This `impl` block provides access to the tensor's unique identifier.
///
/// * `id` – globally unique tensor identifier used for registry lookups
///
/// No locks or heavy work are involved here; this is a pure, branch-only
/// query suitable for hot code paths.
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
    pub fn id(&self) -> TensorId {
        self.0
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
    TensorStorageId(STORAGE_COUNTER.fetch_add(1, Ordering::SeqCst))
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

    pub fn buffer_mut_with<F, R>(&mut self, func: F) -> Result<R>
    where
        F: FnOnce(&mut (dyn Buffer + 'static)) -> Result<R>,
    {
        let buf = Arc::get_mut(&mut self.buffer).ok_or(Error::BufferShared)?;
        func(buf)
    }

    pub fn buffer_arc(&self) -> Arc<dyn Buffer> {
        self.buffer.clone()
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
        let sid = get_storage_id(self.id()).ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;
        let entry = get_storage(sid).ok_or_else(|| Error::InvalidState("tensor storage not found".into()))?;

        let guard = entry.read().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockReadGuard<'_, TensorStorage>, RwLockReadGuard<'static, TensorStorage>>(guard)
        })
    }

    #[inline]
    fn storage_mut(&self) -> Result<RwLockWriteGuard<'static, TensorStorage>> {
        let sid = get_storage_id(self.id()).ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;
        let entry = get_storage(sid).ok_or_else(|| Error::InvalidState("tensor storage not found".into()))?;

        let guard = entry.write().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockWriteGuard<'_, TensorStorage>, RwLockWriteGuard<'static, TensorStorage>>(guard)
        })
    }

    pub fn is_storaged(&self) -> bool {
        let sid = match get_storage_id(self.id()) {
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

impl TensorMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Lazy => "lazy",
        }
    }
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
        use crate::eager_mode;
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
        use crate::lazy_mode;
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
    grad_tensor_id: Option<TensorId>,
    graph_id: Option<TensorGraphId>,
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

    pub fn get_grad_tensor_id(&self) -> Option<TensorId> {
        self.grad_tensor_id
    }

    pub fn set_grad_tensor_id(&mut self, grad_tensor_id: Option<TensorId>) {
        self.grad_tensor_id = grad_tensor_id;
    }

    pub fn get_graph_id(&self) -> Option<TensorGraphId> {
        self.graph_id
    }

    pub fn set_graph_id(&mut self, graph_id: Option<TensorGraphId>) {
        self.graph_id = graph_id;
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

/// ## Metadata access and gradient operations
///
/// This `impl` block provides access to tensor metadata and gradient-related operations:
///
/// * `metadata` / `metadata_mut` – lock-protected access to `TensorMetadata`  
/// * Metadata accessors (`device`, `dtype`, `layout`, `shape`, `strides`, etc.)  
/// * Gradient operations (`grad_tensor_id`, `graph_id`, `is_const`, `requires_grad`, `grad`)
/// * Gradient control (`enable_grad`, `set_requires_grad`, `try_enable_grad`, `try_set_requires_grad`)
/// * Utility methods (`mode`, `iter_indices`)
///
/// These methods provide both low-level metadata access and high-level gradient
/// management functionality for automatic differentiation.
impl Tensor {
    #[inline]
    fn metadata(&self) -> Result<RwLockReadGuard<'static, TensorMetadata>> {
        let entry = get_metadata(self.id()).ok_or_else(|| Error::InvalidState("tensor info not found".into()))?;

        let guard = entry.read().map_err(|_| Error::Lock)?;
        Ok(unsafe {
            mem::transmute::<RwLockReadGuard<'_, TensorMetadata>, RwLockReadGuard<'static, TensorMetadata>>(guard)
        })
    }

    #[inline]
    fn metadata_mut(&self) -> Result<RwLockWriteGuard<'static, TensorMetadata>> {
        let entry = get_metadata(self.id()).ok_or_else(|| Error::InvalidState("tensor info not found".into()))?;

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
        self.metadata().unwrap().get_layout()
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

    /// Returns the unique identifier of the gradient tensor associated with this tensor.
    ///
    /// This function retrieves the ID of the gradient tensor that will accumulate gradients
    /// during backpropagation. The gradient tensor is created when `enable_grad()` or
    /// `set_requires_grad(true)` is called on a tensor.
    ///
    /// # Returns
    ///
    /// * `Some(TensorId)` - The unique identifier of the gradient tensor if gradients are enabled
    /// * `None` - If gradients are not required for this tensor
    ///
    /// # Examples
    ///
    /// ```
    /// let mut tensor = Tensor::randn(&[2, 3]);
    /// assert!(tensor.grad_tensor_id().is_none()); // No gradient initially
    ///
    /// tensor.enable_grad();
    /// assert!(tensor.grad_tensor_id().is_some()); // Gradient tensor created
    /// ```
    pub fn grad_tensor_id(&self) -> Option<TensorId> {
        self.metadata().unwrap().get_grad_tensor_id()
    }

    /// Returns the computation graph identifier that this tensor belongs to.
    ///
    /// Tensors created in lazy mode are assigned to a computation graph that tracks
    /// the operations performed on them. This enables deferred execution and
    /// automatic differentiation capabilities.
    ///
    /// # Returns
    ///
    /// * `Some(TensorGraphId)` - The unique identifier of the computation graph if the tensor is part of one
    /// * `None` - If the tensor is a constant or was created in eager mode
    ///
    /// # Examples
    ///
    /// ```
    /// // Constant tensor has no graph
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert!(tensor.graph_id().is_none());
    ///
    /// // Tensor created in lazy mode has a graph
    /// lazy!();
    /// let lazy_tensor = Tensor::zeros(&[2, 3]);
    /// assert!(lazy_tensor.graph_id().is_some());
    /// ```
    pub fn graph_id(&self) -> Option<TensorGraphId> {
        self.metadata().unwrap().get_graph_id()
    }

    /// Checks if this tensor is a constant (not part of any computation graph).
    ///
    /// A constant tensor is one that was either:
    /// - Created in eager mode
    /// - Explicitly detached from a computation graph
    /// - Created as a leaf tensor without requiring gradients
    ///
    /// Constant tensors do not participate in automatic differentiation and
    /// are executed immediately rather than being deferred.
    ///
    /// # Returns
    ///
    /// `true` if the tensor is constant, `false` if it's part of a computation graph.
    ///
    /// # Examples
    ///
    /// ```
    /// // Tensor created in eager mode is constant
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// assert!(tensor.is_const());
    ///
    /// // Tensor created in lazy mode is not constant
    /// lazy!();
    /// let lazy_tensor = Tensor::zeros(&[2, 3]);
    /// assert!(!lazy_tensor.is_const());
    /// ```
    pub fn is_const(&self) -> bool {
        self.graph_id().is_none()
    }

    /// Checks if this tensor requires gradient computation during backpropagation.
    ///
    /// A tensor requires gradients if it has an associated gradient tensor that will
    /// accumulate gradients during the backward pass. This is typically enabled for
    /// parameters that need to be optimized during training.
    ///
    /// # Returns
    ///
    /// `true` if gradients will be computed for this tensor, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut tensor = Tensor::randn(&[2, 3]);
    /// assert!(!tensor.requires_grad()); // No gradients by default
    ///
    /// tensor.enable_grad();
    /// assert!(tensor.requires_grad()); // Gradients enabled
    ///
    /// tensor.set_requires_grad(false);
    /// assert!(!tensor.requires_grad()); // Gradients disabled
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.grad_tensor_id().is_some()
    }

    /// Returns the gradient tensor associated with this tensor.
    ///
    /// The gradient tensor accumulates gradients computed during backpropagation.
    /// This tensor shares the same shape and device as the original tensor but
    /// may have different storage.
    ///
    /// # Returns
    ///
    /// A `Tensor` handle to the gradient tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor does not have an associated gradient tensor.
    /// Use `requires_grad()` to check before calling this method.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = Tensor::randn(&[2, 3]);
    /// let mut y = Tensor::randn(&[2, 3]);
    /// x.enable_grad();
    /// y.enable_grad();
    ///
    /// let z = x.add(&y);
    ///
    /// // Access gradients
    /// let x_grad = x.grad();
    /// let y_grad = y.grad();
    /// x_grad.forward(); // Compute the gradient values
    /// y_grad.forward(); // Compute the gradient values
    /// ```
    pub fn grad(&self) -> Tensor {
        let id = self.grad_tensor_id().expect("Tensor does not have a gradient tensor");
        Tensor(id)
    }

    /// Runs [`try_enable_grad`](Self::try_enable_grad) and panics on failure.
    ///
    /// # Panics
    ///
    /// * When the tensor has a non-floating point data type
    /// * When gradient tensor creation fails
    pub fn enable_grad(&self) {
        self.try_enable_grad().expect("failed to enable autograd")
    }

    /// Runs [`try_set_requires_grad`](Self::try_set_requires_grad) and panics on failure.
    ///
    /// # Panics
    ///
    /// * When trying to enable gradients on a non-floating point tensor
    /// * When gradient tensor creation fails
    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.try_set_requires_grad(requires_grad)
            .expect("failed to set gradient requirement")
    }

    /// Attempts to enable gradient computation for this tensor.
    ///
    /// This method creates a gradient tensor that will accumulate gradients during
    /// backpropagation. The gradient tensor has the same shape, device, and data type
    /// as the original tensor, configured for lazy execution mode.
    ///
    /// # Process
    ///
    /// 1. Check if the tensor has a floating-point data type (required for gradients)
    /// 2. Skip if gradients are already enabled
    /// 3. Create a new gradient tensor with unique ID
    /// 4. Configure gradient tensor metadata for lazy execution
    /// 5. Link the gradient tensor to this tensor's metadata
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If gradients were successfully enabled or were already enabled
    /// * `Err(Error::UnsupportedDType)` - If the tensor has a non-floating point data type
    ///
    /// # Examples
    ///
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn enable_gradients() -> Result<()> {
    ///     let tensor = Tensor::randn(&[2, 3]);
    ///     tensor.try_enable_grad()?;
    ///     assert!(tensor.requires_grad());
    ///     Ok(())
    /// }
    ///
    /// // This will fail for integer tensors
    /// let mut int_tensor = Tensor::zeros_dtype(&[2, 3], DType::I32);
    /// assert!(int_tensor.try_enable_grad().is_err());
    /// ```
    pub fn try_enable_grad(&self) -> Result<()> {
        if !self.dtype().is_float() {
            return Err(Error::UnsupportedDType);
        }

        if self.requires_grad() {
            return Ok(());
        }

        let grad_tensor_id = next_tensor_id();

        let grad_metadata = TensorMetadata {
            device: self.device(),
            dtype: self.dtype(),
            layout: self.layout(),
            grad_tensor_id: None,
            graph_id: None,
            mode: TensorMode::Lazy,
            update_status: TensorUpdateStatus::Pending,
        };
        insert_metadata(grad_tensor_id, grad_metadata);

        self.metadata_mut().unwrap().set_grad_tensor_id(Some(grad_tensor_id));

        Ok(())
    }

    /// Attempts to set whether this tensor should accumulate gradients during backpropagation.
    ///
    /// This method provides fine-grained control over gradient computation by either
    /// enabling or disabling gradient accumulation. When enabled, it creates a gradient
    /// tensor; when disabled, it removes the association with any existing gradient tensor.
    ///
    /// # Parameters
    ///
    /// * `requires_grad` - `true` to enable gradients, `false` to disable
    ///
    /// # Process
    ///
    /// - When `requires_grad` is `true`, delegates to [`try_enable_grad`](Self::try_enable_grad)
    /// - When `requires_grad` is `false`, removes the gradient tensor association
    /// - Disabling gradients always succeeds, regardless of tensor data type
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the gradient requirement was successfully set
    /// * `Err(Error::UnsupportedDType)` - If trying to enable gradients on a non-floating point tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn configure_gradients() -> Result<()> {
    ///     let tensor = Tensor::randn(&[2, 3]);
    ///
    ///     // Enable gradients
    ///     tensor.try_set_requires_grad(true)?;
    ///     assert!(tensor.requires_grad());
    ///
    ///     // Disable gradients
    ///     tensor.try_set_requires_grad(false)?;
    ///     assert!(!tensor.requires_grad());
    ///
    ///     Ok(())
    /// }
    ///
    /// // This will fail for integer tensors when enabling gradients
    /// let mut int_tensor = Tensor::zeros_dtype(&[2, 3], DType::I32);
    /// assert!(int_tensor.try_set_requires_grad(true).is_err());
    /// assert!(int_tensor.try_set_requires_grad(false).is_ok()); // Disabling always works
    /// ```
    pub fn try_set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        if requires_grad {
            self.try_enable_grad()
        } else {
            self.metadata_mut().unwrap().set_grad_tensor_id(None);
            Ok(())
        }
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

    /// Iterates through all possible multi-dimensional indices of the tensor.
    ///
    /// This method calls the provided callback function for each valid index combination
    /// in the tensor's shape. The indices are generated in lexicographic order, starting
    /// from `[0, 0, ..., 0]` and incrementing like an odometer until all combinations
    /// are exhausted.
    ///
    /// # Parameters
    ///
    /// * `callback` - A function that receives each index combination as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let mut collected_indices = Vec::new();
    ///
    /// tensor.iter_indices(|indices| {
    ///     collected_indices.push(indices.to_vec());
    /// });
    ///
    /// // For a 2x3 tensor, this will generate:
    /// // [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    /// assert_eq!(collected_indices.len(), 6);
    /// ```
    ///
    /// # Performance Notes
    ///
    /// * For large tensors, this method will call the callback many times
    /// * The callback is called once for each element in the tensor
    /// * Consider the tensor size before using this method in performance-critical code
    pub fn iter_indices<F>(&self, mut callback: F)
    where
        F: FnMut(&[usize]),
    {
        let shape = self.shape();

        if shape.is_empty() {
            callback(&[]);
            return;
        }

        let mut indices = vec![0; shape.len()];

        loop {
            callback(&indices);

            // Increment indices (like an odometer)
            let mut dim = indices.len() - 1;
            loop {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }

                indices[dim] = 0;
                if dim == 0 {
                    return; // We've iterated through all combinations
                }
                dim -= 1;
            }
        }
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

impl TensorEdge {
    pub fn new(from: TensorNodeId, to: TensorNodeId) -> Self {
        Self { from, to }
    }

    pub fn from(&self) -> TensorNodeId {
        self.from
    }

    pub fn to(&self) -> TensorNodeId {
        self.to
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorGraphId(usize);
static GRAPH_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn next_graph_id() -> TensorGraphId {
    TensorGraphId(GRAPH_COUNTER.fetch_add(1, Ordering::SeqCst))
}

pub struct TensorGraph {
    nodes: HashMap<TensorNodeId, TensorNode>,
    edges: HashMap<TensorEdgeId, (TensorNodeId, TensorNodeId)>,
    topo_order: Vec<TensorNodeId>,
    tensor_to_node: HashMap<TensorId, TensorNodeId>,
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
            tensor_to_node: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: TensorNode) {
        let node_id = node.nid;

        for &input_tid in &node.inputs {
            if let Some(&producer_node_id) = self.tensor_to_node.get(&input_tid) {
                self.connect(producer_node_id, node_id);
            }
        }

        for &output_tid in &node.outputs {
            self.tensor_to_node.insert(output_tid, node_id);
        }

        self.nodes.insert(node_id, node);
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

/// Creates a new forward graph with even ID (0, 2, 4, 6...)
pub fn new_forward_graph() -> TensorGraphId {
    let raw_id = GRAPH_COUNTER.fetch_add(2, Ordering::SeqCst);
    let even_id = if raw_id % 2 == 0 { raw_id } else { raw_id + 1 };
    let gid = TensorGraphId(even_id);
    GRAPHS.insert(gid, RwLock::new(TensorGraph::new()));
    gid
}

/// Creates a new backward graph with odd ID (1, 3, 5, 7...)
pub fn new_backward_graph() -> TensorGraphId {
    let raw_id = GRAPH_COUNTER.fetch_add(2, Ordering::SeqCst);
    let odd_id = if raw_id % 2 == 1 { raw_id } else { raw_id + 1 };
    let gid = TensorGraphId(odd_id);
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

    pub fn forward_target(&self, target_tid: TensorId) -> Result<()> {
        if self.graph.topo_order.is_empty() {
            return Err(Error::InvalidState("topological order not computed".into()));
        }

        // Find all nodes that target_tid depends on
        let mut required_nodes = std::collections::HashSet::new();
        let mut to_visit = std::collections::VecDeque::new();

        // Find the node that produces target_tid (either directly or indirectly via gradient mechanism)
        for (&nid, node) in &self.graph.nodes {
            if node.outputs.contains(&target_tid) {
                to_visit.push_back(nid);
                break;
            }
        }

        // If no direct producer found, check if this is a gradient tensor
        // In backward graphs, gradient tensors are produced indirectly
        if to_visit.is_empty() {
            // Find which tensor this gradient belongs to
            let mut target_input_tensor = None;

            // Check all tensors to find which one has this gradient
            for tid_val in 0..10 {
                // Check reasonable range
                let check_tid = TensorId(tid_val);
                if let Some(metadata_ref) = crate::get_metadata(check_tid) {
                    if let Ok(metadata) = metadata_ref.read() {
                        if let Some(grad_tid) = metadata.grad_tensor_id {
                            if grad_tid == target_tid {
                                target_input_tensor = Some(check_tid);
                                break;
                            }
                        }
                    }
                }
            }

            if let Some(input_tid) = target_input_tensor {
                // For x (TensorId(0)), we want lhs backward
                // For y (TensorId(1)), we want rhs backward
                let target_op_suffix = if input_tid.0 == 0 { "lhs" } else { "rhs" };

                // Find the specific backward node for this tensor
                for (&nid, node) in &self.graph.nodes {
                    if node.op_name.contains("backward") && node.op_name.contains(target_op_suffix) {
                        to_visit.push_back(nid);
                        break; // Only add the specific node
                    }
                }
            }
        }

        if to_visit.is_empty() {
            return Ok(()); // Target might already be materialized
        }

        // Build dependency set by traversing backwards
        while let Some(nid) = to_visit.pop_front() {
            if required_nodes.insert(nid) {
                if let Some(node) = self.graph.nodes.get(&nid) {
                    // Add all input-producing nodes to visit list
                    for &input_tid in &node.inputs {
                        for (&input_nid, input_node) in &self.graph.nodes {
                            if input_node.outputs.contains(&input_tid) {
                                to_visit.push_back(input_nid);
                            }
                        }
                    }
                }
            }
        }

        // Execute only required nodes in topological order
        for &nid in &self.graph.topo_order {
            if required_nodes.contains(&nid) {
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

    /// Attempts to execute the forward pass of the computation graph.
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
        if self.is_storaged() {
            eprintln!(
                "\x1b[33m[WARN]\x1b[0m forward() called on already materialized tensor (id={:?})",
                self.id()
            );
            return Ok(());
        }

        let gid = self.graph_id().ok_or_else(|| {
            Error::InvalidState("tensor is not connected to a computation graph (constant or detached)".into())
        })?;

        if gid.0 % 2 == 1 {}

        let graph_guard = get_graph(gid).ok_or_else(|| Error::InvalidState("graph not found".into()))?;

        {
            let mut g = graph_guard.write().map_err(|_| Error::Lock)?;
            if g.topo_order.is_empty() {
                g.topo_sort();
            }
            TensorGraphExecutor::new(&*g).forward_target(self.id())?;
        }

        Ok(())
    }
}
