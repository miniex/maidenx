use crate::{
    adapter::TensorAdapter, get_mode, get_storage_id, insert_metadata, insert_storage, link_tensor_to_storage,
    next_storage_id, next_tensor_id, Tensor, TensorId, TensorMetadata, TensorStorage, TensorUpdateStatus,
};
use half::{bf16, f16};
use maidenx_core::{
    buffer::BufferManager,
    device::{get_default_device, Device},
    dtype::{get_default_dtype, DType},
    error::{Error, Result},
    layout::Layout,
    scalar::Scalar,
};
use rand::prelude::*;
use std::sync::Arc;

/// ## Factory & initialization helpers
///
/// This `impl` block contains *constructor-style* convenience functions that
/// allocate new tensors or create shallow views of existing storage.
///
/// * Generic constructors: `new`, `new_with_spec`, `from_flatten_vec`, `from_flatten_vec_with_spec`, `share`  
/// * Pattern initializers: `zeros`, `ones`, `fill`, `randn`, `range`, `arange`  
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).  
/// * **All tensors produced here are considered _constant tensors_:** they are
///   **not** attached to any computation graph (`gid == None`) and therefore
///   do *not* require gradients unless explicitly modified later.
///
/// These methods allocate (or re-use) buffers; they never mutate an existing
/// tensor in place.
impl Tensor {
    /// Runs [`try_new`](Self::try_new) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensor creation fails
    pub fn new<T>(data: T) -> Self
    where
        T: TensorAdapter,
    {
        Self::try_new(data).expect("failed to create tensor")
    }

    /// Runs [`try_new_with_spec`](Self::try_new_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let tensor = Tensor::new_with_spec(&[1.0, 2.0], Device::CPU, DType::F32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensor creation fails
    pub fn new_with_spec<T>(data: T, device: Device, dtype: DType) -> Self
    where
        T: TensorAdapter,
    {
        Self::try_new_with_spec(data, device, dtype).expect("failed to create tensor with specified device and dtype")
    }

    /// Runs [`try_from_flatten_vec`](Self::try_from_flatten_vec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::from_flatten_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensor creation fails
    pub fn from_flatten_vec<T>(data: T, shape: &[usize]) -> Self
    where
        T: TensorAdapter,
    {
        Self::try_from_flatten_vec(data, shape).expect("failed to create tensor from flatten vec")
    }

    /// Runs [`try_from_flatten_vec_with_spec`](Self::try_from_flatten_vec_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let tensor = Tensor::from_flatten_vec_with_spec(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU, DType::F32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When tensor creation fails
    pub fn from_flatten_vec_with_spec<T>(data: T, shape: &[usize], device: Device, dtype: DType) -> Self
    where
        T: TensorAdapter,
    {
        Self::try_from_flatten_vec_with_spec(data, shape, device, dtype)
            .expect("failed to create tensor from flatten vec with specified device and dtype")
    }

    /// Runs [`try_share`](Self::try_share) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::new(&[1, 2, 3, 4]);
    /// let b = Tensor::share(&a);   // b shares the same storage
    /// ```
    ///
    /// # Panics
    ///
    /// * When the target tensor’s storage cannot be located
    pub fn share(target: &Tensor) -> Self {
        Self::try_share(target).expect("failed to create shared tensor")
    }

    /// Runs [`try_empty`](Self::try_empty) and panics on failure.
    ///
    /// Creates a tensor with uninitialized memory. The contents are undefined
    /// and may contain arbitrary values. Use this for performance when you
    /// plan to immediately overwrite all values.
    ///
    /// # Examples
    /// ```
    /// let e = Tensor::empty(&[2, 3]);
    /// // Contents are undefined - must be initialized before use
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn empty(shape: &[usize]) -> Self {
        Self::try_empty(shape).expect("failed to create empty tensor")
    }

    /// Runs [`try_empty_like`](Self::try_empty_like) and panics on failure.
    ///
    /// Creates a tensor with the same shape, device, and dtype as the source tensor,
    /// but with uninitialized memory. The contents are undefined.
    ///
    /// # Examples
    /// ```
    /// let src = Tensor::new(&[1, 2, 3, 4]);
    /// let e = Tensor::empty_like(&src);
    /// assert_eq!(e.shape(), src.shape());
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn empty_like(src: &Tensor) -> Self {
        Self::try_empty_like(src).expect("failed to create empty_like tensor")
    }

    /// Runs [`try_empty_with_spec`](Self::try_empty_with_spec) and panics on failure.
    ///
    /// Creates a tensor with uninitialized memory using the specified device and dtype.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let e = Tensor::empty_with_spec(&[2, 3], Device::CPU, DType::F32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn empty_with_spec(shape: &[usize], device: Device, dtype: DType) -> Self {
        Self::try_empty_with_spec(shape, device, dtype)
            .expect("failed to create empty tensor with specified device and dtype")
    }

    /// Runs [`try_zeros`](Self::try_zeros) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let z = Tensor::zeros(&[2, 3]);
    /// assert!(z.iter::<f32>().all(|&x| x == 0.0));
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn zeros(shape: &[usize]) -> Self {
        Self::try_zeros(shape).expect("failed to create zeros tensor")
    }

    /// Runs [`try_zeros_like`](Self::try_zeros_like) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let src = Tensor::new(&[1, 2, 3, 4]);
    /// let z   = Tensor::zeros_like(&src);
    /// assert_eq!(z.shape(), src.shape());
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn zeros_like(src: &Tensor) -> Self {
        Self::try_zeros_like(src).expect("failed to create zeros_like tensor")
    }

    /// Runs [`try_zeros_with_spec`](Self::try_zeros_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let z = Tensor::zeros_with_spec(&[2, 3], Device::CPU, DType::F32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn zeros_with_spec(shape: &[usize], device: Device, dtype: DType) -> Self {
        Self::try_zeros_with_spec(shape, device, dtype)
            .expect("failed to create zeros tensor with specified device and dtype")
    }

    /// Runs [`try_ones`](Self::try_ones) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let o = Tensor::ones(&[2, 3]);
    /// assert!(o.iter::<f32>().all(|&x| x == 1.0));
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn ones(shape: &[usize]) -> Self {
        Self::try_ones(shape).expect("failed to create ones tensor")
    }

    /// Runs [`try_ones_like`](Self::try_ones_like) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let src = Tensor::new(&[1, 2, 3, 4]);
    /// let o   = Tensor::ones_like(&src);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn ones_like(src: &Tensor) -> Self {
        Self::try_ones_like(src).expect("failed to create ones_like tensor")
    }

    /// Runs [`try_ones_with_spec`](Self::try_ones_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let o = Tensor::ones_with_spec(&[2, 3], Device::CPU, DType::F32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn ones_with_spec(shape: &[usize], device: Device, dtype: DType) -> Self {
        Self::try_ones_with_spec(shape, device, dtype)
            .expect("failed to create ones tensor with specified device and dtype")
    }

    /// Runs [`try_fill`](Self::try_fill) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let t = Tensor::fill(&[2, 3], 42);
    /// assert!(t.iter::<i32>().all(|&x| x == 42));
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn fill<T: Into<Scalar>>(shape: &[usize], value: T) -> Self {
        Self::try_fill(shape, value).expect("failed to create filled tensor")
    }

    /// Runs [`try_fill_like`](Self::try_fill_like) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let src = Tensor::new(&[1, 2, 3, 4]);
    /// let t   = Tensor::fill_like(&src, 7u8);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn fill_like<T: Into<Scalar>>(src: &Tensor, value: T) -> Self {
        Self::try_fill_like(src, value).expect("failed to create fill_like tensor")
    }

    /// Runs [`try_fill_with_spec`](Self::try_fill_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let t = Tensor::fill_with_spec(&[2, 3], 1.23, Device::CPU, DType::F32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn fill_with_spec<T: Into<Scalar>>(shape: &[usize], value: T, device: Device, dtype: DType) -> Self {
        Self::try_fill_with_spec(shape, value, device, dtype)
            .expect("failed to create filled tensor with specified device and dtype")
    }

    /// Runs [`try_randn`](Self::try_randn) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let r = Tensor::randn(&[2, 3]);   // N(0, 1) samples
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails or RNG setup fails
    pub fn randn(shape: &[usize]) -> Self {
        Self::try_randn(shape).expect("failed to create random normal tensor")
    }

    /// Runs [`try_randn_like`](Self::try_randn_like) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let src = Tensor::new(&[0.0; 10]);
    /// let r   = Tensor::randn_like(&src);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails or RNG setup fails
    pub fn randn_like(src: &Tensor) -> Self {
        Self::try_randn_like(src).expect("failed to create random normal tensor like source")
    }

    /// Runs [`try_randn_with_spec`](Self::try_randn_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let r = Tensor::randn_with_spec(&[4, 4], Device::CPU, DType::F64);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails or RNG setup fails
    pub fn randn_with_spec(shape: &[usize], device: Device, dtype: DType) -> Self {
        Self::try_randn_with_spec(shape, device, dtype)
            .expect("failed to create random normal tensor with specified device and dtype")
    }

    /// Runs [`try_range`](Self::try_range) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let r = Tensor::range(5);   // 0‥4
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn range(n: usize) -> Self {
        Self::try_range(n).expect("failed to create range tensor")
    }

    /// Runs [`try_range_with_spec`](Self::try_range_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let r = Tensor::range_with_spec(5, Device::CPU, DType::I32);
    /// ```
    ///
    /// # Panics
    ///
    /// * When buffer allocation fails
    pub fn range_with_spec(n: usize, device: Device, dtype: DType) -> Self {
        Self::try_range_with_spec(n, device, dtype)
            .expect("failed to create range tensor with specified device and dtype")
    }

    /// Runs [`try_arange`](Self::try_arange) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// let v = Tensor::arange(0, 10, 2);   // 0, 2, 4, 6, 8
    /// ```
    ///
    /// # Panics
    ///
    /// * When `step == 0` or buffer allocation fails
    pub fn arange<T>(start: T, end: T, step: T) -> Self
    where
        T: Into<Scalar> + Copy,
    {
        Self::try_arange(start, end, step).expect("failed to create arange tensor")
    }

    /// Runs [`try_arange_with_spec`](Self::try_arange_with_spec) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType};
    ///
    /// let v = Tensor::arange_with_spec(1, 5, 1, Device::CPU, DType::U8); // 1,2,3,4
    /// ```
    ///
    /// # Panics
    ///
    /// * When `step == 0` or buffer allocation fails
    pub fn arange_with_spec<T>(start: T, end: T, step: T, device: Device, dtype: DType) -> Self
    where
        T: Into<Scalar> + Copy,
    {
        Self::try_arange_with_spec(start, end, step, device, dtype)
            .expect("failed to create arange tensor with specified device and dtype")
    }

    /// Attempts to create a new tensor from the given data.
    ///
    /// Uses the default device and data type inferred from the input.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_tensor() -> Result<Tensor> {
    ///     let tensor = Tensor::try_new(&[1.0, 2.0, 3.0, 4.0])?;
    ///     Ok(tensor)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer allocation fails
    /// - Data conversion fails
    /// - The buffer is shared and cannot be modified
    pub fn try_new<T>(data: T) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let device = get_default_device();
        let dtype = data.dtype();
        Self::try_new_with_spec(data, device, dtype)
    }

    /// Attempts to create a new tensor with the specified device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_tensor_with_spec() -> Result<Tensor> {
    ///     let tensor = Tensor::try_new_with_spec(&[1.0, 2.0], Device::CPU, DType::F32)?;
    ///     Ok(tensor)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer allocation fails
    /// - Data conversion fails
    /// - The buffer is shared and cannot be modified
    pub fn try_new_with_spec<T>(data: T, device: Device, dtype: DType) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let shape = data.get_shape();
        let layout = Layout::from_shape(&shape);
        let size = layout.size();

        let src_dtype = data.dtype();
        let src_data = data.to_flatten_vec()?;

        let mut buffer = BufferManager::create(size, device, dtype)?;

        {
            if src_dtype == dtype {
                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(
                        src_data.as_ptr() as *const std::ffi::c_void,
                        size * dtype.size_in_bytes(),
                        0,
                        0,
                    )?;
                }
            } else {
                let mut converted_data = vec![0u8; size * dtype.size_in_bytes()];

                for i in 0..src_data.len() {
                    let scalar = unsafe {
                        src_dtype.read_scalar((src_data.as_ptr() as *const u8).add(i * src_dtype.size_in_bytes()))
                    };

                    unsafe {
                        dtype.write_scalar(converted_data.as_mut_ptr().add(i * dtype.size_in_bytes()), scalar);
                    }
                }

                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(
                        converted_data.as_ptr() as *const std::ffi::c_void,
                        size * dtype.size_in_bytes(),
                        0,
                        0,
                    )?;
                }
            }
        }

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor from flattened data with the specified shape.
    ///
    /// Uses the default device and data type inferred from the input.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_tensor_from_flatten() -> Result<Tensor> {
    ///     let tensor = Tensor::try_from_flatten_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    ///     Ok(tensor)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer allocation fails
    /// - Data conversion fails
    /// - The buffer is shared and cannot be modified
    /// - The shape is incompatible with the data length
    pub fn try_from_flatten_vec<T>(data: T, shape: &[usize]) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let device = get_default_device();
        let dtype = data.dtype();
        Self::try_from_flatten_vec_with_spec(data, shape, device, dtype)
    }

    /// Attempts to create a new tensor from flattened data with the specified shape, device, and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_tensor_from_flatten_with_spec() -> Result<Tensor> {
    ///     let tensor = Tensor::try_from_flatten_vec_with_spec(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU, DType::F32)?;
    ///     Ok(tensor)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer allocation fails
    /// - Data conversion fails
    /// - The buffer is shared and cannot be modified
    /// - The shape is incompatible with the data length (size != src_data.len())
    pub fn try_from_flatten_vec_with_spec<T>(data: T, shape: &[usize], device: Device, dtype: DType) -> Result<Self>
    where
        T: TensorAdapter,
    {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let src_dtype = data.dtype();
        let src_data = data.to_flatten_vec()?;

        if size != src_data.len() {
            return Err(Error::InvalidShape {
                message: format!(
                    "Shape mismatch: shape implies {} elements but data has {} elements",
                    size,
                    src_data.len()
                ),
            });
        }

        let mut buffer = BufferManager::create(size, device, dtype)?;

        {
            if src_dtype == dtype {
                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(
                        src_data.as_ptr() as *const std::ffi::c_void,
                        size * dtype.size_in_bytes(),
                        0,
                        0,
                    )?;
                }
            } else {
                let mut converted_data = vec![0u8; size * dtype.size_in_bytes()];

                for i in 0..src_data.len() {
                    let scalar = unsafe {
                        src_dtype.read_scalar((src_data.as_ptr() as *const u8).add(i * src_dtype.size_in_bytes()))
                    };

                    unsafe {
                        dtype.write_scalar(converted_data.as_mut_ptr().add(i * dtype.size_in_bytes()), scalar);
                    }
                }

                unsafe {
                    let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                    buffer_mut.copy_from_host(
                        converted_data.as_ptr() as *const std::ffi::c_void,
                        size * dtype.size_in_bytes(),
                        0,
                        0,
                    )?;
                }
            }
        }

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor that shares storage with the target tensor.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn share_tensor() -> Result<Tensor> {
    ///     let tensor1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let tensor2 = Tensor::try_share(&tensor1)?;
    ///     Ok(tensor2)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the target tensor's storage information cannot be found.
    pub fn try_share(target: &Tensor) -> Result<Self> {
        let tid = next_tensor_id();
        let sid =
            get_storage_id(target.tid()).ok_or_else(|| Error::InvalidState("tensor storage id not found".into()))?;
        link_tensor_to_storage(tid, sid);
        let metadata = target.metadata()?.clone();
        insert_metadata(tid, metadata);

        Ok(Tensor {
            tid,
            gtid: TensorId(0),
            gid: target.gid(),
        })
    }

    /// Attempts to create a new tensor with uninitialized memory.
    ///
    /// Uses the default device and data type. The tensor contents are undefined
    /// and may contain arbitrary values from previous memory usage. This is the
    /// fastest way to allocate a tensor when you plan to immediately overwrite
    /// all values.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_empty() -> Result<Tensor> {
    ///     let empty = Tensor::try_empty(&[2, 3])?;
    ///     // empty.contents are undefined - initialize before use
    ///     Ok(empty)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    ///
    /// # Performance Notes
    ///
    /// This is the fastest tensor creation method as it skips initialization.
    /// Use when performance is critical and you will immediately fill the tensor.
    pub fn try_empty(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::try_empty_with_spec(shape, device, dtype)
    }

    /// Attempts to create a new tensor with the same shape as the source tensor, but with uninitialized memory.
    ///
    /// The new tensor inherits the device and dtype from the source tensor.
    /// Contents are undefined and may contain arbitrary values.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_empty_like() -> Result<Tensor> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let empty = Tensor::try_empty_like(&tensor)?;
    ///     assert_eq!(empty.shape(), tensor.shape());
    ///     Ok(empty)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_empty_like(src: &Tensor) -> Result<Self> {
        Self::try_empty_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    /// Attempts to create a new tensor with uninitialized memory using the specified device and dtype.
    ///
    /// The tensor contents are undefined and may contain arbitrary values.
    /// This provides maximum control over tensor allocation while maintaining
    /// peak performance by avoiding initialization overhead.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_empty_with_spec() -> Result<Tensor> {
    ///     let empty = Tensor::try_empty_with_spec(&[2, 3], Device::CPU, DType::F32)?;
    ///     // empty.contents are undefined - initialize before use
    ///     Ok(empty)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    ///
    /// # Safety Notes
    ///
    /// The returned tensor contains uninitialized memory. Reading from it
    /// before writing will yield undefined values. Always initialize the
    /// tensor before performing computations.
    pub fn try_empty_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let buffer = BufferManager::create(size, device, dtype)?;

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor filled with zeros.
    ///
    /// Uses the default device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_zeros() -> Result<Tensor> {
    ///     let zeros = Tensor::try_zeros(&[2, 3])?;
    ///     Ok(zeros)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_zeros(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::try_zeros_with_spec(shape, device, dtype)
    }

    /// Attempts to create a new tensor with the same shape as the source tensor, filled with zeros.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_zeros_like() -> Result<Tensor> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let zeros = Tensor::try_zeros_like(&tensor)?;
    ///     Ok(zeros)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_zeros_like(src: &Tensor) -> Result<Self> {
        Self::try_zeros_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    /// Attempts to create a new tensor filled with zeros with the specified shape, device, and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_zeros_with_spec() -> Result<Tensor> {
    ///     let zeros = Tensor::try_zeros_with_spec(&[2, 3], Device::CPU, DType::F32)?;
    ///     Ok(zeros)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_zeros_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let mut buffer = BufferManager::create(size, device, dtype)?;

        let elem_size = dtype.size_in_bytes();
        let total_bytes = size * elem_size;
        let zero_buf = vec![0u8; total_bytes];

        {
            unsafe {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                buffer_mut.copy_from_host(
                    zero_buf.as_ptr() as *const std::ffi::c_void,
                    size * dtype.size_in_bytes(),
                    0,
                    0,
                )?;
            }
        }

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor filled with ones.
    ///
    /// Uses the default device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_ones() -> Result<Tensor> {
    ///     let ones = Tensor::try_ones(&[2, 3])?;
    ///     Ok(ones)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_ones(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::try_ones_with_spec(shape, device, dtype)
    }

    /// Attempts to create a new tensor with the same shape as the source tensor, filled with ones.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_ones_like() -> Result<Tensor> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let ones = Tensor::try_ones_like(&tensor)?;
    ///     Ok(ones)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_ones_like(src: &Tensor) -> Result<Self> {
        Self::try_ones_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    /// Attempts to create a new tensor filled with ones with the specified shape, device, and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_ones_with_spec() -> Result<Tensor> {
    ///     let ones = Tensor::try_ones_with_spec(&[2, 3], Device::CPU, DType::F32)?;
    ///     Ok(ones)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_ones_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();

        let mut buffer = BufferManager::create(size, device, dtype)?;

        let one_bytes = match dtype {
            DType::BF16 => bf16::ONE.to_ne_bytes().to_vec(),
            DType::F16 => f16::ONE.to_ne_bytes().to_vec(),
            DType::F32 => 1.0f32.to_ne_bytes().to_vec(),
            DType::F64 => 1.0f64.to_ne_bytes().to_vec(),
            DType::BOOL => vec![1u8],
            DType::U8 => vec![1u8],
            DType::U16 => 1u16.to_ne_bytes().to_vec(),
            DType::U32 => 1u32.to_ne_bytes().to_vec(),
            DType::U64 => 1u64.to_ne_bytes().to_vec(),
            DType::I8 => 1i8.to_ne_bytes().to_vec(),
            DType::I16 => 1i16.to_ne_bytes().to_vec(),
            DType::I32 => 1i32.to_ne_bytes().to_vec(),
            DType::I64 => 1i64.to_ne_bytes().to_vec(),
        };
        let elem_size = dtype.size_in_bytes();
        let total_bytes = size * elem_size;

        let mut host_buf = Vec::with_capacity(total_bytes);
        for _ in 0..size {
            host_buf.extend_from_slice(&one_bytes);
        }

        {
            unsafe {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                buffer_mut.copy_from_host(
                    host_buf.as_ptr() as *const std::ffi::c_void,
                    size * dtype.size_in_bytes(),
                    0,
                    0,
                )?;
            }
        }

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor filled with a specific value.
    ///
    /// Uses the default device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_filled() -> Result<Tensor> {
    ///     let filled = Tensor::try_fill(&[2, 3], 42.0)?;
    ///     Ok(filled)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_fill<T: Into<Scalar>>(shape: &[usize], value: T) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::try_fill_with_spec(shape, value, device, dtype)
    }

    /// Attempts to create a new tensor with the same shape as the source tensor, filled with a specific value.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_filled_like() -> Result<Tensor> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let filled = Tensor::try_fill_like(&tensor, 42.0)?;
    ///     Ok(filled)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_fill_like<T: Into<Scalar>>(src: &Tensor, value: T) -> Result<Self> {
        Self::try_fill_with_spec(src.layout().shape(), value, src.device(), src.dtype())
    }

    /// Attempts to create a new tensor filled with a specific value with the specified shape, device, and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_filled_with_spec() -> Result<Tensor> {
    ///     let filled = Tensor::try_fill_with_spec(&[2, 3], 42.0, Device::CPU, DType::F32)?;
    ///     Ok(filled)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_fill_with_spec<T: Into<Scalar>>(
        shape: &[usize],
        value: T,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();
        let scalar_value = value.into();

        let mut buffer = BufferManager::create(size, device, dtype)?;

        let value_bytes = match dtype {
            DType::BF16 => bf16::from_f32(scalar_value.as_f32()).to_ne_bytes().to_vec(),
            DType::F16 => f16::from_f32(scalar_value.as_f32()).to_ne_bytes().to_vec(),
            DType::F32 => scalar_value.as_f32().to_ne_bytes().to_vec(),
            DType::F64 => scalar_value.as_f64().to_ne_bytes().to_vec(),
            DType::BOOL => vec![if scalar_value.as_bool() { 1u8 } else { 0u8 }],
            DType::U8 => (scalar_value.as_u32() as u8).to_ne_bytes().to_vec(),
            DType::U16 => scalar_value.as_u16().to_ne_bytes().to_vec(),
            DType::U32 => scalar_value.as_u32().to_ne_bytes().to_vec(),
            DType::U64 => scalar_value.as_u64().to_ne_bytes().to_vec(),
            DType::I8 => (scalar_value.as_i32() as i8).to_ne_bytes().to_vec(),
            DType::I16 => scalar_value.as_i16().to_ne_bytes().to_vec(),
            DType::I32 => scalar_value.as_i32().to_ne_bytes().to_vec(),
            DType::I64 => scalar_value.as_i64().to_ne_bytes().to_vec(),
        };

        let elem_size = dtype.size_in_bytes();
        let mut host_buf = Vec::with_capacity(size * elem_size);

        for _ in 0..size {
            host_buf.extend_from_slice(&value_bytes);
        }

        {
            unsafe {
                let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
                buffer_mut.copy_from_host(
                    host_buf.as_ptr() as *const std::ffi::c_void,
                    size * dtype.size_in_bytes(),
                    0,
                    0,
                )?;
            }
        }

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor filled with random values from a standard normal distribution.
    ///
    /// Uses the default device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_random() -> Result<Tensor> {
    ///     let random = Tensor::try_randn(&[2, 3])?;
    ///     Ok(random)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails or if the normal distribution cannot be created.
    pub fn try_randn(shape: &[usize]) -> Result<Self> {
        let device = get_default_device();
        let dtype = get_default_dtype();

        Self::try_randn_with_spec(shape, device, dtype)
    }

    /// Attempts to create a new tensor with the same shape as the source tensor,
    /// filled with random values from a standard normal distribution.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_random_like() -> Result<Tensor> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let random = Tensor::try_randn_like(&tensor)?;
    ///     Ok(random)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails or if the normal distribution cannot be created.
    pub fn try_randn_like(src: &Tensor) -> Result<Self> {
        Self::try_randn_with_spec(src.layout().shape(), src.device(), src.dtype())
    }

    /// Attempts to create a new tensor filled with random values from a standard normal distribution
    /// with the specified shape, device, and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_random_with_spec() -> Result<Tensor> {
    ///     let random = Tensor::try_randn_with_spec(&[2, 3], Device::CPU, DType::F32)?;
    ///     Ok(random)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails or if the normal distribution cannot be created.
    pub fn try_randn_with_spec(shape: &[usize], device: Device, dtype: DType) -> Result<Self> {
        let layout = Layout::from_shape(shape);
        let size = layout.size();
        let mut buffer = BufferManager::create(size, device, dtype)?;

        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).map_err(|_| Error::External {
            message: "failed to create normal distribution with mean=0.0 and std=1.0".to_string(),
        })?;

        let elem_size = dtype.size_in_bytes();
        let mut host_buf = Vec::with_capacity(size * elem_size);
        for _ in 0..size {
            let x: f32 = normal.sample(&mut rng);
            match dtype {
                DType::BF16 => {
                    host_buf.extend_from_slice(&bf16::from_f32(x as f32).to_ne_bytes());
                },
                DType::F16 => {
                    host_buf.extend_from_slice(&f16::from_f32(x as f32).to_ne_bytes());
                },
                DType::F32 => {
                    host_buf.extend_from_slice(&(x as f32).to_ne_bytes());
                },
                DType::F64 => {
                    host_buf.extend_from_slice(&x.to_ne_bytes());
                },
                DType::BOOL => {
                    host_buf.push((x > 0.0) as u8);
                },
                DType::U8 => {
                    host_buf.push(x.round().abs().min(u8::MAX as f32) as u8);
                },
                DType::U16 => {
                    host_buf.extend_from_slice(&(x.round().abs().min(u16::MAX as f32) as u16).to_ne_bytes());
                },
                DType::U32 => {
                    host_buf.extend_from_slice(&(x.round().abs() as u32).to_ne_bytes());
                },
                DType::U64 => {
                    host_buf.extend_from_slice(&(x.round().abs() as u64).to_ne_bytes());
                },
                DType::I8 => {
                    host_buf.push(x.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8 as u8);
                },
                DType::I16 => {
                    host_buf
                        .extend_from_slice(&(x.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16).to_ne_bytes());
                },
                DType::I32 => {
                    host_buf
                        .extend_from_slice(&(x.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32).to_ne_bytes());
                },
                DType::I64 => {
                    host_buf.extend_from_slice(&(x.round() as i64).to_ne_bytes());
                },
            }
        }

        unsafe {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_host(host_buf.as_ptr() as *const std::ffi::c_void, size * elem_size, 0, 0)?;
        }

        let tid = next_tensor_id();
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

    /// Attempts to create a new tensor with values from 0 to n-1.
    ///
    /// Uses the default device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_range() -> Result<Tensor> {
    ///     let range = Tensor::try_range(5)?; // [0, 1, 2, 3, 4]
    ///     Ok(range)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_range(n: usize) -> Result<Self> {
        Self::try_arange(0, n as i32, 1)
    }

    /// Attempts to create a new tensor with values from 0 to n-1 with the specified device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_range_with_spec() -> Result<Tensor> {
    ///     let range = Tensor::try_range_with_spec(5, Device::CPU, DType::F32)?; // [0, 1, 2, 3, 4]
    ///     Ok(range)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn try_range_with_spec(n: usize, device: Device, dtype: DType) -> Result<Self> {
        Self::try_arange_with_spec(0, n as i32, 1, device, dtype)
    }

    /// Attempts to create a new tensor with evenly spaced values from start to end (exclusive) with the given step.
    ///
    /// Uses the default device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn create_arange() -> Result<Tensor> {
    ///     let arange = Tensor::try_arange(0, 10, 2)?; // [0, 2, 4, 6, 8]
    ///     Ok(arange)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if step is zero or if buffer allocation fails.
    pub fn try_arange<T>(start: T, end: T, step: T) -> Result<Self>
    where
        T: Into<Scalar> + Copy,
    {
        let device = get_default_device();
        let dtype = get_default_dtype();
        Self::try_arange_with_spec(start, end, step, device, dtype)
    }

    /// Attempts to create a new tensor with evenly spaced values from start to end (exclusive) with the given step,
    /// using the specified device and data type.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, dtype::DType, error::Result};
    ///
    /// fn create_arange_with_spec() -> Result<Tensor> {
    ///     let arange = Tensor::try_arange_with_spec(0, 10, 2, Device::CPU, DType::F32)?; // [0, 2, 4, 6, 8]
    ///     Ok(arange)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if step is zero or if buffer allocation fails.
    pub fn try_arange_with_spec<T>(start: T, end: T, step: T, device: Device, dtype: DType) -> Result<Self>
    where
        T: Into<Scalar> + Copy,
    {
        let start_scalar = start.into();
        let end_scalar = end.into();
        let step_scalar = step.into();

        let step_val_f32 = step_scalar.as_f32();
        if step_val_f32 == 0.0 {
            return Err(Error::InvalidArgument("arange: step cannot be zero".to_string()));
        }

        let count = (((end_scalar.as_f64() - start_scalar.as_f64()) / step_scalar.as_f64()).ceil()) as usize;

        macro_rules! build_vec {
            ($ty:ty, $convert:expr) => {{
                let mut v: Vec<$ty> = Vec::with_capacity(count);
                let mut current = start_scalar.as_f64();
                for _ in 0..count {
                    v.push($convert(current));
                    current += step_scalar.as_f64();
                }
                v
            }};
        }

        match dtype {
            DType::BF16 => {
                let data: Vec<bf16> = build_vec!(bf16, |x: f64| bf16::from_f32(x as f32));
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::F16 => {
                let data: Vec<f16> = build_vec!(f16, |x: f64| f16::from_f32(x as f32));
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::F32 => {
                let data: Vec<f32> = build_vec!(f32, |x: f64| x as f32);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::F64 => {
                let data: Vec<f64> = build_vec!(f64, |x: f64| x);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::BOOL => {
                let data: Vec<bool> = build_vec!(bool, |x: f64| x != 0.0);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::U8 => {
                let data: Vec<u8> = build_vec!(u8, |x: f64| x.max(0.0).min(u8::MAX as f64) as u8);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::U16 => {
                let data: Vec<u16> = build_vec!(u16, |x: f64| x.max(0.0).min(u16::MAX as f64) as u16);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::U32 => {
                let data: Vec<u32> = build_vec!(u32, |x: f64| x.max(0.0) as u32);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::U64 => {
                let data: Vec<u64> = build_vec!(u64, |x: f64| x.max(0.0) as u64);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::I8 => {
                let data: Vec<i8> = build_vec!(i8, |x: f64| x.max(i8::MIN as f64).min(i8::MAX as f64) as i8);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::I16 => {
                let data: Vec<i16> = build_vec!(i16, |x: f64| x.max(i16::MIN as f64).min(i16::MAX as f64) as i16);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::I32 => {
                let data: Vec<i32> = build_vec!(i32, |x: f64| x as i32);
                Self::try_new_with_spec(data, device, dtype)
            },
            DType::I64 => {
                let data: Vec<i64> = build_vec!(i64, |x: f64| x as i64);
                Self::try_new_with_spec(data, device, dtype)
            },
        }
    }
}
