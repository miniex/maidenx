use crate::{get_mode, next_tensor_id, utils::graph::add_to_graph, Tensor, TensorId, TensorMode};
use maidenx_core::{
    buffer::BufferManager,
    device::Device,
    dtype::DType,
    error::{Error, Result},
};
use std::sync::Arc;

// Helper function to check MPS 64-bit compatibility
#[cfg(feature = "mps")]
fn check_mps_compatibility(device: Device, dtype: DType) -> Result<()> {
    if device == Device::MPS && dtype.size() == 8 {
        return Err(Error::UnsupportedDevice {
            device,
            message: format!("MPS does not support 64-bit dtypes ({})", dtype).into(),
        });
    }
    Ok(())
}

#[cfg(not(feature = "mps"))]
fn check_mps_compatibility(_device: Device, _dtype: DType) -> Result<()> {
    Ok(())
}

/// ## Device & dtype conversion helpers
///
/// This `impl` block provides methods for converting tensors between devices
/// and data types. There are two main categories:
///
/// * **In-place modification**: `with_device`, `with_dtype` - modifies existing tensors
/// * **New tensor creation**: `to_device`, `to_dtype` - creates new tensors
///
/// Each infallible method is a thin wrapper that **panics on error** around
/// its fallible counterpart (`try_*`).
///
/// **Behavior by execution mode:**
/// * **Eager mode**: Operations execute immediately with direct storage modification
/// * **Lazy mode**: Operations are added to the computation graph for deferred execution
///
/// **MPS compatibility notes:**
/// * 64-bit data types (F64, I64, U64) are not supported on MPS devices
/// * Compatibility checks are performed automatically before operations
///
/// **Memory efficiency:**
/// * In-place operations (`with_*`) modify the original tensor's storage in eager mode
/// * In lazy mode, `with_*` methods internally use `to_*` methods for graph consistency
/// * New tensor operations (`to_*`) always create separate storage
impl Tensor {
    /// Runs [`try_with_device`](Self::try_with_device) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::device::Device;
    ///
    /// let mut tensor = Tensor::zeros(&[2, 3]);
    /// tensor.with_device(Device::CPU);
    /// assert_eq!(tensor.device(), Device::CPU);
    /// ```
    ///
    /// # Panics
    ///
    /// * When device transfer fails
    /// * When forward pass fails (for tensors in computation graphs)
    pub fn with_device(&mut self, device: Device) {
        self.try_with_device(device)
            .expect("failed to transfer tensor to device")
    }

    /// Runs [`try_with_dtype`](Self::try_with_dtype) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::dtype::DType;
    ///
    /// let mut tensor = Tensor::zeros(&[2, 3]);
    /// tensor.with_dtype(DType::F16);
    /// assert_eq!(tensor.dtype(), DType::F16);
    /// ```
    ///
    /// # Panics
    ///
    /// * When dtype conversion fails
    /// * When forward pass fails (for tensors in computation graphs)
    /// * On MPS, if a 64-bit dtype is requested (when the `mps` feature is enabled)
    pub fn with_dtype(&mut self, dtype: DType) {
        self.try_with_dtype(dtype).expect("failed to convert tensor dtype")
    }

    /// Runs [`try_to_device`](Self::try_to_device) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::device::Device;
    ///
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let cpu_tensor = tensor.to_device(Device::CPU);
    /// assert_eq!(cpu_tensor.device(), Device::CPU);
    /// ```
    ///
    /// # Panics
    ///
    /// * When device transfer fails
    /// * When buffer creation or data copy fails
    pub fn to_device(&self, device: Device) -> Self {
        self.try_to_device(device).expect("failed to transfer tensor to device")
    }

    /// Runs [`try_to_dtype`](Self::try_to_dtype) and panics on failure.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::dtype::DType;
    ///
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let f16_tensor = tensor.to_dtype(DType::F16);
    /// assert_eq!(f16_tensor.dtype(), DType::F16);
    /// ```
    ///
    /// # Panics
    ///
    /// * When dtype conversion fails
    /// * When buffer creation or data conversion fails
    /// * On MPS, if a 64-bit dtype is requested (when the `mps` feature is enabled)
    pub fn to_dtype(&self, dtype: DType) -> Self {
        self.try_to_dtype(dtype).expect("failed to convert tensor dtype")
    }

    /// Attempts to move this tensor to a different device, modifying it in place.
    ///
    /// This operation transfers the tensor's data to the target device.
    /// In Eager mode, the tensor's storage is modified directly.
    /// In Lazy mode, this creates a new tensor using `try_to_device` internally.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, error::Result};
    ///
    /// fn transfer_tensor() -> Result<()> {
    ///     let mut tensor = Tensor::zeros(&[2, 3]);
    ///     tensor.try_with_device(Device::CPU)?;
    ///     assert_eq!(tensor.device(), Device::CPU);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer creation on the new device fails
    /// - The data copy operation fails
    /// - Graph operations fail in lazy mode
    pub fn try_with_device(&mut self, device: Device) -> Result<()> {
        if self.device() == device {
            return Ok(());
        }
        check_mps_compatibility(device, self.dtype())?;

        match get_mode() {
            TensorMode::Eager => self.execute_with_device(device),
            TensorMode::Lazy => {
                *self = self.try_to_device(device)?;
                Ok(())
            },
        }
    }

    /// Attempts to convert this tensor to a different data type, modifying it in place.
    ///
    /// This operation converts the tensor's data to the target data type.
    /// In Eager mode, the tensor's storage is modified directly.
    /// In Lazy mode, this creates a new tensor using `try_to_dtype` internally.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{dtype::DType, error::Result};
    ///
    /// fn convert_tensor() -> Result<()> {
    ///     let mut tensor = Tensor::zeros(&[2, 3]);
    ///     tensor.try_with_dtype(DType::F16)?;
    ///     assert_eq!(tensor.dtype(), DType::F16);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer creation with the new data type fails
    /// - The data type conversion operation fails
    /// - When using MPS device (if MPS feature is enabled), if trying to use 64-bit data types
    /// - Graph operations fail in lazy mode
    pub fn try_with_dtype(&mut self, dtype: DType) -> Result<()> {
        if self.dtype() == dtype {
            return Ok(());
        }
        check_mps_compatibility(self.device(), dtype)?;

        match get_mode() {
            TensorMode::Eager => self.execute_with_dtype(dtype),
            TensorMode::Lazy => {
                *self = self.try_to_dtype(dtype)?;
                Ok(())
            },
        }
    }

    /// Attempts to convert tensor to a different device, returning a new tensor.
    ///
    /// This operation creates a new tensor on the target device with copied data.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{device::Device, error::Result};
    ///
    /// fn transfer_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[2, 3]);
    ///     let cpu_tensor = tensor.try_to_device(Device::CPU)?;
    ///     assert_eq!(cpu_tensor.device(), Device::CPU);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer creation on the new device fails
    /// - The data copy operation fails
    /// - Graph operations fail in lazy mode
    pub fn try_to_device(&self, device: Device) -> Result<Self> {
        check_mps_compatibility(device, self.dtype())?;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = crate::TensorMetadata {
                    device,
                    dtype: self.dtype(),
                    layout: self.layout(),
                    mode: get_mode(),
                    update_status: crate::TensorUpdateStatus::Pending,
                };
                crate::insert_metadata(target_tid, metadata);
                self.execute_to_device(device, target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_graph(
                    &[self],
                    "device_transfer",
                    &[device],
                    &[self.dtype()],
                    &[self.layout()],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_to_device(device, target_tids[0])?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    /// Attempts to convert tensor to a different data type, returning a new tensor.
    ///
    /// This operation creates a new tensor with the target data type and converted data.
    /// In Eager mode, the operation is executed immediately.
    /// In Lazy mode, the operation is added to the computation graph.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::{dtype::DType, error::Result};
    ///
    /// fn convert_tensor() -> Result<()> {
    ///     let tensor = Tensor::zeros(&[2, 3]);
    ///     let f16_tensor = tensor.try_to_dtype(DType::F16)?;
    ///     assert_eq!(f16_tensor.dtype(), DType::F16);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer creation with the new data type fails
    /// - The data type conversion operation fails
    /// - When using MPS device (if MPS feature is enabled), if trying to use 64-bit data types
    /// - Graph operations fail in lazy mode
    pub fn try_to_dtype(&self, dtype: DType) -> Result<Self> {
        check_mps_compatibility(self.device(), dtype)?;

        match get_mode() {
            TensorMode::Eager => {
                let target_tid = next_tensor_id();
                let metadata = crate::TensorMetadata {
                    device: self.device(),
                    dtype,
                    layout: self.layout(),
                    mode: get_mode(),
                    update_status: crate::TensorUpdateStatus::Pending,
                };
                crate::insert_metadata(target_tid, metadata);
                self.execute_to_dtype(dtype, target_tid)
            },
            TensorMode::Lazy => {
                let result = add_to_graph(
                    &[self],
                    "dtype_conversion",
                    &[self.device()],
                    &[dtype],
                    &[self.layout()],
                    move |tensors, target_tids| Ok(vec![tensors[0].execute_to_dtype(dtype, target_tids[0])?]),
                )?;
                Ok(result.into_iter().next().unwrap())
            },
        }
    }

    fn execute_with_device(&mut self, device: Device) -> Result<()> {
        let dtype = self.dtype();
        let buffer_len = self.storage()?.buffer().len();
        let mut buffer = BufferManager::create(buffer_len, device, dtype)?;
        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_device(self.storage()?.buffer(), 0, 0, buffer_len)?;
        }

        let mut storage = self.storage_mut()?;
        storage.buffer = buffer;
        let mut metadata = self.metadata_mut()?;
        metadata.set_device(device);
        Ok(())
    }

    fn execute_with_dtype(&mut self, dtype: DType) -> Result<()> {
        let buffer_len = self.storage()?.buffer().len();
        let device = self.device();
        let mut buffer = BufferManager::create(buffer_len, device, dtype)?;
        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(self.storage()?.buffer(), 0, 0, buffer_len)?;
        }

        let mut storage = self.storage_mut()?;
        storage.buffer = buffer;
        let mut metadata = self.metadata_mut()?;
        metadata.set_dtype(dtype);
        Ok(())
    }

    fn execute_to_device(&self, device: Device, target_tid: TensorId) -> Result<Self> {
        let dtype = self.dtype();
        let buffer_len = self.storage()?.buffer().len();

        let mut buffer = BufferManager::create(buffer_len, device, dtype)?;
        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_device(self.storage()?.buffer(), 0, 0, buffer_len)?;
        }

        // Create storage and link to target tensor
        let sid = crate::next_storage_id();
        crate::link_tensor_to_storage(target_tid, sid);
        crate::insert_storage(sid, crate::TensorStorage::new(buffer));

        // Update status to materialized
        crate::utils::tensor::update_tensor_status(target_tid, crate::TensorUpdateStatus::Materialized)?;

        Ok(Tensor {
            tid: target_tid,
            gtid: crate::TensorId(0),
            gid: None,
        })
    }

    fn execute_to_dtype(&self, dtype: DType, target_tid: TensorId) -> Result<Self> {
        let buffer_len = self.storage()?.buffer().len();
        let device = self.device();

        let mut buffer = BufferManager::create(buffer_len, device, dtype)?;
        {
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(self.storage()?.buffer(), 0, 0, buffer_len)?;
        }

        // Create storage and link to target tensor
        let sid = crate::next_storage_id();
        crate::link_tensor_to_storage(target_tid, sid);
        crate::insert_storage(sid, crate::TensorStorage::new(buffer));

        // Update status to materialized
        crate::utils::tensor::update_tensor_status(target_tid, crate::TensorUpdateStatus::Materialized)?;

        Ok(Tensor {
            tid: target_tid,
            gtid: crate::TensorId(0),
            gid: None,
        })
    }
}
