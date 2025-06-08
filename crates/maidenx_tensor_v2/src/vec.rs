use crate::Tensor;
use maidenx_core::{
    buffer::BufferManager,
    dtype::DType,
    error::{Error, Result},
};
use std::sync::Arc;

/// ## Converting tensors to Rust vectors
///
/// These methods *copy* tensor data from device/storage back to host memory and
/// convert it into plain Rust containers.
///
/// **Flattened vector conversion:**
/// * **`to_flatten_vec`** – infallible wrapper that panics on failure.  
/// * **`try_to_flatten_vec`** – fallible variant returning `Result<Vec<T>>`.
///
/// **Dimensional vector conversion:**
/// * **`to_vec1d`/`try_to_vec1d`** – convert 1D tensors to `Vec<T>`
/// * **`to_vec2d`/`try_to_vec2d`** – convert 2D tensors to `Vec<Vec<T>>`
/// * **`to_vec3d`/`try_to_vec3d`** – convert 3D tensors to `Vec<Vec<Vec<T>>>`
/// * **`to_vec4d`/`try_to_vec4d`** – convert 4D tensors to `Vec<Vec<Vec<Vec<T>>>>`
/// * **`to_vec5d`/`try_to_vec5d`** – convert 5D tensors to `Vec<Vec<Vec<Vec<Vec<T>>>>>`
/// * **`to_vec6d`/`try_to_vec6d`** – convert 6D tensors to `Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>`
///
/// All methods respect the tensor's logical view (`shape`, `strides`, `offset`) to
/// correctly represent the tensor data in the resulting vector structure.
///
/// **Note:** It is recommended to use the tensor's `dtype` for conversions.  
/// If the `dtype` of the tensor does not match the native Rust type, the method will
/// create a new buffer and copy the data with type casting.
impl Tensor {
    /// Runs [`try_to_flatten_vec`](Self::try_to_flatten_vec) and panics on failure.
    ///
    /// If the tensor’s internal [`DType`] differs from `T`, a temporary buffer is
    /// allocated and the data are copied with element-wise type casting.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///
    /// // Explicit type annotation
    /// let v1: Vec<f32> = tensor.to_flatten_vec();
    /// assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);
    ///
    /// // Turbofish syntax
    /// let v2 = tensor.to_flatten_vec::<f32>();
    /// assert_eq!(v2, v1);
    /// ```
    ///
    /// # Panics
    ///
    /// * When forward pass fails (for tensors in computation graphs)
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_flatten_vec<T: Default + Clone + 'static>(&self) -> Vec<T> {
        self.try_to_flatten_vec::<T>().expect("failed to flatten tensor to Vec")
    }

    /// Runs [`try_to_vec1d`](Self::try_to_vec1d) and panics on failure.
    ///
    /// Converts a 1D tensor to a vector of type `T`. This function requires the tensor
    /// to be exactly 1-dimensional.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]); // 1D tensor
    /// let vec = tensor.to_vec1d::<f32>();
    /// assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When the tensor is not 1-dimensional
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_vec1d<T: Default + Clone + 'static>(&self) -> Vec<T> {
        self.try_to_vec1d().expect("failed to convert tensor to Vec1D")
    }

    /// Runs [`try_to_vec2d`](Self::try_to_vec2d) and panics on failure.
    ///
    /// Converts a 2D tensor to a nested vector of type `Vec<Vec<T>>`. This function requires
    /// the tensor to be exactly 2-dimensional.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0]]); // 2D tensor
    /// let matrix = tensor.to_vec2d::<f32>();
    /// assert_eq!(matrix, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When the tensor is not 2-dimensional
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_vec2d<T: Default + Clone + 'static>(&self) -> Vec<Vec<T>> {
        self.try_to_vec2d().expect("failed to convert tensor to Vec2D")
    }

    /// Runs [`try_to_vec3d`](Self::try_to_vec3d) and panics on failure.
    ///
    /// Converts a 3D tensor to a triply nested vector of type `Vec<Vec<Vec<T>>>`. This function
    /// requires the tensor to be exactly 3-dimensional.
    ///
    /// # Examples
    /// ```
    /// // 3D tensor with shape [2, 2, 2]
    /// let tensor = Tensor::from_data(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
    /// let cube = tensor.to_vec3d::<f32>();
    /// assert_eq!(cube, vec![
    ///     vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    ///     vec![vec![5.0, 6.0], vec![7.0, 8.0]]
    /// ]);
    /// ```
    ///
    /// # Panics
    ///
    /// * When the tensor is not 3-dimensional
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_vec3d<T: Default + Clone + 'static>(&self) -> Vec<Vec<Vec<T>>> {
        self.try_to_vec3d().expect("failed to convert tensor to Vec3D")
    }

    /// Runs [`try_to_vec4d`](Self::try_to_vec4d) and panics on failure.
    ///
    /// Converts a 4D tensor to a quadruply nested vector of type `Vec<Vec<Vec<Vec<T>>>>`. This
    /// function requires the tensor to be exactly 4-dimensional.
    ///
    /// # Examples
    /// ```
    /// // Create a 4D tensor with shape [2, 2, 2, 2]
    /// let tensor = create_4d_tensor(); // Assume this function exists
    /// let hypercube = tensor.to_vec4d::<f32>();
    /// ```
    ///
    /// # Panics
    ///
    /// * When the tensor is not 4-dimensional
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_vec4d<T: Default + Clone + 'static>(&self) -> Vec<Vec<Vec<Vec<T>>>> {
        self.try_to_vec4d().expect("failed to convert tensor to Vec4D")
    }

    /// Runs [`try_to_vec5d`](Self::try_to_vec5d) and panics on failure.
    ///
    /// Converts a 5D tensor to a quintuply nested vector of type `Vec<Vec<Vec<Vec<Vec<T>>>>>`. This
    /// function requires the tensor to be exactly 5-dimensional.
    ///
    /// # Panics
    ///
    /// * When the tensor is not 5-dimensional
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_vec5d<T: Default + Clone + 'static>(&self) -> Vec<Vec<Vec<Vec<Vec<T>>>>> {
        self.try_to_vec5d().expect("failed to convert tensor to Vec5D")
    }

    /// Runs [`try_to_vec6d`](Self::try_to_vec6d) and panics on failure.
    ///
    /// Converts a 6D tensor to a sextuply nested vector of type `Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>`. This
    /// function requires the tensor to be exactly 6-dimensional.
    ///
    /// # Panics
    ///
    /// * When the tensor is not 6-dimensional
    /// * When the tensor cannot be converted to the requested element type `T`
    ///   (e.g. unsupported cast or internal buffer error)
    pub fn to_vec6d<T: Default + Clone + 'static>(&self) -> Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>> {
        self.try_to_vec6d().expect("failed to convert tensor to Vec6D")
    }

    /// Attempts to convert the tensor to a flattened vector of type T.
    ///
    /// This function takes into account the tensor's shape, strides, and offset to
    /// correctly convert the multi-dimensional tensor data into a flattened vector.
    /// If the tensor is part of a computation graph and not yet materialized,
    /// it will be computed first before the conversion.
    ///
    /// **Note:** If the tensor's `dtype` does not match the requested type `T`, a new buffer
    /// will be created, and the data will be copied with type casting.
    ///
    /// # Examples
    /// ```
    /// use maidenx_core::error::Result;
    ///
    /// fn flatten_tensor() -> Result<Vec<f32>> {
    ///     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0]);
    ///     let vec = tensor.try_to_flatten_vec()?;
    ///     // or let vec = tensor.try_to_flatten_vec::<f32>()?;
    ///     assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    ///     Ok(vec)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forward pass fails (for tensors in computation graphs)
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    /// - There are issues with memory operations
    pub fn try_to_flatten_vec<T: Default + Clone + 'static>(&self) -> Result<Vec<T>> {
        // If tensor is in a computation graph and not materialized, compute it first
        if !self.is_const() && !self.is_storaged() {
            self.try_forward()?;
        }
        let target_dtype =
            get_dtype_for_type::<T>().ok_or_else(|| Error::InvalidArgument("Unsupported type".into()))?;

        let size = self.size();
        let shape = self.shape();
        let strides = self.strides();
        let offset = self.offset();
        let elem_size = target_dtype.size_in_bytes();

        let guard = self.storage()?;
        let buffer_size = guard.buffer().len() * elem_size;
        let mut raw_data = vec![0u8; buffer_size];

        if self.dtype() == target_dtype {
            unsafe {
                guard
                    .buffer()
                    .copy_to_host(raw_data.as_mut_ptr() as *mut std::ffi::c_void, buffer_size, 0, 0)?;
            }
        } else {
            let mut buffer = BufferManager::create(size, self.device(), target_dtype)?;
            let buffer_mut = Arc::get_mut(&mut buffer).ok_or(Error::BufferShared)?;
            buffer_mut.copy_from_with_dtype_cast(guard.buffer(), 0, 0, size.min(guard.buffer().len()))?;

            unsafe {
                buffer_mut.copy_to_host(raw_data.as_mut_ptr() as *mut std::ffi::c_void, buffer_size, 0, 0)?;
            }
        }

        let mut result = vec![T::default(); size];
        let mut indices = vec![0; shape.len()];
        let mut dst_idx = 0;

        let calc_src_offset = |indices: &[usize], strides: &[usize], offset: usize| -> usize {
            offset
                + indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>()
        };

        loop {
            let src_offset = calc_src_offset(&indices, &strides, offset);

            if src_offset * elem_size < raw_data.len() {
                // Copy element from source to destination
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw_data.as_ptr().add(src_offset * elem_size),
                        (result.as_mut_ptr() as *mut u8).add(dst_idx * elem_size),
                        elem_size,
                    );
                }
            }

            dst_idx += 1;

            // Update indices
            if dst_idx == size {
                return Ok(result);
            }

            let mut dim = shape.len();
            while dim > 0 {
                dim -= 1;
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    /// Attempts to convert a 1D tensor to a `Vec<T>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not 1-dimensional
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    pub fn try_to_vec1d<T: Default + Clone + 'static>(&self) -> Result<Vec<T>> {
        if self.ndim() != 1 {
            return Err(Error::InvalidArgument(format!(
                "Expected 1D tensor, but got {}D tensor",
                self.ndim()
            )));
        }
        self.try_to_flatten_vec()
    }

    /// Attempts to convert a 2D tensor to a `Vec<Vec<T>>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not 2-dimensional
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    pub fn try_to_vec2d<T: Default + Clone + 'static>(&self) -> Result<Vec<Vec<T>>> {
        if self.ndim() != 2 {
            return Err(Error::InvalidArgument(format!(
                "Expected 2D tensor, but got {}D tensor",
                self.ndim()
            )));
        }

        let flat_data = self.try_to_flatten_vec::<T>()?;
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(flat_data[start..end].to_vec());
        }

        Ok(result)
    }

    /// Attempts to convert a 3D tensor to a `Vec<Vec<Vec<T>>>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not 3-dimensional
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    pub fn try_to_vec3d<T: Default + Clone + 'static>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        if self.ndim() != 3 {
            return Err(Error::InvalidArgument(format!(
                "Expected 3D tensor, but got {}D tensor",
                self.ndim()
            )));
        }

        let flat_data = self.try_to_flatten_vec::<T>()?;
        let shape = self.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let idx = (i * dim1 * dim2) + (j * dim2) + k;
                    slice_j.push(flat_data[idx].clone());
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
    }

    /// Attempts to convert a 4D tensor to a `Vec<Vec<Vec<Vec<T>>>>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not 4-dimensional
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    pub fn try_to_vec4d<T: Default + Clone + 'static>(&self) -> Result<Vec<Vec<Vec<Vec<T>>>>> {
        if self.ndim() != 4 {
            return Err(Error::InvalidArgument(format!(
                "Expected 4D tensor, but got {}D tensor",
                self.ndim()
            )));
        }

        let flat_data = self.try_to_flatten_vec::<T>()?;
        let shape = self.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];
        let dim3 = shape[3];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let mut slice_k = Vec::with_capacity(dim3);
                    for l in 0..dim3 {
                        let idx = (i * dim1 * dim2 * dim3) + (j * dim2 * dim3) + (k * dim3) + l;
                        slice_k.push(flat_data[idx].clone());
                    }
                    slice_j.push(slice_k);
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
    }

    /// Attempts to convert a 5D tensor to a `Vec<Vec<Vec<Vec<Vec<T>>>>>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not 5-dimensional
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    pub fn try_to_vec5d<T: Default + Clone + 'static>(&self) -> Result<Vec<Vec<Vec<Vec<Vec<T>>>>>> {
        if self.ndim() != 5 {
            return Err(Error::InvalidArgument(format!(
                "Expected 5D tensor, but got {}D tensor",
                self.ndim()
            )));
        }

        let flat_data = self.try_to_flatten_vec::<T>()?;
        let shape = self.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];
        let dim3 = shape[3];
        let dim4 = shape[4];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let mut slice_k = Vec::with_capacity(dim3);
                    for l in 0..dim3 {
                        let mut slice_l = Vec::with_capacity(dim4);
                        for m in 0..dim4 {
                            let idx = (i * dim1 * dim2 * dim3 * dim4)
                                + (j * dim2 * dim3 * dim4)
                                + (k * dim3 * dim4)
                                + (l * dim4)
                                + m;
                            slice_l.push(flat_data[idx].clone());
                        }
                        slice_k.push(slice_l);
                    }
                    slice_j.push(slice_k);
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
    }

    /// Attempts to convert a 6D tensor to a `Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not 6-dimensional
    /// - The tensor storage cannot be accessed
    /// - The buffer cannot be copied to host memory
    pub fn try_to_vec6d<T: Default + Clone + 'static>(&self) -> Result<Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>> {
        if self.ndim() != 6 {
            return Err(Error::InvalidArgument(format!(
                "Expected 6D tensor, but got {}D tensor",
                self.ndim()
            )));
        }

        let flat_data = self.try_to_flatten_vec::<T>()?;
        let shape = self.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];
        let dim3 = shape[3];
        let dim4 = shape[4];
        let dim5 = shape[5];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let mut slice_k = Vec::with_capacity(dim3);
                    for l in 0..dim3 {
                        let mut slice_l = Vec::with_capacity(dim4);
                        for m in 0..dim4 {
                            let mut slice_m = Vec::with_capacity(dim5);
                            for n in 0..dim5 {
                                let idx = (i * dim1 * dim2 * dim3 * dim4 * dim5)
                                    + (j * dim2 * dim3 * dim4 * dim5)
                                    + (k * dim3 * dim4 * dim5)
                                    + (l * dim4 * dim5)
                                    + (m * dim5)
                                    + n;
                                slice_m.push(flat_data[idx].clone());
                            }
                            slice_l.push(slice_m);
                        }
                        slice_k.push(slice_l);
                    }
                    slice_j.push(slice_k);
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
    }
}

fn get_dtype_for_type<T: 'static>() -> Option<DType> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>() {
        Some(DType::BF16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
        Some(DType::F16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        Some(DType::F32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        Some(DType::F64)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
        Some(DType::BOOL)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
        Some(DType::U8)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>() {
        Some(DType::U16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        Some(DType::U32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        Some(DType::U64)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        Some(DType::I8)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
        Some(DType::I16)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        Some(DType::I32)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        Some(DType::I64)
    } else {
        None
    }
}
