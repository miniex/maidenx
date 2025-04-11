use crate::{Tensor, TensorData, TensorMetadata};
use maidenx_core::{buffer::Buffer, device::Device, dtype::DType, error, layout::Layout};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize)]
struct SerializedTensorData {
    buffer_data: Vec<u8>,
    buffer_len: usize,
    buffer_dtype: DType,
    buffer_device: Device,
}

#[derive(Serialize, Deserialize)]
struct SerializedTensorMetadata {
    device: Device,
    dtype: DType,
    layout: Layout,
    requires_grad: bool,
}

#[derive(Serialize, Deserialize)]
struct SerializedTensor {
    data: SerializedTensorData,
    metadata: SerializedTensorMetadata,
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let contiguous_tensor = match self.contiguous() {
            Ok(tensor) => tensor,
            Err(e) => {
                return Err(serde::ser::Error::custom(format!(
                    "Failed to make tensor contiguous: {}",
                    e
                )))
            },
        };

        let buffer = contiguous_tensor.buffer();
        let buffer_len = buffer.len();
        let buffer_dtype = buffer.dtype();
        let buffer_device = buffer.device();

        let actual_size = contiguous_tensor.metadata.layout.size();
        let elem_size = buffer_dtype.size_in_bytes();
        let actual_bytes = actual_size * elem_size;

        let mut buffer_data = vec![0u8; actual_bytes];

        unsafe {
            buffer
                .copy_to_host(buffer_data.as_mut_ptr() as *mut std::ffi::c_void, actual_bytes, 0, 0)
                .map_err(serde::ser::Error::custom)?;
        }

        let serialized = SerializedTensor {
            data: SerializedTensorData {
                buffer_data,
                buffer_len,
                buffer_dtype,
                buffer_device,
            },
            metadata: SerializedTensorMetadata {
                device: self.metadata.device,
                dtype: self.metadata.dtype,
                layout: self.metadata.layout.clone(),
                requires_grad: self.metadata.requires_grad,
            },
        };

        serialized.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized = SerializedTensor::deserialize(deserializer)?;

        let buffer: Arc<dyn Buffer> = match serialized.data.buffer_device {
            Device::CPU => {
                use maidenx_core::buffer::cpu::CpuBuffer;

                let mut buffer = CpuBuffer::new(serialized.data.buffer_len, serialized.data.buffer_dtype)
                    .map_err(de::Error::custom)?;

                let data_ptr = serialized.data.buffer_data.as_ptr() as *const std::ffi::c_void;
                let data_size = serialized.data.buffer_data.len();

                unsafe {
                    buffer
                        .copy_from_host(data_ptr, data_size, 0, 0)
                        .map_err(de::Error::custom)?;
                }

                Arc::new(buffer)
            },
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                use maidenx_core::buffer::cuda::CudaBuffer;

                let mut buffer = CudaBuffer::new(serialized.data.buffer_len, serialized.data.buffer_dtype, device_id)
                    .map_err(de::Error::custom)?;

                let data_ptr = serialized.data.buffer_data.as_ptr() as *const std::ffi::c_void;
                let data_size = serialized.data.buffer_data.len();

                unsafe {
                    buffer
                        .copy_from_host(data_ptr, data_size, 0, 0)
                        .map_err(de::Error::custom)?;
                }

                Arc::new(buffer)
            },
            #[cfg(feature = "mps")]
            Device::MPS => {
                use maidenx_core::buffer::mps::MpsBuffer;

                let mut buffer = MpsBuffer::new(serialized.data.buffer_len, serialized.data.buffer_dtype)
                    .map_err(de::Error::custom)?;

                let data_ptr = serialized.data.buffer_data.as_ptr() as *const std::ffi::c_void;
                let data_size = serialized.data.buffer_data.len();

                unsafe {
                    buffer
                        .copy_from_host(data_ptr, data_size, 0, 0)
                        .map_err(de::Error::custom)?;
                }
                Arc::new(buffer)
            },
            #[allow(unreachable_patterns)]
            _ => return Err(de::Error::custom("Unsupported device for deserialization")),
        };

        let mut tensor = Tensor {
            data: TensorData { buffer, grad: None },
            metadata: TensorMetadata {
                device: serialized.metadata.device,
                dtype: serialized.metadata.dtype,
                layout: serialized.metadata.layout,
                requires_grad: serialized.metadata.requires_grad,
            },
            node: None,
        };

        if tensor.requires_grad() {
            let grad_storage = match Tensor::zeros_like(&tensor) {
                Ok(t) => t,
                Err(e) => return Err(de::Error::custom(format!("Failed to create grad tensor: {}", e))),
            };
            tensor.data.grad = Some(Arc::new(Mutex::new(grad_storage)));
        }

        Ok(tensor)
    }
}

impl Tensor {
    pub fn to_bytes(&self) -> error::Result<Vec<u8>> {
        let config = bincode::config::legacy();
        bincode::serde::encode_to_vec(self, config)
            .map_err(|e| error::Error::SerializationError(format!("Failed to serialize tensor: {}", e)))
    }

    pub fn from_bytes(bytes: &[u8]) -> error::Result<Self> {
        let config = bincode::config::legacy();
        bincode::serde::decode_from_slice(bytes, config)
            .map(|(value, _)| value)
            .map_err(|e| error::Error::DeserializationError(format!("Failed to deserialize tensor: {}", e)))
    }

    pub fn to_json(&self) -> error::Result<String> {
        serde_json::to_string(self)
            .map_err(|e| error::Error::SerializationError(format!("Failed to serialize tensor to JSON: {}", e)))
    }

    pub fn from_json(json: &str) -> error::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| error::Error::DeserializationError(format!("Failed to deserialize tensor from JSON: {}", e)))
    }
}
