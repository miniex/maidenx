#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    #[cfg(feature = "cuda")]
    CUDA(usize),
    #[cfg(feature = "mps")]
    MPS,
}

#[cfg(feature = "serde")]
impl Serialize for Device {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.name())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Device {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DeviceVisitor;

        impl serde::de::Visitor<'_> for DeviceVisitor {
            type Value = Device;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string representing a Device (CPU, CUDA|id, or MPS)")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value == "CPU" {
                    return Ok(Device::CPU);
                }

                #[cfg(feature = "cuda")]
                if let Some(cuda_str) = value.strip_prefix("CUDA|") {
                    if let Ok(id) = cuda_str.parse::<usize>() {
                        return Ok(Device::CUDA(id));
                    } else {
                        return Err(E::custom(format!("invalid CUDA device ID: {}", cuda_str)));
                    }
                }

                #[cfg(feature = "mps")]
                if value == "MPS" {
                    return Ok(Device::MPS);
                }

                Err(E::custom(format!("unknown device: {}", value)))
            }
        }

        deserializer.deserialize_str(DeviceVisitor)
    }
}

impl Device {
    pub fn name(&self) -> String {
        match self {
            Device::CPU => "CPU".to_string(),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => format!("CUDA|{}", id),
            #[cfg(feature = "mps")]
            Device::MPS => "MPS".to_string(),
        }
    }
}

thread_local! {
    static DEFAULT_DEVICE: std::cell::Cell<Device> = const { std::cell::Cell::new(Device::CPU) };
}

pub fn get_default_device() -> Device {
    DEFAULT_DEVICE.with(|d| d.get())
}

pub fn set_default_device(device: Device) {
    DEFAULT_DEVICE.with(|d| d.set(device));
}

pub fn auto_set_device() {
    #[cfg(feature = "cuda")]
    set_default_device(Device::CUDA(0));
    #[cfg(feature = "mps")]
    set_default_device(Device::MPS);
    #[cfg(not(any(feature = "cuda", feature = "mps")))]
    set_default_device(Device::CPU);
}

