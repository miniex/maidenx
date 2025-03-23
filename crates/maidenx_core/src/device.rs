#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    #[cfg(feature = "cuda")]
    CUDA(usize),
    #[cfg(feature = "mps")]
    MPS,
}

impl Device {
    pub fn name(&self) -> String {
        match self {
            Device::CPU => "CPU".to_string(),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => format!("CUDA Device {}", id),
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
