pub mod error;

use error::DeviceResult;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(device_index: usize) -> Self {
        Device::Cuda(device_index)
    }

    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "CPU".to_string(),
            #[cfg(feature = "cuda")]
            Device::Cuda(device_index) => format!("CUDA Device {}", device_index),
        }
    }
}

thread_local! {
    static CURRENT_DEVICE: std::cell::Cell<Device> = const { std::cell::Cell::new(Device::Cpu) };
}

pub fn get_current_device() -> Device {
    CURRENT_DEVICE.with(|d| d.get())
}

pub fn set_current_device(device: Device) -> DeviceResult<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(device_index) = device {
        maidenx_cuda::is_device_available(device_index as i32)?;
    }

    CURRENT_DEVICE.with(|d| d.set(device));
    Ok(())
}

pub struct DeviceGuard {
    prev_device: Device,
}

impl DeviceGuard {
    pub fn new(device: Device) -> DeviceResult<Self> {
        let prev_device = get_current_device();
        set_current_device(device)?;
        Ok(Self { prev_device })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        let _ = set_current_device(self.prev_device);
    }
}
