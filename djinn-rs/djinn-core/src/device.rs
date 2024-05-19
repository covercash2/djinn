use candle_core::Device as CandleDevice;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
}

impl Default for Device {
    fn default() -> Self {
        cfg_if::cfg_if! {
            if #[cfg(feature = "cuda")] {
                Device::Cuda
            } else if #[cfg(feature = "mac")] {
                Device::Metal,
            } else {
                Device::Cpu
            }
        }
    }
}

impl TryFrom<Device> for CandleDevice {
    type Error = candle_core::Error;

    fn try_from(value: Device) -> Result<Self, Self::Error> {
        match value {
            Device::Cpu => Ok(CandleDevice::Cpu),
            Device::Cuda => CandleDevice::new_cuda(0),
            Device::Metal => CandleDevice::new_metal(0),
        }
    }
}
