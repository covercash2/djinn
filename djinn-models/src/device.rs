//! Mapping from the config-surface [`Device`] to a runtime candle device.
//!
//! This cannot be a `TryFrom` impl: `Device` lives in `djinn-core` and the candle device in
//! `candle-core`, so the orphan rule forbids implementing the conversion here.

use candle_core::Device as CandleDevice;
use djinn_core::device::Device;

/// The device used when a config does not name one: the best accelerator compiled into this
/// build (`cuda` or `mac` feature), otherwise the CPU.
pub fn default_device() -> Device {
    if cfg!(feature = "cuda") {
        Device::Cuda
    } else if cfg!(feature = "mac") {
        Device::Metal
    } else {
        Device::Cpu
    }
}

/// Returns the explicit device if one is given, otherwise [`default_device`].
pub fn resolve(device: Option<Device>) -> Device {
    device.unwrap_or_else(default_device)
}

/// Creates the candle device for a config-surface device.
pub fn to_candle(device: Device) -> candle_core::Result<CandleDevice> {
    match device {
        Device::Cpu => Ok(CandleDevice::Cpu),
        Device::Cuda => CandleDevice::new_cuda(0),
        Device::Metal => CandleDevice::new_metal(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_converts_to_candle_cpu() {
        assert!(to_candle(Device::Cpu).unwrap().is_cpu());
    }

    #[test]
    fn explicit_device_wins_over_default() {
        assert_eq!(resolve(Some(Device::Cpu)), Device::Cpu);
        assert_eq!(resolve(Some(Device::Metal)), Device::Metal);
    }

    #[test]
    fn missing_device_uses_build_default() {
        assert_eq!(resolve(None), default_device());
    }

    #[test]
    #[cfg(not(any(feature = "cuda", feature = "mac")))]
    fn default_is_cpu_without_accelerators() {
        assert_eq!(default_device(), Device::Cpu);
    }
}
