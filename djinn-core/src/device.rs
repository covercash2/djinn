use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// A enum of supported devices to run models on
///
/// Turning this into a runtime device is the job of the `djinn-models` crate.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    ValueEnum,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
pub enum Device {
    #[default]
    Cpu,
    Cuda,
    Metal,
}
