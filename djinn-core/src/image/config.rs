//! File-based configuration for image generation.
//!
//! Config is stored at `$XDG_CONFIG_HOME/djinn/image-gen.toml`
//! (typically `~/.config/djinn/image-gen.toml`).
//!
//! Priority order (lowest → highest): config file → env var → CLI flag.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::Result;

use crate::device::Device;
use super::gen::StableDiffusionVersion;

const CONFIG_FILENAME: &str = "image-gen.toml";

/// File-based configuration for image generation.
///
/// All fields are optional; unset fields fall back to the env-var and CLI
/// layers.  See [`load`] and [`crate::config::xdg`] for the loading logic.
///
/// Example `~/.config/djinn/image-gen.toml`:
/// ```toml
/// sd_version = "xl"
/// device = "metal"
/// use_f16 = true
/// num_samples = 4
/// final_image = "/tmp/output.png"
/// ```
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GenConfig {
    /// Text prompt describing the image to generate.
    pub prompt: Option<String>,
    /// Negative prompt (describe what to exclude from the image).
    pub uncond_prompt: Option<String>,
    /// Compute device to run inference on.
    pub device: Option<Device>,
    /// Stable Diffusion model version.
    pub sd_version: Option<StableDiffusionVersion>,
    /// Number of images to generate sequentially.
    pub num_samples: Option<usize>,
    /// Batch size (images generated in parallel per step).
    pub bsize: Option<usize>,
    /// Number of diffusion denoising steps.
    pub n_steps: Option<usize>,
    /// Use FP16 weights to reduce VRAM usage.
    pub use_f16: Option<bool>,
    /// Use flash attention (requires CUDA).
    pub use_flash_attn: Option<bool>,
    /// Classifier-free guidance scale.
    pub guidance_scale: Option<f64>,
    /// Output image height in pixels.
    pub height: Option<usize>,
    /// Output image width in pixels.
    pub width: Option<usize>,
    /// Output image file path.
    pub final_image: Option<PathBuf>,
    /// Sliced attention size (0 = automatic).
    pub sliced_attention_size: Option<usize>,
    /// Path to a local UNet weight file (.safetensors). Skips HF Hub download.
    pub unet_weights: Option<String>,
    /// Path to a local primary CLIP weight file (.safetensors).
    pub clip_weights: Option<String>,
    /// Path to a local secondary CLIP weight file (.safetensors, SDXL only).
    pub clip2_weights: Option<String>,
    /// Path to a local VAE weight file (.safetensors).
    pub vae_weights: Option<String>,
    /// Path to a local tokenizer file.
    pub tokenizer: Option<String>,
    /// Path to an image used to initialize the latents (img2img / inpainting).
    pub img2img: Option<PathBuf>,
    /// img2img transformation strength in the range 0.0–1.0.
    pub img2img_strength: Option<f64>,
    /// RNG seed for reproducible generation.
    pub seed: Option<u64>,
    /// Path to an inpainting mask image.
    pub mask_path: Option<String>,
}

/// Loads [`GenConfig`] from `path`, or from the default XDG location when
/// `path` is `None`.  Returns a default (all-`None`) config if the file does
/// not exist.
pub fn load(path: Option<&Path>) -> Result<GenConfig> {
    crate::config::xdg::load(path, CONFIG_FILENAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn load_returns_default_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("image-gen.toml");
        let config = load(Some(&path)).unwrap();
        assert!(config.prompt.is_none());
        assert!(config.num_samples.is_none());
    }

    #[test]
    fn load_parses_prompt_and_num_samples() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, r#"prompt = "a red balloon""#).unwrap();
        writeln!(f, "num_samples = 3").unwrap();

        let config = load(Some(f.path())).unwrap();
        assert_eq!(config.prompt.as_deref(), Some("a red balloon"));
        assert_eq!(config.num_samples, Some(3));
    }

    #[test]
    fn load_parses_use_f16_flag() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "use_f16 = true").unwrap();

        let config = load(Some(f.path())).unwrap();
        assert_eq!(config.use_f16, Some(true));
    }
}
