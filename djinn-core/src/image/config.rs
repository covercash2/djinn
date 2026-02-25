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
/// When the `clap` feature is enabled this struct also implements
/// [`clap::Args`], so it can be used directly as a CLI argument group.
///
/// Example `~/.config/djinn/image-gen.toml`:
/// ```toml
/// sd_version = "v1_5"
/// device = "cpu"
/// use_f16 = true
/// num_samples = 4
/// final_image = "/tmp/output.png"
/// ```
#[derive(Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct GenConfig {
    /// Text prompt describing the image to generate.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_PROMPT"))]
    pub prompt: Option<String>,
    /// Negative prompt (describe what to exclude from the image).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_UNCOND_PROMPT"))]
    pub uncond_prompt: Option<String>,
    /// Compute device to run inference on.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_DEVICE"))]
    pub device: Option<Device>,
    /// Stable Diffusion model version.
    #[cfg_attr(feature = "clap", arg(long, value_enum, env = "DJINN_SD_VERSION"))]
    pub sd_version: Option<StableDiffusionVersion>,
    /// Number of images to generate sequentially.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_NUM_SAMPLES"))]
    pub num_samples: Option<usize>,
    /// Batch size (images generated in parallel per step).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_BSIZE"))]
    pub bsize: Option<usize>,
    /// Number of diffusion denoising steps.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_N_STEPS"))]
    pub n_steps: Option<usize>,
    /// Use FP16 weights to reduce VRAM usage.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_USE_F16"))]
    pub use_f16: Option<bool>,
    /// Use flash attention (requires CUDA).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_USE_FLASH_ATTN"))]
    pub use_flash_attn: Option<bool>,
    /// Classifier-free guidance scale.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_GUIDANCE_SCALE"))]
    pub guidance_scale: Option<f64>,
    /// Output image height in pixels.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_HEIGHT"))]
    pub height: Option<usize>,
    /// Output image width in pixels.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_WIDTH"))]
    pub width: Option<usize>,
    /// Output image file path.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_FINAL_IMAGE", value_name = "FILE"))]
    pub final_image: Option<PathBuf>,
    /// Sliced attention size (0 = automatic).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_SLICED_ATTENTION_SIZE"))]
    pub sliced_attention_size: Option<usize>,
    /// Path to a local UNet weight file (.safetensors). Skips HF Hub download.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_UNET_WEIGHTS", value_name = "FILE"))]
    pub unet_weights: Option<String>,
    /// Path to a local primary CLIP weight file (.safetensors).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_CLIP_WEIGHTS", value_name = "FILE"))]
    pub clip_weights: Option<String>,
    /// Path to a local secondary CLIP weight file (.safetensors, SDXL only).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_CLIP2_WEIGHTS", value_name = "FILE"))]
    pub clip2_weights: Option<String>,
    /// Path to a local VAE weight file (.safetensors).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_VAE_WEIGHTS", value_name = "FILE"))]
    pub vae_weights: Option<String>,
    /// Path to a local tokenizer file.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_TOKENIZER", value_name = "FILE"))]
    pub tokenizer: Option<String>,
    /// Path to an image used to initialize the latents (img2img / inpainting).
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_IMG2IMG", value_name = "FILE"))]
    pub img2img: Option<PathBuf>,
    /// img2img transformation strength in the range 0.0–1.0.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_IMG2IMG_STRENGTH"))]
    pub img2img_strength: Option<f64>,
    /// RNG seed for reproducible generation.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_SEED"))]
    pub seed: Option<u64>,
    /// Path to an inpainting mask image.
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_SD_MASK_PATH", value_name = "FILE"))]
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

    /// Helper: write `contents` to a temp file and parse it as [`GenConfig`].
    fn parse_toml(contents: &str) -> GenConfig {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        load(Some(f.path())).unwrap()
    }

    #[test]
    fn empty_toml_deserializes_to_defaults() {
        let cfg = parse_toml("");
        assert!(cfg.prompt.is_none());
        assert!(cfg.sd_version.is_none());
        assert!(cfg.device.is_none());
    }

    #[test]
    fn partial_toml_sets_only_specified_fields() {
        let cfg = parse_toml(
            r#"
            prompt = "a cat"
            num_samples = 3
        "#,
        );
        assert_eq!(cfg.prompt.as_deref(), Some("a cat"));
        assert_eq!(cfg.num_samples, Some(3));
        assert!(cfg.sd_version.is_none());
    }

    #[test]
    fn full_template_deserializes_correctly() {
        let template = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../configs/image-gen.toml"
        ))
        .expect("configs/image-gen.toml template must exist");
        let cfg: GenConfig = toml::from_str(&template).expect("template must parse");

        // Spot-check a few fields from the template.
        assert!(cfg.prompt.is_some());
        assert!(cfg.sd_version.is_some());
        assert!(cfg.num_samples.is_some());
        assert!(cfg.guidance_scale.is_some());
    }

    #[test]
    fn full_template_snapshot() {
        let template = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../configs/image-gen.toml"
        ))
        .expect("configs/image-gen.toml template must exist");
        let cfg: GenConfig = toml::from_str(&template).expect("template must parse");
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn nonexistent_path_returns_default() {
        let cfg =
            load(Some(std::path::Path::new("/tmp/djinn-nonexistent-config-xyz.toml"))).unwrap();
        assert!(cfg.prompt.is_none());
    }
}
