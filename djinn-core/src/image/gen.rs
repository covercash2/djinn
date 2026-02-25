use std::ops::Div;
use std::path::{Path, PathBuf};

use bon::builder;
use candle_core::{DType, Device as CandleDevice, IndexOp as _, Result as CandleResult, Tensor, D};
use candle_nn::Module as _;
use candle_transformers::models::stable_diffusion::{self, vae::AutoEncoderKL};
use clap::Parser;
use hf_hub::api::sync::{Api, ApiError};
use rand::Rng as _;
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt as _;

use crate::device::Device;
use super::error::Error;

/// CLI arguments for Stable Diffusion image generation.
///
/// All settings can be provided via CLI flag, environment variable, or a config
/// file at `$XDG_CONFIG_HOME/djinn/image-gen.toml`.  The priority order is:
/// config file < env var < explicit CLI flag.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to a TOML config file. Defaults to `$XDG_CONFIG_HOME/djinn/image-gen.toml`.
    #[arg(long, env = "DJINN_SD_CONFIG", value_name = "FILE")]
    config_file: Option<PathBuf>,

    /// The prompt to be used for image generation.
    #[arg(long, env = "DJINN_SD_PROMPT")]
    prompt: Option<String>,
    /// Negative prompt — describe what to exclude from the image.
    #[arg(long, env = "DJINN_SD_UNCOND_PROMPT")]
    uncond_prompt: Option<String>,
    /// Compute device to run inference on.
    #[arg(long, env = "DJINN_SD_DEVICE")]
    device: Option<Device>,
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long, env = "DJINN_SD_TRACING")]
    tracing: bool,
    /// The height in pixels of the generated image.
    #[arg(long, env = "DJINN_SD_HEIGHT")]
    height: Option<usize>,
    /// The width in pixels of the generated image.
    #[arg(long, env = "DJINN_SD_WIDTH")]
    width: Option<usize>,
    /// The UNet weight file, in .safetensors format.
    #[arg(long, env = "DJINN_SD_UNET_WEIGHTS", value_name = "FILE")]
    unet_weights: Option<String>,
    /// The CLIP weight file, in .safetensors format.
    #[arg(long, env = "DJINN_SD_CLIP_WEIGHTS", value_name = "FILE")]
    clip_weights: Option<String>,
    /// The CLIP2 weight file, in .safetensors format.
    #[arg(long, env = "DJINN_SD_CLIP2_WEIGHTS", value_name = "FILE")]
    clip2_weights: Option<String>,
    /// The VAE weight file, in .safetensors format.
    #[arg(long, env = "DJINN_SD_VAE_WEIGHTS", value_name = "FILE")]
    vae_weights: Option<String>,
    /// The tokenizer file.
    #[arg(long, env = "DJINN_SD_TOKENIZER", value_name = "FILE")]
    tokenizer: Option<String>,
    /// The size of the sliced attention or 0 for automatic slicing (disabled by default).
    #[arg(long, env = "DJINN_SD_SLICED_ATTENTION_SIZE")]
    sliced_attention_size: Option<usize>,
    /// The number of steps to run the diffusion for.
    #[arg(long, env = "DJINN_SD_N_STEPS")]
    n_steps: Option<usize>,
    /// The number of samples to generate iteratively.
    #[arg(long, env = "DJINN_SD_NUM_SAMPLES")]
    num_samples: Option<usize>,
    /// The number of samples to generate simultaneously (batch size).
    #[arg(long, env = "DJINN_SD_BSIZE")]
    bsize: Option<usize>,
    /// The name of the final image to generate.
    #[arg(long, env = "DJINN_SD_FINAL_IMAGE", value_name = "FILE")]
    final_image: Option<PathBuf>,
    /// Stable Diffusion model version.
    #[arg(long, value_enum, env = "DJINN_SD_VERSION")]
    sd_version: Option<StableDiffusionVersion>,
    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,
    /// Use flash attention (CUDA only).
    #[arg(long, env = "DJINN_SD_USE_FLASH_ATTN")]
    use_flash_attn: bool,
    /// Use FP16 weights to reduce VRAM usage.
    #[arg(long, env = "DJINN_SD_USE_F16")]
    use_f16: bool,
    /// Classifier-free guidance scale.
    #[arg(long, env = "DJINN_SD_GUIDANCE_SCALE")]
    guidance_scale: Option<f64>,
    /// Path to the mask image for inpainting.
    #[arg(long, env = "DJINN_SD_MASK_PATH", value_name = "FILE")]
    mask_path: Option<String>,
    /// Path to the image used to initialize the latents. For inpainting, this is the image to be masked.
    #[arg(long, env = "DJINN_SD_IMG2IMG", value_name = "FILE")]
    img2img: Option<PathBuf>,
    /// How much to transform the initial image (0.0 = no change, 1.0 = full diffusion).
    #[arg(long, env = "DJINN_SD_IMG2IMG_STRENGTH")]
    img2img_strength: Option<f64>,
    /// The seed to use when generating random samples.
    #[arg(long, env = "DJINN_SD_SEED")]
    seed: Option<u64>,
    /// Force the saved image to update only the masked region.
    #[arg(long)]
    only_update_masked: bool,
}

/// Supported Stable Diffusion model versions.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StableDiffusionVersion {
    /// Stable Diffusion v1.5
    V1_5,
    /// Stable Diffusion v1.5 inpainting variant
    V1_5Inpaint,
    /// Stable Diffusion v2.1
    V2_1,
    /// Stable Diffusion v2 inpainting variant
    V2Inpaint,
    /// Stable Diffusion XL base
    Xl,
    /// Stable Diffusion XL inpainting variant
    XlInpaint,
    /// SDXL Turbo (single-step distilled)
    Turbo,
}

impl StableDiffusionVersion {
    /// get the model repository for the variant.
    fn repo(&self) -> &'static str {
        match self {
            Self::XlInpaint => "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2Inpaint => "stabilityai/stable-diffusion-2-inpainting",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::V1_5Inpaint => "stable-diffusion-v1-5/stable-diffusion-inpainting",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    /// get the UNet file for the variant.
    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5
            | Self::V1_5Inpaint
            | Self::V2_1
            | Self::V2Inpaint
            | Self::Xl
            | Self::XlInpaint
            | Self::Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    /// get the [VAE] file for the variant.
    ///
    /// [VAE]: https://en.m.wikipedia.org/wiki/Variational_autoencoder
    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5
            | Self::V1_5Inpaint
            | Self::V2_1
            | Self::V2Inpaint
            | Self::Xl
            | Self::XlInpaint
            | Self::Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    /// get the CLIP file for the variant.
    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5
            | Self::V1_5Inpaint
            | Self::V2_1
            | Self::V2Inpaint
            | Self::Xl
            | Self::XlInpaint
            | Self::Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    /// get the CLIP2 file for the variant.
    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5
            | Self::V1_5Inpaint
            | Self::V2_1
            | Self::V2Inpaint
            | Self::Xl
            | Self::XlInpaint
            | Self::Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

/// Weight and tokenizer files required for a Stable Diffusion pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    /// Primary CLIP tokenizer
    Tokenizer,
    /// Secondary CLIP tokenizer (SDXL only)
    Tokenizer2,
    /// Primary CLIP text encoder weights
    Clip,
    /// Secondary CLIP text encoder weights (SDXL only)
    Clip2,
    /// Variational autoencoder weights
    Vae,
    /// UNet denoising model weights
    Unet,
}

impl ModelFile {
    /// download the model file from HuggingFace Hub
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf, ApiError> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5
                            | StableDiffusionVersion::V2_1
                            | StableDiffusionVersion::V1_5Inpaint
                            | StableDiffusionVersion::V2Inpaint => "openai/clip-vit-base-patch32",
                            StableDiffusionVersion::Xl
                            | StableDiffusionVersion::XlInpaint
                            | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

/// generate a filename for the output image.
///
/// ```rust
/// use djinn_core::image::gen::output_filename;
///
/// let filename = output_filename("output", 0, 1, None);
/// assert_eq!(filename, "output.png");
///
/// let filename = output_filename("output", 0, 2, None);
/// assert_eq!(filename, "output.0.png");
///
/// let filename = output_filename("output.png", 0, 2, Some(0));
/// assert_eq!(filename, "output-0.png");
/// ```
pub fn output_filename(
    basename: &str,
    sample_idx: usize,
    num_samples: usize,
    timestep_idx: Option<usize>,
) -> String {
    let mut base = match basename.rsplit_once('.') {
        Some((without_ext, ext)) => {
            if ext == "png" {
                without_ext.to_string()
            } else {
                basename.to_string()
            }
        }
        None => basename.to_string(),
    };

    if num_samples > 1 && timestep_idx.is_none() {
        base = format!("{base}.{sample_idx}");
    }

    if let Some(timestep_idx) = timestep_idx {
        base = format!("{base}-{timestep_idx}");
    }

    format!("{base}.png")
}

/// decode the latents and save the generated image.
/// save the generated image to disk.
#[builder]
fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> CandleResult<()> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&CandleDevice::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..bsize {
        let image = images.i(batch)?;
        let image_filename = output_filename(
            final_image,
            (bsize * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        super::save_image(&image, image_filename)?;
    }
    Ok(())
}

/// Encodes a prompt (and optional negative prompt) into CLIP text embeddings.
///
/// When `use_guide_scale` is true, concatenates unconditional and conditional
/// embeddings along the batch dimension for classifier-free guidance.
#[builder]
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    clip2_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &CandleDevice,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor, Error> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer_path = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|source| Error::TokenizerLoad { path: tokenizer_path.clone(), source })?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer
            .get_vocab(true)
            .get(padding.as_str())
            .ok_or_else(|| Error::MissingToken { token: padding.to_string() })?,
        None => *tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| Error::MissingToken { token: "<|endoftext|>".to_string() })?,
    };
    tracing::info!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::Tokenize)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        return Err(Error::PromptTooLong {
            len: tokens.len(),
            max: sd_config.clip.max_position_embeddings,
        });
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    tracing::info!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = if first {
        clip_weights_file.get(clip_weights, sd_version, use_f16)?
    } else {
        clip_weights_file.get(clip2_weights, sd_version, use_f16)?
    };
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().ok_or(Error::MissingClip2Config)?
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(Error::Tokenize)?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            return Err(Error::PromptTooLong {
                len: uncond_tokens.len(),
                max: sd_config.clip.max_position_embeddings,
            });
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

/// open an image from a file and load it into a [`Tensor`].
fn image_preprocess(path: impl AsRef<Path>) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &CandleDevice::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

/// convert the mask image to a single channel tensor.
/// also ensure the image is a multiple of 32 in both dimensions.
fn mask_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let img = image::open(path)?.to_luma8();
    let (new_width, new_height) = {
        let (width, height) = img.dimensions();
        (width - width % 32, height - height % 32)
    };
    let img = image::imageops::resize(
        &img,
        new_width,
        new_height,
        image::imageops::FilterType::CatmullRom,
    )
    .into_raw();
    let mask = Tensor::from_vec(
        img,
        (new_height as usize, new_width as usize),
        &CandleDevice::Cpu,
    )?
    .unsqueeze(0)?
    .to_dtype(DType::F32)?
    .div(255.0)?
    .unsqueeze(0)?;
    Ok(mask)
}

/// Generates the mask latents, scaled mask and mask_4 for inpainting. Returns a tuple of None if inpainting is not
/// being used.
#[builder]
fn inpainting_tensors(
    sd_version: StableDiffusionVersion,
    mask_path: Option<String>,
    dtype: DType,
    device: &CandleDevice,
    use_guide_scale: bool,
    vae: &AutoEncoderKL,
    image: Option<Tensor>,
    vae_scale: f64,
) -> anyhow::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
    match sd_version {
        StableDiffusionVersion::XlInpaint
        | StableDiffusionVersion::V2Inpaint
        | StableDiffusionVersion::V1_5Inpaint => {
            let inpaint_mask = mask_path.ok_or_else(|| {
                anyhow::anyhow!("An inpainting model was requested but mask-path is not provided.")
            })?;
            // Get the mask image with shape [1, 1, 128, 128]
            let mask = mask_preprocess(inpaint_mask)?
                .to_device(device)?
                .to_dtype(dtype)?;
            // Generate the masked image from the image and the mask with shape [1, 3, 1024, 1024]
            let xmask = mask.le(0.5)?.repeat(&[1, 3, 1, 1])?.to_dtype(dtype)?;
            let image = &image
                .ok_or_else(|| anyhow::anyhow!(
                    "An inpainting model was requested but img2img which is used as the input image is not provided."
                ))?;
            let masked_img = (image * xmask)?;
            // Scale down the mask
            let shape = masked_img.shape();
            let (w, h) = (shape.dims()[3] / 8, shape.dims()[2] / 8);
            let mask = mask.interpolate2d(w, h)?;
            // shape: [1, 4, 128, 128]
            let mask_latents = vae.encode(&masked_img)?;
            let mask_latents = (mask_latents.sample()? * vae_scale)?.to_device(device)?;

            let mask_4 = mask.as_ref().repeat(&[1, 4, 1, 1])?;
            let (mask_latents, mask) = if use_guide_scale {
                (
                    Tensor::cat(&[&mask_latents, &mask_latents], 0)?,
                    Tensor::cat(&[&mask, &mask], 0)?,
                )
            } else {
                (mask_latents, mask)
            };
            Ok((Some(mask_latents), Some(mask), Some(mask_4)))
        }
        _ => Ok((None, None, None)),
    }
}

impl Args {
    /// Run the full Stable Diffusion pipeline and write the output image(s) to disk.
    pub fn run(self) -> anyhow::Result<()> {
        // Load file config from the XDG path (or --config-file override).
        // clap has already merged env vars into `self`, so the priority chain is:
        //   config file  <  env var (via clap)  <  explicit CLI flag
        let file_config = super::config::load(self.config_file.as_deref())?;

        let prompt = self.prompt
            .or(file_config.prompt)
            .ok_or_else(|| anyhow::anyhow!(
                "prompt is required — pass --prompt, set DJINN_SD_PROMPT, \
                 or add `prompt = \"...\"` to the config file"
            ))?;
        let uncond_prompt = self.uncond_prompt.or(file_config.uncond_prompt).unwrap_or_default();
        let device      = self.device.or(file_config.device).unwrap_or_default();
        let height      = self.height.or(file_config.height);
        let width       = self.width.or(file_config.width);
        let n_steps     = self.n_steps.or(file_config.n_steps);
        let tokenizer   = self.tokenizer.or(file_config.tokenizer);
        let final_image = self.final_image.or(file_config.final_image)
            .unwrap_or_else(|| PathBuf::from("sd_final.png"));
        let sliced_attention_size = self.sliced_attention_size.or(file_config.sliced_attention_size);
        let num_samples = self.num_samples.or(file_config.num_samples).unwrap_or(1);
        let bsize       = self.bsize.or(file_config.bsize).unwrap_or(1);
        let sd_version  = self.sd_version.or(file_config.sd_version)
            .unwrap_or(StableDiffusionVersion::V2_1);
        let clip_weights  = self.clip_weights.or(file_config.clip_weights);
        let clip2_weights = self.clip2_weights.or(file_config.clip2_weights);
        let vae_weights   = self.vae_weights.or(file_config.vae_weights);
        let unet_weights  = self.unet_weights.or(file_config.unet_weights);
        let tracing       = self.tracing;
        let use_f16       = self.use_f16 || file_config.use_f16.unwrap_or(false);
        let guidance_scale    = self.guidance_scale.or(file_config.guidance_scale);
        let use_flash_attn    = self.use_flash_attn || file_config.use_flash_attn.unwrap_or(false);
        let mask_path         = self.mask_path.or(file_config.mask_path);
        let img2img           = self.img2img.or(file_config.img2img);
        let img2img_strength  = self.img2img_strength.or(file_config.img2img_strength).unwrap_or(0.8);
        let seed              = self.seed.or(file_config.seed);
        let only_update_masked  = self.only_update_masked;
        let intermediary_images = self.intermediary_images;

        if !(0. ..=1.).contains(&img2img_strength) {
            anyhow::bail!("img2img-strength should be between 0 and 1, got {img2img_strength}")
        }

        let _guard = if tracing {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };

        let guidance_scale = match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V1_5Inpaint
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::V2Inpaint
                | StableDiffusionVersion::XlInpaint
                | StableDiffusionVersion::Xl => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };
        let n_steps = match n_steps {
            Some(n_steps) => n_steps,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V1_5Inpaint
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::V2Inpaint
                | StableDiffusionVersion::XlInpaint
                | StableDiffusionVersion::Xl => 30,
                StableDiffusionVersion::Turbo => 1,
            },
        };
        let dtype = if use_f16 { DType::F16 } else { DType::F32 };
        let sd_config = match sd_version {
            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V1_5Inpaint => {
                stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::V2_1 | StableDiffusionVersion::V2Inpaint => {
                stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Xl | StableDiffusionVersion::XlInpaint => {
                stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
                sliced_attention_size,
                height,
                width,
            ),
        };

        let mut scheduler = sd_config.build_scheduler(n_steps)?;
        let device: CandleDevice = device.try_into()?;
        // If a seed is not given, generate a random seed and print it
        let seed = seed.unwrap_or(rand::rng().random_range(0u64..u64::MAX));
        println!("Using seed {seed}");
        device.set_seed(seed)?;
        let use_guide_scale = guidance_scale > 1.0;

        let which = match sd_version {
            StableDiffusionVersion::Xl
            | StableDiffusionVersion::XlInpaint
            | StableDiffusionVersion::Turbo => vec![true, false],
            _ => vec![true],
        };
        let text_embeddings = which
            .iter()
            .map(|first| {
                text_embeddings()
                    .prompt(&prompt)
                    .uncond_prompt(&uncond_prompt)
                    .maybe_tokenizer(tokenizer.clone())
                    .maybe_clip_weights(clip_weights.clone())
                    .maybe_clip2_weights(clip2_weights.clone())
                    .sd_version(sd_version)
                    .sd_config(&sd_config)
                    .use_f16(use_f16)
                    .device(&device)
                    .dtype(dtype)
                    .use_guide_scale(use_guide_scale)
                    .first(*first)
                    .call()
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;
        println!("{text_embeddings:?}");

        println!("Building the autoencoder.");
        let vae_weights = ModelFile::Vae.get(vae_weights, sd_version, use_f16)?;
        let vae = sd_config.build_vae(vae_weights, &device, dtype)?;

        let (image, init_latent_dist) = match &img2img {
            None => (None, None),
            Some(image) => {
                let image = image_preprocess(image)?
                    .to_device(&device)?
                    .to_dtype(dtype)?;
                (Some(image.clone()), Some(vae.encode(&image)?))
            }
        };

        println!("Building the unet.");
        let unet_weights = ModelFile::Unet.get(unet_weights, sd_version, use_f16)?;
        let in_channels = match sd_version {
            StableDiffusionVersion::XlInpaint
            | StableDiffusionVersion::V2Inpaint
            | StableDiffusionVersion::V1_5Inpaint => 9,
            _ => 4,
        };
        let unet =
            sd_config.build_unet(unet_weights, &device, in_channels, use_flash_attn, dtype)?;

        let t_start = if img2img.is_some() {
            n_steps - (n_steps as f64 * img2img_strength) as usize
        } else {
            0
        };

        let vae_scale = match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V1_5Inpaint
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::V2Inpaint
            | StableDiffusionVersion::XlInpaint
            | StableDiffusionVersion::Xl => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };

        let (mask_latents, mask, mask_4) = inpainting_tensors()
            .sd_version(sd_version)
            .maybe_mask_path(mask_path)
            .dtype(dtype)
            .device(&device)
            .use_guide_scale(use_guide_scale)
            .vae(&vae)
            .maybe_image(image.clone())
            .vae_scale(vae_scale)
            .call()?;

        for idx in 0..num_samples {
            let timesteps = scheduler.timesteps().to_vec();
            let latents = match &init_latent_dist {
                Some(init_latent_dist) => {
                    let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        scheduler.add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }
                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        (bsize, 4, sd_config.height / 8, sd_config.width / 8),
                        &device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(dtype)?;

            println!("starting sampling");
            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let start_time = std::time::Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                let latent_model_input =
                    scheduler.scale_model_input(latent_model_input, timestep)?;

                let latent_model_input = match sd_version {
                    StableDiffusionVersion::XlInpaint
                    | StableDiffusionVersion::V2Inpaint
                    | StableDiffusionVersion::V1_5Inpaint => Tensor::cat(
                        &[
                            &latent_model_input,
                            mask.as_ref().unwrap(),
                            mask_latents.as_ref().unwrap(),
                        ],
                        1,
                    )?,
                    _ => latent_model_input,
                }
                .to_device(&device)?;

                let noise_pred =
                    unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond
                        + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                let dt = start_time.elapsed().as_secs_f32();
                println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

                // Replace all pixels in the unmasked region with the original pixels discarding any changes.
                if only_update_masked {
                    let mask = mask_4.as_ref().unwrap();
                    let latent_to_keep = mask_latents
                        .as_ref()
                        .unwrap()
                        .get_on_dim(0, 0)? // shape: [4, H, W]
                        .unsqueeze(0)?; // shape: [1, 4, H, W]

                    latents = ((&latents * mask)? + &latent_to_keep * (1.0 - mask))?;
                }

                if intermediary_images {
                    save_image()
                        .vae(&vae)
                        .latents(&latents)
                        .vae_scale(vae_scale)
                        .bsize(bsize)
                        .idx(idx)
                        .final_image(final_image.to_str().unwrap())
                        .num_samples(num_samples)
                        .timestep_ids(timestep_index + 1)
                        .call()?;
                }
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );
            save_image()
                .vae(&vae)
                .latents(&latents)
                .vae_scale(vae_scale)
                .bsize(bsize)
                .idx(idx)
                .final_image(final_image.to_str().unwrap())
                .num_samples(num_samples)
                .call()?;
        }
        Ok(())
    }
}
