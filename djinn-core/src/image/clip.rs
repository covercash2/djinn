use std::path::PathBuf;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::hf_hub_ext::{Hub, HubError};

pub type ClipResult<T> = std::result::Result<T, ClipError>;

#[derive(Debug, thiserror::Error)]
pub enum ClipError {
    #[error("unable to load hub: {0}")]
    InitHub(HubError),

    #[error("couldn't download tokenizer: {0}")]
    DownloadTokenizer(HubError),

    #[error("couldn't load tokenizer from file {path}: {source}")]
    LoadTokenizer {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("unable to encode sequence: {0}")]
    TokenizerEncode(tokenizers::Error),

    #[error("missing tokenizer file")]
    MissingTokenizerFile,

    #[error("couln't create tensor: {0}")]
    TensorCreation(candle_core::Error),

    #[error("missing pad token in tokenizer")]
    MissingPadToken,
}

pub struct ModelFile {
    name: String,
    revision: String,
    file: String,
}

impl ModelFile {
    pub fn new(
        name: impl Into<String>,
        revision: impl Into<String>,
        file: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            revision: revision.into(),
            file: file.into(),
        }
    }

    pub fn clip_tokenizer() -> Self {
        Self::new(
            "openai/clip-vit-base-patch32",
            "refs/pr/15",
            "tokenizer/tokenizer.json",
        )
    }

    pub fn clip_model() -> Self {
        Self::new(
            "openai/clip-vit-base-patch32",
            "refs/pr/15",
            "model.safetensors",
        )
    }
}

pub struct ClipArgs {
    pub tokenizer: PathBuf,
    pub sequences: Option<Vec<String>>,
    pub device: Device,
}

pub struct Clip {
    #[allow(dead_code)] tokenizer: Tokenizer,
    #[allow(dead_code)] device: Device,
}

impl Clip {
    pub async fn new(args: ClipArgs) -> ClipResult<Self> {
        let tokenizer_file = if args.tokenizer.exists() {
            args.tokenizer
        } else {
            let hub = Hub::new().await.map_err(ClipError::InitHub)?;
            load_tokenizer(hub).await?
        };

        let tokenizer = Tokenizer::from_file(tokenizer_file.as_path()).map_err(|source| {
            ClipError::LoadTokenizer {
                path: tokenizer_file.clone(),
                source,
            }
        })?;

        Ok(Self {
            tokenizer,
            device: args.device,
        })
    }
}

async fn load_tokenizer(hub: Hub) -> ClipResult<PathBuf> {
    let model_file = ModelFile::clip_tokenizer();

    let tokenizer_path = hub
        .get_model_file(model_file.name, model_file.revision, &model_file.file)
        .await
        .map_err(ClipError::DownloadTokenizer)?;

    Ok(tokenizer_path)
}

#[allow(dead_code)] async fn load_model(hub: Hub) -> ClipResult<PathBuf> {
    let model_file = ModelFile::clip_tokenizer();

    let model_path = hub
        .get_model_file(model_file.name, model_file.revision, &model_file.file)
        .await
        .map_err(ClipError::DownloadTokenizer)?;

    Ok(model_path)
}

#[allow(dead_code)] fn tokenize_sequences(
    sequences: Vec<String>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> ClipResult<(Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(ClipError::MissingPadToken)?;
    let mut tokens = vec![];
    for seq in sequences.clone() {
        let encoding = tokenizer
            .encode(seq, true)
            .map_err(ClipError::TokenizerEncode)?;
        tokens.push(encoding.get_ids().to_vec());
    }
    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }
    let input_ids = Tensor::new(tokens, device).map_err(ClipError::TensorCreation)?;
    Ok((input_ids, sequences))
}
