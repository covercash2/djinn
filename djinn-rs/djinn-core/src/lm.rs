use clap::{Parser, Subcommand};

use crate::{llama, mistral};

#[derive(Parser)]
pub struct Args {
    prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 100)]
    sample_len: usize,

    #[command(subcommand)]
    model: LanguageModel,
}

/// Language models
#[derive(Subcommand)]
pub enum LanguageModel {
    Llama(llama::Args),
    Mistral(mistral::Args),
}
