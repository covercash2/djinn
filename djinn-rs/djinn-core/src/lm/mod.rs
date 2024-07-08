//! Language Models and configurations
use clap::{Parser, Subcommand};
use futures_util::Stream;

use crate::error::Result;

pub mod mistral;

// #[derive(Parser)]
// pub struct Args {
//     prompt: String,
//     /// The seed to use when generating random samples.
//     #[arg(long, default_value_t = 299792458)]
//     seed: u64,
//     /// The length of the sample to generate (in tokens).
//     #[arg(long, short = 'n', default_value_t = 100)]
//     sample_len: usize,
//
//     #[command(subcommand)]
//     model: LanguageModel,
// }

/// Language models
// #[derive(Subcommand)]
// pub enum LanguageModel {
//     Mistral(mistral::Args),
// }

pub trait Lm {
    type Config;

    fn run(
        &mut self,
        prompt: String,
        config: Self::Config,
    ) -> impl Stream<Item = Result<String>> + '_;
}
