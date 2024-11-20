use std::sync::Arc;

use clap::Parser;
use futures::StreamExt;
use ollama_rs::generation::completion::{request::GenerationRequest, GenerationResponseStream};
use tokio::io::AsyncWriteExt as _;

use super::{Client, ModelName};
use crate::error::Result;

#[derive(Parser)]
pub struct Request {
    pub prompt: Arc<str>,
    #[arg(default_value_t)]
    pub model: ModelName,
    pub system: Option<String>,
}

impl From<Request> for GenerationRequest {
    fn from(value: Request) -> Self {
        let mut builder = GenerationRequest::new(value.model.to_string(), value.prompt.to_string());
        builder.system = value.system;

        builder
    }
}

impl Client {
    pub async fn generate(&self, request: Request) -> Result<GenerationResponseStream> {
        let request: GenerationRequest = request.into();
        Ok(self.client.generate_stream(request).await?)
    }

    pub async fn generate_stdout(&self, request: Request) -> anyhow::Result<()> {
        let mut stream = self.generate(request).await?;
        let mut output_sink = tokio::io::stdout();
        while let Some(responses) = stream.next().await {
            for response in responses? {
                output_sink.write_all(response.response.as_bytes()).await?;
            }
        }

        Ok(())
    }
}
