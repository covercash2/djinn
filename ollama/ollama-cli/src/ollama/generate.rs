use clap::Parser;
use futures::StreamExt as _;
use ollama_rs::generation::completion::request::GenerationRequest;
use tokio::io::AsyncWriteExt as _;

use super::{Client, ModelName};

#[derive(Parser)]
pub struct Request {
    pub prompt: String,
    #[arg(default_value_t)]
    pub model: ModelName,
    pub system: Option<String>,
}

impl From<Request> for GenerationRequest {
    fn from(value: Request) -> Self {
        let mut builder = GenerationRequest::new(value.model.to_string(), value.prompt);
        builder.system = value.system;

        builder
    }
}

impl Client {
    pub async fn generate(&self, request: Request) -> anyhow::Result<()> {
        let request: GenerationRequest = request.into();
        let mut stream = self.client.generate_stream(request).await?;
        let mut output_sink = tokio::io::stdout();

        while let Some(responses) = stream.next().await {
            for response in responses? {
                output_sink.write_all(response.response.as_bytes()).await?;
            }
        }

        Ok(())
    }
}
