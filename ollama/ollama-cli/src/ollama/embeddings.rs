use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;

use super::{generate::Request, Client};

impl From<Request> for GenerateEmbeddingsRequest {
    fn from(value: Request) -> Self {
        GenerateEmbeddingsRequest::new(value.model.to_string(), value.prompt.into())
    }
}

impl Client {
    pub async fn embed(&self, request: Request) -> anyhow::Result<Vec<Vec<f32>>> {
        let request: GenerateEmbeddingsRequest = request.into();

        let response = self.client.generate_embeddings(request).await?;

        Ok(response.embeddings)
    }
}
