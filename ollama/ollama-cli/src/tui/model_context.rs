use std::sync::Arc;

use futures::StreamExt;
use tokio::{
    sync::mpsc::{Receiver, Sender},
    task::JoinHandle,
};
use tracing::instrument;

use crate::{
    error::Result,
    lm::{Prompt, Response},
    ollama::{self, chat::ChatRequest, generate::Request},
};

#[derive(Debug)]
pub struct ModelContext {
    _handle: JoinHandle<Result<()>>,
    pub prompt_sender: Sender<Prompt>,
    pub response_receiver: Receiver<Response>,
}

impl ModelContext {
    pub fn spawn(client: ollama::Client) -> ModelContext {
        let (prompt_sender, mut prompt_receiver): (Sender<Prompt>, Receiver<Prompt>) =
            tokio::sync::mpsc::channel(5);
        let (response_sender, response_receiver) = tokio::sync::mpsc::channel(20);

        let context = ModeContext {
            client,
            response_sender,
        };

        let handle: JoinHandle<Result<()>> = tokio::spawn(async move {
            while let Some(prompt) = prompt_receiver.recv().await {
                match prompt {
                    Prompt::Generate(string) => context.handle_generate_mode(string).await?,
                    Prompt::Chat(request) => context.handle_chat_mode(request).await?,
                }
            }

            Ok(())
        });

        ModelContext {
            _handle: handle,
            prompt_sender,
            response_receiver,
        }
    }
}

#[derive(Debug)]
pub struct ModeContext {
    pub client: ollama::Client,
    pub response_sender: Sender<Response>,
}

impl ModeContext {
    async fn handle_generate_mode(&self, prompt: Arc<str>) -> Result<()> {
        let result = self
            .client
            .generate(Request {
                prompt,
                model: Default::default(),
                system: None,
            })
            .await;

        match result {
            Ok(mut stream) => {
                while let Some(responses) = stream.next().await {
                    match responses {
                        Ok(responses) => {
                            for response in responses {
                                self.response_sender
                                    .send(Response::Token(response.response.into()))
                                    .await?
                            }
                        }
                        Err(error) => {
                            self.response_sender
                                .send(Response::Error(error.to_string().into()))
                                .await?;
                        }
                    }
                }
                self.response_sender.send(Response::Eos).await?;
            }
            Err(error) => {
                self.response_sender
                    .send(Response::Error(error.to_string().into()))
                    .await?;
            }
        }
        Ok(())
    }

    #[instrument]
    async fn handle_chat_mode(&self, prompt: ChatRequest) -> Result<()> {
        let result = self.client.chat(prompt).await;

        match result {
            Ok(mut stream) => {
                while let Some(responses) = stream.next().await {
                    match responses {
                        Ok(response) => {
                            if let Some(response) = response.message {
                                self.response_sender
                                    .send(Response::Token(response.content.into()))
                                    .await?
                            }
                        }
                        Err(()) => {
                            self.response_sender
                                .send(Response::Error("error in response".into()))
                                .await?;
                        }
                    }
                }
                self.response_sender.send(Response::Eos).await?;
            }
            Err(error) => {
                self.response_sender
                    .send(Response::Error(error.to_string().into()))
                    .await?;
            }
        }
        Ok(())
    }
}
