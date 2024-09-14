use std::sync::Arc;

use futures::StreamExt;
use tokio::{
    sync::mpsc::{Receiver, Sender},
    task::JoinHandle,
};

use crate::{
    error::Result,
    lm::Response,
    ollama::{self, generate::Request},
};

pub struct ModelContext {
    _handle: JoinHandle<Result<()>>,
    pub prompt_sender: Sender<Arc<str>>,
    pub response_receiver: Receiver<Response>,
}

impl ModelContext {
    pub fn spawn(client: ollama::Client) -> ModelContext {
        let (prompt_sender, mut prompt_rx): (Sender<Arc<str>>, Receiver<Arc<str>>) =
            tokio::sync::mpsc::channel(5);
        let (response_tx, response_receiver) = tokio::sync::mpsc::channel(20);

        let handle: JoinHandle<Result<()>> = tokio::spawn(async move {
            while let Some(prompt) = prompt_rx.recv().await {
                // add to queue
                let result = client
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
                                        response_tx
                                            .send(Response::Token(response.response.into()))
                                            .await?
                                    }
                                }
                                Err(error) => {
                                    response_tx
                                        .send(Response::Error(error.to_string().into()))
                                        .await?;
                                }
                            }
                        }
                        response_tx.send(Response::Eos).await?;
                    }
                    Err(error) => {
                        response_tx
                            .send(Response::Error(error.to_string().into()))
                            .await?;
                    }
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
