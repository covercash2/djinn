use std::ops::DerefMut;
use std::sync::Arc;

use axum::extract::State;
use djinn_core::lm::Lm;
use futures_util::{pin_mut, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{instrument, Instrument};

use crate::error::{Error, Result};
use crate::server::{Context, Json};

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteRequest {
    prompt: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteResponse {
    output: String,
}

#[instrument(skip(model_context))]
pub async fn complete(
    State(model_context): State<Arc<Mutex<Context>>>,
    Json(payload): Json<CompleteRequest>,
) -> Result<Json<CompleteResponse>> {
    let span = tracing::info_span!("complete");

    let mut lock = model_context.lock().instrument(span).await;
    tracing::info!("got model lock");

    let context: &mut Context = lock.deref_mut();

    // setup output stream
    let stream = context
        .model
        .run(payload.prompt, context.run_config.clone());

    // consume the stream
    pin_mut!(stream);
    let mut output = String::new();
    while let Some(value) = stream.next().await {
        value
            .map(|string_token| {
                output.push_str(&string_token);
                tracing::trace!("{string_token}");
            })
            .map_err(Error::from)?;
    }
    let response = CompleteResponse { output };

    tracing::info!("sending response: {response:?}");

    Ok(Json(response))
}
