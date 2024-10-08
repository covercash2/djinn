use std::ops::DerefMut;
use std::sync::Arc;

use axum::extract::State;
use djinn_core::lm::config::RunConfig;
use futures::{pin_mut, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{instrument, Instrument};

use crate::error::{Error, Result};
use crate::server::{Context, Json};

pub const ROUTE_COMPLETE: &str = "/complete";

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteRequest {
    prompt: String,
    #[serde(default, flatten)]
    config: RunConfig,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteResponse {
    prompt: String,
    output: String,
}

#[instrument(skip(model_context))]
pub async fn complete(
    State(model_context): State<Arc<Mutex<Context>>>,
    Json(payload): Json<CompleteRequest>,
) -> Result<Json<CompleteResponse>> {
    let span = tracing::info_span!("complete JSON");

    let mut lock = model_context.lock().instrument(span).await;
    tracing::info!("got model lock");

    let context: &mut Context = lock.deref_mut();

    let response = run_model(context, payload).await?;

    Ok(Json(response))
}

#[instrument(skip(model_context))]
async fn run_model(
    model_context: &mut Context,
    request: CompleteRequest,
) -> Result<CompleteResponse> {
    let prompt = request.prompt;

    let config = request.config;

    // setup output stream
    let stream = model_context.model.run(prompt.clone(), config);

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
    let response = CompleteResponse { prompt, output };

    tracing::info!("sending response: {response:?}");

    Ok(response)
}
