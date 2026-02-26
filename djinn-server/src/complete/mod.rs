use std::convert::Infallible;
use std::ops::DerefMut;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use djinn_core::lm::config::RunConfig;
use futures::{pin_mut, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{instrument, Instrument};

use crate::error::{Error, Result};
use crate::server::{Context, Json};

pub const ROUTE_COMPLETE: &str = "/complete";
pub const ROUTE_COMPLETE_STREAM: &str = "/complete/stream";

#[derive(Serialize, Deserialize, Debug, utoipa::ToSchema)]
pub struct CompleteRequest {
    /// Text to continue
    prompt: String,
    #[serde(default, flatten)]
    #[schema(inline)]
    config: RunConfig,
}

#[derive(Serialize, Deserialize, Debug, utoipa::ToSchema)]
pub struct CompleteResponse {
    /// The original prompt
    prompt: String,
    /// The generated continuation
    output: String,
}

/// Stream language-model tokens as Server-Sent Events.
///
/// Each token is sent as `data: <token>`.  The final event is
/// `event: done` with an empty data field.  On error the event name
/// is `error` and the data contains the error message.
#[instrument(skip(model_context))]
pub async fn stream_complete(
    State(model_context): State<Arc<Mutex<Context>>>,
    Json(payload): Json<CompleteRequest>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let mut lock = model_context.lock().await;
        let token_stream = lock.model.run(payload.prompt, payload.config);
        pin_mut!(token_stream);

        while let Some(result) = token_stream.next().await {
            let event = match result {
                Ok(token) => Event::default().data(token),
                Err(e) => {
                    tracing::error!(%e, "token stream error");
                    let _ = tx
                        .send(Ok(Event::default().event("error").data(e.to_string())))
                        .await;
                    return;
                }
            };
            if tx.send(Ok(event)).await.is_err() {
                return; // client disconnected
            }
        }

        let _ = tx
            .send(Ok(Event::default().event("done").data("")))
            .await;
    });

    Sse::new(ReceiverStream::new(rx))
}

/// Run a language-model completion.
#[utoipa::path(
    post,
    path = "/complete",
    request_body = CompleteRequest,
    responses(
        (status = 200, description = "Completion result", body = CompleteResponse),
        (status = 400, description = "Invalid JSON request"),
        (status = 500, description = "Inference error"),
    ),
    tag = "lm",
)]
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
