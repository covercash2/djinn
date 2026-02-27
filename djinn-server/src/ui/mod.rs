use std::ops::DerefMut;
use std::sync::Arc;

use axum::extract::State;
use axum::response::Html;
use axum::Form;
use futures::{pin_mut, StreamExt};
use serde::Deserialize;
use tokio::sync::Mutex;
use tracing::{instrument, Instrument};

use crate::error::Result;
use crate::server::Context;

pub const ROUTE_UI_COMPLETE: &str = "/ui/complete";

#[derive(Deserialize, Debug)]
pub struct UiCompleteRequest {
    prompt: String,
}

/// Accept a form-encoded completion request and return an HTML fragment for HTMX.
#[instrument(skip(context))]
pub async fn ui_complete(
    State(context): State<Arc<Mutex<Context>>>,
    Form(payload): Form<UiCompleteRequest>,
) -> Result<Html<String>> {
    let prompt = payload.prompt.trim().to_string();

    if prompt.is_empty() {
        return Ok(Html(
            r#"<p class="error">Prompt must not be empty.</p>"#.to_string(),
        ));
    }

    let config = djinn_core::lm::config::RunConfig::default();

    let span = tracing::info_span!("ui_complete");
    let mut lock = context.lock().instrument(span).await;
    tracing::info!("got model lock");
    let ctx: &mut Context = lock.deref_mut();

    let stream = ctx.model.run(prompt.clone(), config);
    pin_mut!(stream);

    let mut output = String::new();
    while let Some(value) = stream.next().await {
        match value {
            Ok(token) => output.push_str(&token),
            Err(e) => {
                // Return an HTML fragment so HTMX can render the error correctly.
                let message = html_escape(&e.to_string());
                return Ok(Html(format!(
                    r#"<p class="error">Error while generating response: {message}</p>"#
                )));
            }
        }
    }

    let rendered = markdown::to_html(&output);
    // `markdown::to_html` uses comrak with `allow_dangerous_html = false` by
    // default, so inline HTML from the model output is already sanitized.

    tracing::info!("sending ui response for prompt of {} chars", prompt.len());

    Ok(Html(format!(
        r#"<section class="response-block">
  <h3 class="response-prompt">Prompt</h3>
  <pre class="response-prompt-text">{prompt}</pre>
  <h3 class="response-output">Response</h3>
  <div class="response-content">{rendered}</div>
</section>"#,
        prompt = html_escape(&prompt),
        rendered = rendered,
    )))
}

/// Minimal HTML-escaping for user-supplied text placed inside element content.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
