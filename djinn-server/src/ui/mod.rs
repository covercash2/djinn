use std::ops::DerefMut;
use std::sync::Arc;

use axum::extract::Multipart;
use axum::extract::State;
use axum::response::Html;
use axum::Form;
use futures::{pin_mut, StreamExt};
use serde::Deserialize;
use tokio::sync::Mutex;
use tracing::{instrument, Instrument};

use crate::error::{Error, Result};
use crate::server::Context;

pub const ROUTE_UI_COMPLETE: &str = "/ui/complete";
pub const ROUTE_UI_CLIP: &str = "/ui/clip";

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

/// Convert a cosine similarity in `[-1.0, 1.0]` to a percentage in `[0.0, 100.0]`
/// for use as a progress-bar width.
fn similarity_to_pct(similarity: f32) -> f32 {
    ((similarity + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0)
}

/// Accept a multipart upload (prompt + image file) and return an HTML fragment
/// with the CLIP cosine-similarity score for HTMX.
#[instrument(skip(context))]
pub async fn ui_clip(
    State(context): State<Arc<Mutex<Context>>>,
    mut multipart: Multipart,
) -> Result<Html<String>> {
    let mut prompt: Option<String> = None;
    let mut image_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| Error::Multipart(e.to_string()))?
    {
        match field.name() {
            Some("prompt") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| Error::Multipart(e.to_string()))?;
                prompt = Some(text);
            }
            Some("image") => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| Error::Multipart(e.to_string()))?;
                image_bytes = Some(bytes.to_vec());
            }
            _ => {}
        }
    }

    let prompt = match prompt {
        Some(p) if !p.trim().is_empty() => p.trim().to_string(),
        _ => {
            return Ok(Html(
                r#"<p class="error">Prompt must not be empty.</p>"#.to_string(),
            ))
        }
    };

    let image_bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            return Ok(Html(
                r#"<p class="error">Image must not be empty.</p>"#.to_string(),
            ))
        }
    };

    let lock = context.lock().await;
    let text_features = lock.clip.encode_text(&prompt)?;
    let image_features = lock.clip.encode_image_from_bytes(&image_bytes)?;
    drop(lock);

    let similarity =
        djinn_core::tensor_ext::cosine_similarity(&text_features, &image_features)
            .map_err(djinn_core::image::VisionEncoderError::Candle)?;

    let prompt_escaped = html_escape(&prompt);
    Ok(Html(format!(
        r#"<section class="response-block">
  <h3 class="response-prompt">Prompt</h3>
  <pre class="response-prompt-text">{prompt_escaped}</pre>
  <h3 class="response-output">CLIP Similarity</h3>
  <div class="response-content clip-similarity">
    <p>Cosine similarity: <strong>{similarity:.4}</strong></p>
    <div class="similarity-bar-container"
         role="progressbar"
         aria-valuenow="{bar_pct:.1}"
         aria-valuemin="0"
         aria-valuemax="100"
         aria-label="Similarity: {similarity:.4}">
      <div class="similarity-bar" style="width:{bar_pct:.1}%"></div>
    </div>
  </div>
</section>"#,
        bar_pct = similarity_to_pct(similarity),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_replaces_all_special_chars() {
        assert_eq!(html_escape("&"), "&amp;");
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(html_escape("it's"), "it&#x27;s");
        assert_eq!(
            html_escape("<b>a & b</b>"),
            "&lt;b&gt;a &amp; b&lt;/b&gt;"
        );
    }

    #[test]
    fn html_escape_is_idempotent_on_plain_text() {
        let plain = "hello world 123";
        assert_eq!(html_escape(plain), plain);
    }

    /// Verifies that a similarity in [-1, 1] maps to a bar width in [0, 100].
    #[test]
    fn similarity_bar_percent_clamps_correctly() {
        assert!((similarity_to_pct(-1.0) - 0.0).abs() < 1e-4);
        assert!((similarity_to_pct(0.0) - 50.0).abs() < 1e-4);
        assert!((similarity_to_pct(1.0) - 100.0).abs() < 1e-4);
        // Values outside [-1,1] are clamped.
        assert_eq!(similarity_to_pct(-2.0), 0.0);
        assert_eq!(similarity_to_pct(2.0), 100.0);
    }
}
