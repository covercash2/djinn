use std::sync::Arc;

use axum::extract::State;
use base64::{engine::general_purpose::STANDARD, Engine};
use djinn_core::tensor_ext::cosine_similarity;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::instrument;

use crate::error::{Error, Result};
use crate::server::{Context, Json};

pub const ROUTE_CLIP: &str = "/clip";

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub struct ClipRequest {
    /// Natural-language description to compare against the image
    prompt: String,
    /// Image file contents encoded as standard base64 (JPEG, PNG, WebP, …)
    #[schema(format = Byte)]
    image: String,
}

#[derive(Serialize, Debug, utoipa::ToSchema)]
pub struct ClipResponse {
    /// Cosine similarity in [-1.0, 1.0]; higher means more semantically similar
    similarity: f32,
}

/// Compute cosine similarity between a text prompt and an image using CLIP (ViT-B/32).
#[utoipa::path(
    post,
    path = "/clip",
    request_body = ClipRequest,
    responses(
        (status = 200, description = "Similarity score", body = ClipResponse),
        (status = 400, description = "image field is not valid base64"),
        (status = 415, description = "Request body is not application/json"),
        (status = 500, description = "Model inference or image decoding failed"),
    ),
    tag = "clip",
)]
#[instrument(skip(context))]
pub async fn clip_similarity(
    State(context): State<Arc<Mutex<Context>>>,
    Json(payload): Json<ClipRequest>,
) -> Result<Json<ClipResponse>> {
    let bytes = STANDARD.decode(&payload.image).map_err(Error::Base64)?;

    let lock = context.lock().await;
    let text_features = lock.clip.encode_text(&payload.prompt)?;
    let image_features = lock.clip.encode_image_from_bytes(&bytes)?;
    drop(lock);

    let similarity = cosine_similarity(&text_features, &image_features)
        .map_err(djinn_core::image::clip::ClipError::Candle)?;
    Ok(Json(ClipResponse { similarity }))
}
