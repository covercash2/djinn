use axum::{extract::rejection::JsonRejection, http::StatusCode, response::IntoResponse};
use serde::Serialize;

use crate::server::Json;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Json(#[from] JsonRejection),
    #[error(transparent)]
    Core(#[from] djinn_core::Error),
    #[error("invalid base64 image: {0}")]
    Base64(base64::DecodeError),
    #[error(transparent)]
    VisionEncoder(#[from] djinn_core::image::VisionEncoderError),
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        #[derive(Serialize)]
        struct ErrorResponse {
            message: String,
        }

        let (status, message) = match self {
            Error::Json(err) => (err.status(), err.body_text()),
            Error::Core(err) => {
                tracing::error!(%err, "djinn_core error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Something went wrong D:".to_string(),
                )
            }
            Error::Base64(err) => (StatusCode::BAD_REQUEST, format!("invalid base64: {err}")),
            Error::VisionEncoder(err) => {
                tracing::error!(%err, "vision encoder error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Something went wrong D:".to_string(),
                )
            }
        };

        (status, Json(ErrorResponse { message })).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use djinn_core::image::VisionEncoderError;

    #[test]
    fn core_error_returns_500() {
        let err = Error::Core(djinn_core::Error::Anyhow(anyhow::anyhow!("test")));
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn vision_encoder_error_returns_500() {
        let err = Error::VisionEncoder(VisionEncoderError::MissingPadToken);
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn base64_error_returns_400() {
        let decode_err =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, "not-base64!!!")
                .unwrap_err();
        let err = Error::Base64(decode_err);
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
