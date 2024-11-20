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
        };

        (status, Json(ErrorResponse { message })).into_response()
    }
}
