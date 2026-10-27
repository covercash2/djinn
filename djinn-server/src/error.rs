use axum::{extract::rejection::JsonRejection, http::StatusCode, response::IntoResponse};
use serde::Serialize;

use crate::server::Json;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Json(#[from] JsonRejection),
    #[error(transparent)]
    Models(#[from] djinn_models::Error),
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        #[derive(Serialize)]
        struct ErrorResponse {
            message: String,
        }

        let (status, message) = match self {
            Error::Json(err) => (err.status(), err.body_text()),
            Error::Models(err) => {
                tracing::error!(%err, "djinn_models error");
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

    #[test]
    fn models_error_returns_500() {
        let err = Error::Models(djinn_models::Error::Anyhow(anyhow::anyhow!("test")));
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
