use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(title = "djinn-server", version = "0.1.0"),
    paths(
        crate::server::health_check_handler,
        crate::complete::complete,
        crate::clip::clip_similarity,
    ),
    components(schemas(
        crate::complete::CompleteRequest,
        crate::complete::CompleteResponse,
        crate::clip::ClipRequest,
        crate::clip::ClipResponse,
        djinn_core::lm::config::RunConfig,
    )),
    tags(
        (name = "health", description = "Server health"),
        (name = "lm", description = "Language model completion"),
        (name = "clip", description = "Image-text similarity via CLIP"),
    ),
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    use utoipa::OpenApi;
    use utoipa_swagger_ui::SwaggerUi;

    #[test]
    fn spec_contains_all_paths() {
        let spec = ApiDoc::openapi();
        let paths: Vec<&str> = spec.paths.paths.keys().map(String::as_str).collect();
        assert!(paths.contains(&"/health-check"), "missing /health-check");
        assert!(paths.contains(&"/complete"), "missing /complete");
        assert!(paths.contains(&"/clip"), "missing /clip");
    }

    #[test]
    fn spec_contains_all_schemas() {
        let spec = ApiDoc::openapi();
        let schemas = spec
            .components
            .as_ref()
            .expect("components should be present")
            .schemas
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        for name in ["CompleteRequest", "CompleteResponse", "ClipRequest", "ClipResponse", "RunConfig"] {
            assert!(schemas.contains(&name), "missing schema: {name}");
        }
    }

    #[tokio::test]
    async fn openapi_json_endpoint_returns_200() {
        let app = axum::Router::new().merge(
            SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", ApiDoc::openapi()),
        );

        let response = app
            .oneshot(
                Request::get("/api-doc/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("response is valid JSON");
        assert_eq!(json["info"]["title"], "djinn-server");
    }
}
