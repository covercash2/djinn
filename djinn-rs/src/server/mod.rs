use axum::{
    routing::{get, IntoMakeService},
    Router,
};
use derive_new::new;
use std::{fmt::Display, net::SocketAddr, sync::Arc};

#[derive(new)]
pub struct Context {
    config: Config,
}

#[derive(new)]
pub struct Config {
    socker_addr: SocketAddr,
}

#[derive(new)]
pub struct HttpServer {
    context: Arc<Context>,
}

async fn health_check_handler() -> &'static str {
    "OK"
}

fn build_service() -> IntoMakeService<Router> {
    let router = Router::new().route(
        &ServiceRoutes::HealthCheck.to_string(),
        get(health_check_handler),
    );

    router.into_make_service()
}

enum ServiceRoutes {
    HealthCheck,
}

impl Display for ServiceRoutes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRoutes::HealthCheck => write!(f, "/health-check"),
        }
    }
}

impl HttpServer {
    pub async fn start(self) -> anyhow::Result<()> {
        let context = self.context;
        let socket_addr = context.config.socker_addr;

        let listener = tokio::net::TcpListener::bind(&socket_addr).await?;

        axum::serve(listener, build_service()).await?;

        tracing::info!("HTTP server shutdown");

        Ok(())
    }
}
