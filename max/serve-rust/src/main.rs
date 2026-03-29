mod types;
mod zmq_interface;
mod openai;
mod kserve;
mod config;
mod python_bridge;
mod error;

use axum::{
    routing::get,
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use crate::openai::{openai_routes, AppState};
use crate::kserve::kserve_routes;
use crate::zmq_interface::ZmqModelWorkerProxy;
use crate::config::Settings;
use crate::python_bridge::PythonBridge;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let settings = Settings::from_env();
    tracing::info!("Server settings: {:?}", settings);

    let request_addr = crate::types::generate_zmq_ipc_path();
    let response_addr = crate::types::generate_zmq_ipc_path();
    let cancel_addr = crate::types::generate_zmq_ipc_path();
    let proxy = Arc::new(ZmqModelWorkerProxy::<crate::openai::ChatCompletionRequest, Vec<i32>>::new(
        &request_addr,
        &response_addr,
        &cancel_addr,
    ).await);

    Arc::clone(&proxy).start_response_worker();

    let state = Arc::new(AppState {
        proxy: Arc::clone(&proxy),
        python_bridge: Arc::new(PythonBridge::new()),
    });

    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/", get(root))
        .route("/health", get(health))
        .route("/version", get(version))
        .nest("/v1", openai_routes())
        .nest("/v2", kserve_routes())
        .with_state(state);

    // run our app with hyper, listening globally on configured port
    let addr: SocketAddr = format!("{}:{}", settings.host, settings.port)
        .parse()
        .expect("Invalid address");
    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.expect("Failed to bind TCP listener");
    axum::serve(listener, app).await.expect("Failed to start axum server");
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "MAX Serve Rust API"
}

async fn health() -> &'static str {
    "OK"
}

async fn version() -> &'static str {
    "0.1.0"
}
