use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::Path,
};
use std::sync::Arc;
use crate::openai::AppState;

pub fn kserve_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/models/:model_name/versions/:model_version/infer", post(infer))
}

async fn live() -> &'static str {
    "Live"
}

async fn ready() -> &'static str {
    "Ready"
}

async fn infer(
    Path((model_name, model_version)): Path<(String, String)>,
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    // Placeholder for KServe inference
    Json(serde_json::json!({
        "model_name": model_name,
        "model_version": model_version,
        "outputs": []
    }))
}
