use crate::error::{AppError, AppResult};
use crate::openai::AppState;
use axum::{
    extract::Path,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;

pub fn kserve_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route(
            "/models/:model_name/versions/:model_version/infer",
            post(infer),
        )
}

async fn live() -> AppResult<&'static str> {
    Ok("Live")
}

async fn ready() -> AppResult<&'static str> {
    Ok("Ready")
}

async fn infer(
    Path((model_name, model_version)): Path<(String, String)>,
    Json(_payload): Json<serde_json::Value>,
) -> AppResult<Json<serde_json::Value>> {
    if model_name.trim().is_empty() {
        return Err(AppError::Internal(
            "model_name must not be empty".to_string(),
        ));
    }
    // Placeholder for KServe inference
    Ok(Json(serde_json::json!({
        "model_name": model_name,
        "model_version": model_version,
        "outputs": []
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn live_and_ready_endpoints_return_expected_strings() {
        assert_eq!(live().await.expect("live"), "Live");
        assert_eq!(ready().await.expect("ready"), "Ready");
    }

    #[tokio::test]
    async fn infer_echoes_model_identity() {
        let Json(body) = infer(
            Path(("demo-model".to_string(), "v1".to_string())),
            Json(serde_json::json!({"inputs": []})),
        )
        .await
        .expect("infer");
        assert_eq!(body["model_name"], "demo-model");
        assert_eq!(body["model_version"], "v1");
        assert!(body["outputs"].is_array());
    }
}
