use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Serialize, Deserialize)]
pub enum APIType {
    #[serde(rename = "kserve")]
    KSERVE,
    #[serde(rename = "openai")]
    OPENAI,
    #[serde(rename = "sagemaker")]
    SAGEMAKER,
    #[serde(rename = "responses")]
    OPENRESPONSES,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub metrics_port: u16,
    pub api_types: Vec<APIType>,
    pub disable_telemetry: bool,
    pub offline_inference: bool,
    pub headless: bool,
    pub logs_console_level: String,
    pub rust_request_queue_capacity: usize,
    pub rust_cancel_queue_capacity: usize,
    pub rust_request_batch_max_size: usize,
    pub rust_request_batch_wait_us: u64,
}

impl Settings {
    pub fn from_env() -> Self {
        Self {
            host: env::var("MAX_SERVE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("MAX_SERVE_PORT").map_or(8000, |val| {
                val.parse().unwrap_or_else(|e| {
                    tracing::warn!(
                        "Could not parse MAX_SERVE_PORT: '{}'. Error: {}. Using default 8000.",
                        val,
                        e
                    );
                    8000
                })
            }),
            metrics_port: env::var("MAX_SERVE_METRICS_ENDPOINT_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8001),
            api_types: env::var("MAX_SERVE_API_TYPES")
                .ok()
                .map(|s| {
                    s.split(',')
                        .filter_map(|t| serde_json::from_str(&format!("\"{}\"", t.trim())).ok())
                        .collect()
                })
                .unwrap_or_else(|| vec![APIType::OPENAI, APIType::SAGEMAKER]),
            disable_telemetry: env::var("MAX_SERVE_DISABLE_TELEMETRY").is_ok(),
            offline_inference: env::var("MAX_SERVE_OFFLINE_INFERENCE").is_ok(),
            headless: env::var("MAX_SERVE_HEADLESS").is_ok(),
            logs_console_level: env::var("MAX_SERVE_LOGS_CONSOLE_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string()),
            rust_request_queue_capacity: env::var("MAX_SERVE_RUST_REQUEST_QUEUE_CAPACITY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
            rust_cancel_queue_capacity: env::var("MAX_SERVE_RUST_CANCEL_QUEUE_CAPACITY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024),
            rust_request_batch_max_size: env::var("MAX_SERVE_RUST_REQUEST_BATCH_MAX_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32),
            rust_request_batch_wait_us: env::var("MAX_SERVE_RUST_REQUEST_BATCH_WAIT_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200),
        }
    }
}
