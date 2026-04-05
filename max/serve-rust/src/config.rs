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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_env() {
        for key in [
            "MAX_SERVE_HOST",
            "MAX_SERVE_PORT",
            "MAX_SERVE_METRICS_ENDPOINT_PORT",
            "MAX_SERVE_API_TYPES",
            "MAX_SERVE_DISABLE_TELEMETRY",
            "MAX_SERVE_OFFLINE_INFERENCE",
            "MAX_SERVE_HEADLESS",
            "MAX_SERVE_LOGS_CONSOLE_LEVEL",
            "MAX_SERVE_RUST_REQUEST_QUEUE_CAPACITY",
            "MAX_SERVE_RUST_CANCEL_QUEUE_CAPACITY",
            "MAX_SERVE_RUST_REQUEST_BATCH_MAX_SIZE",
            "MAX_SERVE_RUST_REQUEST_BATCH_WAIT_US",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn from_env_uses_defaults_when_unset() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_env();

        let settings = Settings::from_env();
        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 8000);
        assert_eq!(settings.metrics_port, 8001);
        assert_eq!(settings.rust_request_queue_capacity, 4096);
        assert_eq!(settings.rust_cancel_queue_capacity, 1024);
        assert_eq!(settings.rust_request_batch_max_size, 32);
        assert_eq!(settings.rust_request_batch_wait_us, 200);
        assert_eq!(settings.api_types.len(), 2);
    }

    #[test]
    fn from_env_parses_custom_values() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var("MAX_SERVE_HOST", "127.0.0.1");
        std::env::set_var("MAX_SERVE_PORT", "9000");
        std::env::set_var("MAX_SERVE_METRICS_ENDPOINT_PORT", "9001");
        std::env::set_var("MAX_SERVE_API_TYPES", "openai,kserve,responses");
        std::env::set_var("MAX_SERVE_DISABLE_TELEMETRY", "1");
        std::env::set_var("MAX_SERVE_OFFLINE_INFERENCE", "1");
        std::env::set_var("MAX_SERVE_HEADLESS", "1");
        std::env::set_var("MAX_SERVE_LOGS_CONSOLE_LEVEL", "DEBUG");
        std::env::set_var("MAX_SERVE_RUST_REQUEST_QUEUE_CAPACITY", "128");
        std::env::set_var("MAX_SERVE_RUST_CANCEL_QUEUE_CAPACITY", "64");
        std::env::set_var("MAX_SERVE_RUST_REQUEST_BATCH_MAX_SIZE", "8");
        std::env::set_var("MAX_SERVE_RUST_REQUEST_BATCH_WAIT_US", "500");

        let settings = Settings::from_env();
        assert_eq!(settings.host, "127.0.0.1");
        assert_eq!(settings.port, 9000);
        assert_eq!(settings.metrics_port, 9001);
        assert!(settings.disable_telemetry);
        assert!(settings.offline_inference);
        assert!(settings.headless);
        assert_eq!(settings.logs_console_level, "DEBUG");
        assert_eq!(settings.rust_request_queue_capacity, 128);
        assert_eq!(settings.rust_cancel_queue_capacity, 64);
        assert_eq!(settings.rust_request_batch_max_size, 8);
        assert_eq!(settings.rust_request_batch_wait_us, 500);
        assert_eq!(settings.api_types.len(), 3);
    }

    #[test]
    fn from_env_falls_back_when_port_is_invalid() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var("MAX_SERVE_PORT", "not-a-number");

        let settings = Settings::from_env();
        assert_eq!(settings.port, 8000);
    }
}
