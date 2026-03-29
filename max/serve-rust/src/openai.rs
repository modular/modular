use axum::{
    http::StatusCode,
    routing::{get, post},
    Router,
    Json,
    extract::State,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::types::RequestID;
use crate::zmq_interface::{ZmqModelWorkerProxy, StreamInitError};
use crate::python_bridge::PythonBridge;
use crate::metrics::{RustMetrics, RustMetricsSnapshot};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: Option<String>,
}

pub struct AppState {
    pub proxy: Arc<ZmqModelWorkerProxy<ChatCompletionRequest, Vec<i32>>>, // Returning token IDs
    pub python_bridge: Arc<PythonBridge>,
    pub metrics: Arc<RustMetrics>,
}

pub fn openai_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/chat/completions", post(create_chat_completion))
        .route("/models", get(list_models))
        .route("/rust/metrics", get(rust_metrics))
}

use futures::StreamExt;
use axum::response::IntoResponse;

use axum::response::sse::{Event, Sse};
use std::convert::Infallible;

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let request_id = RequestID::generate();
    let is_stream = payload.stream.unwrap_or(false);
    let mut stream = match state.proxy.stream(request_id.clone(), payload).await {
        Ok(stream) => stream,
        Err(StreamInitError::Overloaded) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorBody {
                    error: "Server is overloaded. Try again shortly.".to_string(),
                }),
            )
                .into_response();
        }
        Err(StreamInitError::Unavailable) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody {
                    error: "Inference queue unavailable.".to_string(),
                }),
            )
                .into_response();
        }
    };

    if is_stream {
        let sse_stream = async_stream::stream! {
            while let Some(chunk) = stream.next().await {
                // Decode once per scheduler chunk to reduce Python GIL crossings.
                let mut chunk_tokens = Vec::new();
                for tokens in chunk {
                    chunk_tokens.extend(tokens);
                }
                if !chunk_tokens.is_empty() {
                    let token_count = chunk_tokens.len();
                    let started = Instant::now();
                    let text = state.python_bridge.decode_tokens(chunk_tokens).unwrap_or_default();
                    state
                        .metrics
                        .record_decode(token_count, started.elapsed());
                    yield Ok::<Event, Infallible>(Event::default().data(text));
                }
            }
        };
        Sse::new(sse_stream).into_response()
    } else {
        let mut full_tokens = Vec::new();
        while let Some(chunk) = stream.next().await {
            for output in chunk {
                full_tokens.extend(output);
            }
        }

        let token_count = full_tokens.len();
        let started = Instant::now();
        let full_content = state.python_bridge.decode_tokens(full_tokens).unwrap_or_else(|_| "Error decoding tokens".to_string());
        state.metrics.record_decode(token_count, started.elapsed());

        Json(ChatCompletionResponse {
            id: request_id.0,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
            model: "max-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: full_content,
                },
                finish_reason: Some("stop".to_string()),
            }],
        }).into_response()
    }
}

async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "max-model",
                "object": "model",
                "created": 1677652288,
                "owned_by": "modular"
            }
        ]
    }))
}

async fn rust_metrics(
    State(state): State<Arc<AppState>>,
) -> Json<RustMetricsSnapshot> {
    Json(state.proxy.metrics_snapshot())
}
