use crate::metrics::{RustMetrics, RustMetricsSnapshot};
use crate::python_bridge::PythonBridge;
use crate::types::RequestID;
use crate::zmq_interface::{StreamInitError, ZmqModelWorkerProxy};
use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest<'a> {
    #[serde(borrow)]
    pub model: Cow<'a, str>,
    #[serde(borrow)]
    pub messages: Vec<Message<'a>>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message<'a> {
    #[serde(borrow)]
    pub role: Cow<'a, str>,
    #[serde(borrow)]
    pub content: Cow<'a, str>,
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
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

pub struct AppState {
    pub proxy: Arc<ZmqModelWorkerProxy<Vec<i32>>>, // Returning token IDs
    pub python_bridge: Arc<PythonBridge>,
    pub metrics: Arc<RustMetrics>,
}

pub fn openai_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/chat/completions", post(create_chat_completion))
        .route("/models", get(list_models))
        .route("/rust/metrics", get(rust_metrics))
}

use axum::response::IntoResponse;
use futures::StreamExt;

use axum::response::sse::{Event, Sse};
use std::convert::Infallible;

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> axum::response::Response {
    let parse_started = Instant::now();
    let payload: ChatCompletionRequest<'_> = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: format!("Invalid JSON payload: {}", err),
                }),
            )
                .into_response();
        }
    };
    state.metrics.record_ingress_parse(parse_started.elapsed());

    let request_id = RequestID::generate();
    let response_id = request_id.0.clone();
    let is_stream = payload.stream.unwrap_or(false);
    let mut stream = match state.proxy.stream(request_id, payload).await {
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
                let chunk_capacity = chunk.iter().map(Vec::len).sum();
                let mut chunk_tokens = Vec::with_capacity(chunk_capacity);
                for mut tokens in chunk {
                    chunk_tokens.append(&mut tokens);
                }
                if !chunk_tokens.is_empty() {
                    let token_count = chunk_tokens.len();
                    let started = Instant::now();
                    let text = state
                        .python_bridge
                        .decode_tokens(&chunk_tokens)
                        .unwrap_or_default();
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
            let chunk_capacity: usize = chunk.iter().map(Vec::len).sum();
            full_tokens.reserve(chunk_capacity);
            for mut output in chunk {
                full_tokens.append(&mut output);
            }
        }

        let token_count = full_tokens.len();
        let started = Instant::now();
        let full_content = state
            .python_bridge
            .decode_tokens(&full_tokens)
            .unwrap_or_else(|_| "Error decoding tokens".to_string());
        state.metrics.record_decode(token_count, started.elapsed());

        Json(ChatCompletionResponse {
            id: response_id.to_string(),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: "max-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: full_content,
                },
                finish_reason: Some("stop".to_string()),
            }],
        })
        .into_response()
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

async fn rust_metrics(State(state): State<Arc<AppState>>) -> Json<RustMetricsSnapshot> {
    Json(state.proxy.metrics_snapshot())
}
