use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::types::RequestID;
use crate::zmq_interface::ZmqModelWorkerProxy;
use crate::python_bridge::PythonBridge;

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
}

pub fn openai_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/chat/completions", post(create_chat_completion))
        .route("/models", get(list_models))
}

use futures::StreamExt;
use axum::response::IntoResponse;

use axum::response::sse::{Event, Sse};
use std::convert::Infallible;

async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let request_id = RequestID::generate();
    let is_stream = payload.stream.unwrap_or(false);
    let mut stream = state.proxy.stream(request_id.clone(), payload).await;

    if is_stream {
        let sse_stream = async_stream::stream! {
            while let Some(chunk) = stream.next().await {
                for tokens in chunk {
                    let text = state.python_bridge.decode_tokens(tokens).unwrap_or_default();
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

        let full_content = state.python_bridge.decode_tokens(full_tokens).unwrap_or_else(|_| "Error decoding tokens".to_string());

        Json(ChatCompletionResponse {
            id: request_id.0,
            object: "chat.completion".to_string(),
            created: 1677652288,
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
