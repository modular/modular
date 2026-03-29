use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("ZeroMQ error: {0}")]
    ZmqError(#[from] zeromq::ZmqError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] rmp_serde::encode::Error),

    #[error("Deserialization error: {0}")]
    DeserializationError(#[from] rmp_serde::decode::Error),

    #[error("Python error: {0}")]
    PythonError(#[from] pyo3::PyErr),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not found")]
    NotFound,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::ZmqError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::SerializationError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::DeserializationError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::PythonError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::NotFound => (StatusCode::NOT_FOUND, "Resource not found".to_string()),
            AppError::Internal(s) => (StatusCode::INTERNAL_SERVER_ERROR, s),
        };

        let body = Json(json!({
            "error": error_message,
        }));

        (status, body).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
