use serde::{Deserialize, Serialize};
use std::env;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone, Hash, PartialEq, Eq)]
pub struct RequestID(pub String);

impl RequestID {
    pub fn generate() -> Self {
        RequestID(Uuid::new_v4().to_string())
    }
}

pub fn generate_zmq_ipc_path() -> String {
    let temp_dir = env::temp_dir();
    let short_uuid = &Uuid::new_v4().to_string().replace("-", "")[..18];
    format!("ipc://{}/{}", temp_dir.display(), short_uuid)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchedulerResult<T> {
    pub result: Option<T>,
    pub is_done: bool,
}

// These are placeholders for the actual data structures used in MAX.
// In a real implementation, we would use pyo3 or shared message definitions.

#[derive(Debug, Serialize, Deserialize)]
pub struct TextGenerationContext<T> {
    pub request_id: RequestID,
    pub request: T,
}
