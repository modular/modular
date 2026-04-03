use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone, Hash, PartialEq, Eq)]
pub struct RequestID(pub Arc<str>);

impl RequestID {
    pub fn generate() -> Self {
        RequestID(Arc::<str>::from(Uuid::new_v4().simple().to_string()))
    }
}

pub fn generate_zmq_ipc_path() -> String {
    let temp_dir = env::temp_dir();
    let uuid = Uuid::new_v4().simple().to_string();
    let short_uuid = &uuid[..18];
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
