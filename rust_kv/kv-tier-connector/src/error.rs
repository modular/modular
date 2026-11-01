//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//! Error type for the tiered KV connector.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ConnectorError>;

/// Aborts the entire process after logging `msg`.
///
/// Used for unrecoverable transfer failures (CUDA copy/API, disk read/write)
/// that run on lane / disk-worker threads. A panic there would only unwind the
/// worker thread.
pub fn fatal(msg: impl fmt::Display) -> ! {
    // Write straight to stderr (not `eprintln!`, which the workspace lint
    // denies) so the cause is always visible even with no tracing subscriber
    // installed -- a silent `abort()` is useless to debug.
    use std::io::Write as _;
    drop(writeln!(
        std::io::stderr(),
        "kv-tier-connector FATAL: {msg}"
    ));
    tracing::error!("kv-tier-connector FATAL: {msg}");
    std::process::abort();
}

/// A connector error (bad geometry, failed CUDA copy/event, disk I/O). Most
/// call sites treat these as unrecoverable and [`fatal`] out; setup errors are
/// surfaced to the caller as a `Result`.
#[derive(Debug, Clone)]
pub enum ConnectorError {
    /// A configuration / setup problem (bad buffer geometry, no CUDA driver).
    Config(String),
    /// A transfer (CUDA copy, event, disk I/O) failed.
    TransferFailed(String),
}

impl fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectorError::Config(m) => write!(f, "connector config error: {m}"),
            ConnectorError::TransferFailed(m) => write!(f, "connector transfer failed: {m}"),
        }
    }
}

impl std::error::Error for ConnectorError {}
