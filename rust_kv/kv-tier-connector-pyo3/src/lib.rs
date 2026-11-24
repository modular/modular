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
//! Python bindings for the `kv-tier-connector` Rust crate.
//!
//! Compiles to a Python extension module (`kv_tier_connector._kv_tier_connector_pyo3`,
//! re-exported by `python/kv_tier_connector/__init__.py`) so MAX's `KVConnector`
//! shim can drive the host/disk tiered connector without the transfer lanes
//! holding the Python GIL. Every decision lives in `kv-tier-connector`; this
//! crate is glue.

mod connector;

use pyo3::prelude::*;

/// Points max-profiler at the MAX wheel's tool-annotation shim
/// (`site-packages/modular/lib/libmax_profiler_shim.so`) so the connector's
/// trace ranges reach nsys/rocprof — inside a Python host the shim is not on
/// the loader search path. Best effort: without the wheel, max-profiler's
/// env/soname fallbacks still apply, and spans silently no-op.
fn init_profiler_shim(py: Python<'_>) {
    let shim = py
        .eval(
            c"__import__('importlib.util').util.find_spec('modular').submodule_search_locations[0]",
            None,
            None,
        )
        .ok()
        .and_then(|loc| loc.extract::<String>().ok())
        .map(|dir| {
            std::path::PathBuf::from(dir)
                .join("lib")
                .join("libmax_profiler_shim.so")
        });
    if let Some(path) = shim
        && path.exists()
    {
        max_profiler::init_with_path(&path);
    }
}

/// CPython looks up `PyInit__kv_tier_connector_pyo3` in the cdylib at import.
#[pymodule]
fn _kv_tier_connector_pyo3(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_profiler_shim(py);
    m.add_class::<connector::TierConnector>()?;
    m.add_class::<connector::TierTransfer>()?;
    Ok(())
}
