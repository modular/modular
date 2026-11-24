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
//! ABI handshake against a real bazel-built shim.
//!
//! Skipped unless `MAX_PROFILER_TEST_SHIM` points at a built
//! `libmax_profiler_shim.{so,dylib}`:
//!
//! ```text
//! ./bazelw build //max/internal/profiler:max_profiler_shim
//! MAX_PROFILER_TEST_SHIM=$(pwd)/../bazel-bin/max/internal/profiler/libmax_profiler_shim.so \
//!     cargo test -p max-profiler --test with_shim
//! ```
//!
//! Exercises resolution + the gating contract; span *emission* needs a
//! vendor library and is validated with nsys/rocprof in the consumer PRs.

use std::path::PathBuf;

#[test]
fn resolves_and_gates() {
    let Ok(shim_path) = std::env::var("MAX_PROFILER_TEST_SHIM") else {
        eprintln!("MAX_PROFILER_TEST_SHIM not set; skipping");
        return;
    };
    assert!(
        max_profiler::init_with_path(&PathBuf::from(&shim_path)),
        "failed to resolve shim at {shim_path}"
    );
    assert_eq!(max_profiler::backend(), max_profiler::Backend::Resolved);

    // Suppressed: is_enabled is definitively false.
    max_profiler::set_enabled(false);
    assert!(!max_profiler::is_enabled());

    // Requested: emission additionally needs a vendor library, so this is
    // machine-dependent; the calls must be safe either way.
    max_profiler::set_enabled(true);
    {
        max_profiler::profile_range!(c"with_shim::range", max_profiler::color::EXECUTOR_EXECUTE);
        max_profiler::mark(c"with_shim::mark", 0);
    }
    let started = max_profiler::StartedRange::start(c"with_shim::started", 0);
    started.end();

    max_profiler::set_enabled(false);
    assert!(!max_profiler::is_enabled());
}
