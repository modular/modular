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
//! Behavior without any assumption about the shim being present.
//!
//! This binary links nothing, so on most dev machines and CI workers the
//! shim is unresolved and everything must be a silent no-op; when a
//! developer happens to have the shim on the loader path the calls must
//! still be safe. Assertions therefore key off `backend()` instead of
//! hardcoding one environment.

#[test]
fn calls_are_safe_regardless_of_backend() {
    let backend = max_profiler::backend();

    max_profiler::set_enabled(true);
    {
        max_profiler::profile_range!(c"no_op::outer", max_profiler::color::ENGINE_STEP);
        let _inner = max_profiler::Range::new(c"no_op::inner", 0);
        let _dynamic = max_profiler::Range::with_dynamic_name(|| format!("batch id={}", 42), 0);
        max_profiler::mark(c"no_op::mark", max_profiler::color::MODULAR_PURPLE);
    }
    let started = max_profiler::StartedRange::start(c"no_op::started", 0);
    started.end();
    drop(max_profiler::StartedRange::start(c"no_op::dropped", 0));

    if backend == max_profiler::Backend::Unresolved {
        // No shim: nothing can flip emission on.
        assert!(!max_profiler::is_enabled());
    }
    max_profiler::set_enabled(false);

    // Interior NUL bytes in dynamic names must not panic.
    let _r = max_profiler::Range::with_dynamic_name(|| "nul\0inside".to_string(), 0);
}
