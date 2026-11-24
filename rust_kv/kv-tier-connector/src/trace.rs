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
//! Profiler ranges for the tier connector, via the MAX tool-annotation shim.
//!
//! Thin adapter over the `max-profiler` crate, which resolves
//! `libmax_profiler_shim.so` at runtime and emits NVTX on NVIDIA (nsys) or
//! rocTX on AMD (rocprof) — replacing this crate's direct `dlopen` of the
//! vendor libraries. Semantics are unchanged: ranges are opt-in via the
//! same `MODULAR_ENABLE_PROFILING` env var MAX's Python tracer checks, and
//! every call is a no-op when profiling is off or the shim / vendor library
//! is absent. NVTX carries an ARGB color; rocTX takes only a message
//! string, so the color is dropped there.
//!
//! Inside a Python host the shim is not on the loader search path; the
//! pyo3 module init points the crate at the MAX wheel's copy (see
//! `kv-tier-connector-pyo3`).

/// RAII profiler range: pushes on construction, pops on drop. No-op when
/// profiling is disabled or the shim is unresolved.
pub struct Range {
    _inner: max_profiler::Range,
}

impl Range {
    /// Opens a range named `name` with ARGB `color` (color is used by NVTX
    /// and ignored by rocTX). `name` may be dynamic (e.g. `"kv-disk
    /// read_block"`); it is only formatted into a C string when the range
    /// will actually emit.
    #[must_use]
    pub fn new(name: &str, color: u32) -> Self {
        Self {
            _inner: max_profiler::Range::with_dynamic_name(|| name.to_owned(), color),
        }
    }
}

/// ARGB colors roughly matching the Python tiers (informational only; used
/// by NVTX, ignored by rocTX).
pub mod color {
    pub const ONLOAD: u32 = 0xFF00_AA44;
    pub const OFFLOAD: u32 = 0xFFFF_8800;
    pub const DISK_READ: u32 = 0xFF66_CCFF;
    pub const DISK_WRITE: u32 = 0xFF88_44CC;
}

#[cfg(test)]
mod tests {
    use super::{Range, color};

    // No-crash smoke over the adapter boundary; runs without the shim (every
    // call degrades to a no-op) so it is deterministic anywhere. Emission
    // itself is covered by max-profiler's tests and the nsys/rocprof recipe.
    #[test]
    fn ranges_are_safe_without_shim() {
        let outer = Range::new("kv-onload h2d", color::ONLOAD);
        let inner = Range::new(&"kv-disk read_block ".repeat(64), color::DISK_READ);
        drop(inner);
        drop(outer);
        let _nul = Range::new("nul\0inside", color::OFFLOAD);
    }
}
