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
//! Safe bindings to the MAX tool-annotation shim (`libmax_profiler_shim.so`).
//!
//! Emits profiler spans that out-of-process tools see — NVTX for
//! nsight-systems on NVIDIA, roctx for rocprof on AMD — through a MAX-owned
//! C ABI instead of dlopen'ing the vendor libraries directly. The shim is
//! resolved at runtime with `dlsym`/`dlopen` (this crate links nothing), so
//! every call degrades to a no-op when the shim or a vendor library is
//! absent, and the crate builds anywhere without the CUDA/ROCm toolkits.
//!
//! Emission requires the shim to be resolvable AND profiling to be
//! requested: either the `MODULAR_ENABLE_PROFILING` environment variable is
//! truthy, or the process calls [`set_enabled`]`(true)`.
//!
//! Resolution is latched on first use. Processes where the shim is not on
//! the loader search path (e.g. a Python host) must call [`init_with_path`]
//! before the first span.
//!
//! Color scheme for engine components (see [`color`]):
//! - Engine     — blue shades
//! - Executor   — orange/yellow shades
//! - Scheduler  — green shades
//! - Sampler    — purple shades
//! - Spec       — teal shades
//! - KV cache   — magenta shades

mod ffi;

use std::{
    ffi::{CStr, CString},
    path::Path,
    sync::OnceLock,
};

/// Resolved function table for the `m_ext_profiler_*` surface.
#[derive(Clone, Copy)]
struct Shim {
    is_enabled: ffi::IsEnabledFn,
    set_enabled: ffi::SetEnabledFn,
    range_push: ffi::RangePushFn,
    range_pop: ffi::RangePopFn,
    range_start: ffi::RangeStartFn,
    range_end: ffi::RangeEndFn,
    mark: ffi::MarkFn,
}

static SHIM: OnceLock<Option<Shim>> = OnceLock::new();

/// Resolves one symbol from `handle`, or `None` if absent.
fn sym<T: Copy>(handle: *mut std::ffi::c_void, name: &CStr) -> Option<T> {
    // SAFETY: dlsym with a valid handle (or RTLD_DEFAULT) and a
    // NUL-terminated name; the transmute_copy narrows a data pointer to a
    // function pointer of the caller-declared type, which is only sound
    // because the shim's contract fixes each symbol's signature.
    unsafe {
        let ptr = ffi::dlsym(handle, name.as_ptr());
        if ptr.is_null() {
            return None;
        }
        Some(std::mem::transmute_copy::<*mut std::ffi::c_void, T>(&ptr))
    }
}

/// Resolves the full table from `handle`; all-or-nothing so a
/// partially-matching library is treated as absent.
fn table_from(handle: *mut std::ffi::c_void) -> Option<Shim> {
    Some(Shim {
        is_enabled: sym(handle, c"m_ext_profiler_is_enabled")?,
        set_enabled: sym(handle, c"m_ext_profiler_set_enabled")?,
        range_push: sym(handle, c"m_ext_profiler_range_push")?,
        range_pop: sym(handle, c"m_ext_profiler_range_pop")?,
        range_start: sym(handle, c"m_ext_profiler_range_start")?,
        range_end: sym(handle, c"m_ext_profiler_range_end")?,
        mark: sym(handle, c"m_ext_profiler_mark")?,
    })
}

fn table_from_library(path: &CStr) -> Option<Shim> {
    // SAFETY: dlopen with a NUL-terminated path. The handle is
    // intentionally leaked: the function pointers must stay valid for the
    // process lifetime.
    let handle = unsafe { ffi::dlopen(path.as_ptr(), ffi::RTLD_NOW) };
    if handle.is_null() {
        return None;
    }
    table_from(handle)
}

fn resolve(explicit_path: Option<&Path>) -> Option<Shim> {
    if let Some(path) = explicit_path
        && let Ok(cpath) = CString::new(path.as_os_str().as_encoded_bytes())
        && let Some(shim) = table_from_library(&cpath)
    {
        return Some(shim);
    }
    if let Ok(env_path) = std::env::var("MODULAR_PROFILER_SHIM")
        && let Ok(cpath) = CString::new(env_path)
        && let Some(shim) = table_from_library(&cpath)
    {
        return Some(shim);
    }
    // The symbols may already be in the process (a host that loaded the
    // shim, or a future libmax that absorbs it).
    if let Some(shim) = table_from(ffi::RTLD_DEFAULT) {
        return Some(shim);
    }
    table_from_library(ffi::SHIM_SONAME)
}

fn shim() -> Option<&'static Shim> {
    SHIM.get_or_init(|| resolve(None)).as_ref()
}

/// Whether the shim resolved; latched on first use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// No shim found: every call in this crate is a no-op.
    Unresolved,
    /// Shim symbols resolved (spans still require profiling to be enabled
    /// and a vendor annotation library on the machine).
    Resolved,
}

/// Reports whether the shim resolved, for consumer logging and tests.
/// Triggers resolution.
#[must_use]
pub fn backend() -> Backend {
    if shim().is_some() {
        Backend::Resolved
    } else {
        Backend::Unresolved
    }
}

/// Resolves the shim from an explicit library path (e.g.
/// `<site-packages>/modular/lib/libmax_profiler_shim.so` computed from
/// Python). Must run before the first span or [`backend`] query — the
/// outcome of the first resolution attempt is latched for the process
/// lifetime. Returns whether the shim is resolved, regardless of whether
/// this call performed the resolution (a caller that lost the race to an
/// earlier query gets that resolution's outcome, without `path`).
pub fn init_with_path(path: &Path) -> bool {
    SHIM.get_or_init(|| resolve(Some(path))).is_some()
}

/// Returns true when spans will actually emit: shim resolved, profiling
/// requested, and a vendor annotation library present.
#[must_use]
pub fn is_enabled() -> bool {
    // SAFETY: resolved shim function with the contract-fixed signature.
    shim().is_some_and(|s| unsafe { (s.is_enabled)() } != 0)
}

/// Programmatically requests (true) or suppresses (false) profiling,
/// overriding the `MODULAR_ENABLE_PROFILING` default. No-op when the shim
/// is unresolved.
pub fn set_enabled(enabled: bool) {
    if let Some(s) = shim() {
        // SAFETY: resolved shim function with the contract-fixed signature.
        unsafe { (s.set_enabled)(i32::from(enabled)) }
    }
}

/// Emits an instantaneous marker.
pub fn mark(name: &CStr, color_argb: u32) {
    if let Some(s) = shim() {
        // SAFETY: resolved shim function; `name` is NUL-terminated and the
        // vendor library copies it before returning.
        unsafe { (s.mark)(name.as_ptr(), color_argb) }
    }
}

/// RAII guard: pushes a colored, thread-local nested range on
/// construction, pops on destruction. Use via the
/// [`profile_range!`](crate::profile_range) macro.
pub struct Range {
    // Whether the shim reported an actual vendor push; the drop pops
    // exactly when it did, so a suppressed push (profiling disabled at
    // construction) can never pop an unrelated outer range.
    pushed: bool,
    // Ranges nest per-thread, so the guard must be dropped on the thread
    // that created it.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Range {
    #[must_use]
    pub fn new(name: &CStr, color_argb: u32) -> Self {
        let pushed = match shim() {
            // SAFETY: resolved shim function; `name` is NUL-terminated and
            // the vendor library copies it at push time.
            Some(s) => (unsafe { (s.range_push)(name.as_ptr(), color_argb) }) != 0,
            None => false,
        };
        Self {
            pushed,
            _not_send: std::marker::PhantomData,
        }
    }

    /// Pushes a range whose name is built at runtime — e.g. a
    /// per-iteration identity marker like `batch id=42 type=decode n=8`.
    /// The name closure runs only when spans will actually emit, so the
    /// disabled path pays no allocation or formatting.
    #[must_use]
    pub fn with_dynamic_name(name: impl FnOnce() -> String, color_argb: u32) -> Self {
        let mut pushed = false;
        if let Some(s) = shim()
            && is_enabled()
        {
            // Interior NUL bytes can't appear in a C string; drop them
            // rather than lose the whole marker.
            if let Ok(cname) = CString::new(name().replace('\0', "")) {
                // SAFETY: resolved shim function; `cname` is NUL-terminated
                // and the vendor library copies it at push time.
                pushed = unsafe { (s.range_push)(cname.as_ptr(), color_argb) } != 0;
            }
        }
        Self {
            pushed,
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if !self.pushed {
            return;
        }
        if let Some(s) = shim() {
            // SAFETY: paired with the vendor push the constructor reported;
            // the shim executes closers even if profiling was disabled
            // between push and drop, keeping the vendor's per-thread stack
            // balanced.
            unsafe { (s.range_pop)() }
        }
    }
}

/// A process-wide range that may end on a different thread than it
/// started on (safe across `await` points, unlike [`Range`]). Ends on
/// [`StartedRange::end`] or drop.
#[must_use]
pub struct StartedRange(u64);

impl StartedRange {
    pub fn start(name: &CStr, color_argb: u32) -> Self {
        let id = match shim() {
            // SAFETY: resolved shim function; `name` is NUL-terminated and
            // the vendor library copies it before returning.
            Some(s) => unsafe { (s.range_start)(name.as_ptr(), color_argb) },
            None => 0,
        };
        Self(id)
    }

    /// Ends the range now; equivalent to dropping, but reads better at
    /// call sites that end a range far from where it started.
    pub fn end(self) {}
}

impl Drop for StartedRange {
    fn drop(&mut self) {
        if self.0 == 0 {
            return;
        }
        if let Some(s) = shim() {
            // SAFETY: `self.0` came from the paired range_start.
            unsafe { (s.range_end)(self.0) }
        }
    }
}

/// Component color constants (ARGB). The engine palette matches the mach
/// engine's original NVTX colors verbatim so existing nsys workflows keep
/// their visual identity.
pub mod color {
    /// The `max.profiler` Python API's default `modular_purple` (RGB
    /// `0xB5BAF5` in the Mojo palette), with full alpha added.
    pub const MODULAR_PURPLE: u32 = 0xFFB5_BAF5;

    pub const ENGINE_STEP: u32 = 0xFF66_CCFF;
    pub const ENGINE_EXECUTOR_LOOP: u32 = 0xFF88_DDFF;

    pub const EXECUTOR_EXECUTE: u32 = 0xFFFF_8800;
    pub const EXECUTOR_FORWARD_PASS: u32 = 0xFFFF_AA33;
    pub const EXECUTOR_SAMPLE: u32 = 0xFFFF_CC66;

    pub const SCHEDULER_CONSTRUCT_BATCH: u32 = 0xFF00_AA44;
    pub const SCHEDULER_ADVANCE: u32 = 0xFF00_DD66;

    pub const SAMPLER_SAMPLE: u32 = 0xFF88_44CC;

    pub const SPEC_BUILD_INPUT: u32 = 0xFF00_CCAA;
    pub const SPEC_EXECUTE: u32 = 0xFF00_AA88;
    pub const SPEC_PARSE_OUTPUT: u32 = 0xFF33_DDBB;
    pub const SPEC_TEARDOWN: u32 = 0xFF00_8866;

    pub const KV_EXTEND: u32 = 0xFFCC_4488;
}

/// Opens a profiler range named `$name` (a `&CStr` literal) with `$color`.
/// The range pops when the surrounding scope ends.
///
/// ```ignore
/// profile_range!(c"Executor::execute", max_profiler::color::EXECUTOR_EXECUTE);
/// ```
#[macro_export]
macro_rules! profile_range {
    ($name:expr, $color:expr) => {
        let _profile_range = $crate::Range::new($name, $color);
    };
}
