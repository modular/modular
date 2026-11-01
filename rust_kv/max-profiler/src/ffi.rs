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
//! Hand-declared dynamic-loader FFI and the `m_ext_profiler_*` symbol
//! types.
//!
//! Declared by hand (rather than via `libloading` or the `libc` crate) to
//! keep this crate dependency-free: it is consumed both by the cargo-only
//! mach workspace and by the `rust_kv` bazel `crate.from_cargo` universe, where a
//! zero-dep crate adds nothing to the third-party crate set.

use std::ffi::{c_char, c_int, c_void};

unsafe extern "C" {
    pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

pub const RTLD_NOW: c_int = 2;

/// Pseudo-handle asking `dlsym` to search the process's global scope.
#[cfg(target_os = "linux")]
pub const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();
#[cfg(target_os = "macos")]
pub const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

/// The shim's soname; resolved against the loader search path (RUNPATH of
/// the executable, `LD_LIBRARY_PATH`) when the symbols are not already in
/// the process.
#[cfg(target_os = "linux")]
pub const SHIM_SONAME: &std::ffi::CStr = c"libmax_profiler_shim.so";
#[cfg(target_os = "macos")]
pub const SHIM_SONAME: &std::ffi::CStr = c"libmax_profiler_shim.dylib";

// Signatures of the m_ext_profiler_* surface; must stay in sync with
// max/internal/profiler/lib/ProfilerShim.h (the shim's additive-only
// contract makes drift a missing symbol, not a silent ABI break).
pub type IsEnabledFn = unsafe extern "C" fn() -> c_int;
pub type SetEnabledFn = unsafe extern "C" fn(c_int);
pub type RangePushFn = unsafe extern "C" fn(*const c_char, u32) -> c_int;
pub type RangePopFn = unsafe extern "C" fn();
pub type RangeStartFn = unsafe extern "C" fn(*const c_char, u32) -> u64;
pub type RangeEndFn = unsafe extern "C" fn(u64);
pub type MarkFn = unsafe extern "C" fn(*const c_char, u32);
