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
//! Minimal GPU driver shim for the pinned-host <-> device copies, cross-stream
//! ordering, and peer (p2p) copies — for BOTH NVIDIA (CUDA driver API) and AMD
//! (HIP runtime API).
#![allow(
    unsafe_code,
    reason = "inherently unsafe FFI over the CUDA/HIP driver APIs; each block is documented with SAFETY"
)]

use std::{ffi::c_void, sync::OnceLock};

use libloading::{Library, Symbol};

use crate::error::{ConnectorError, Result, fatal};

// Both CUDA (`CUresult`) and HIP (`hipError_t`) use 0 for success and 704 for
// "peer access already enabled".
const SUCCESS: i32 = 0;
const PEER_ACCESS_ALREADY_ENABLED: i32 = 704;
// `CU_STREAM_NON_BLOCKING` == `hipStreamNonBlocking` == 0x1;
// `CU_EVENT_DISABLE_TIMING` == `hipEventDisableTiming` == 0x2.
const STREAM_NON_BLOCKING: u32 = 0x1;
const EVENT_DISABLE_TIMING: u32 = 0x2;

/// A device + its (CUDA) primary context. `ctx` is a `CUcontext` on NVIDIA and
/// unused (0) on AMD, where `device_id` (the ordinal) identifies the device.
#[derive(Clone, Copy)]
pub struct DeviceCtx {
    pub device_id: i32,
    pub ctx: u64,
}

/// A stream handle (`CUstream` / `hipStream_t`).
pub type Stream = u64;
/// An event handle (`CUevent` / `hipEvent_t`).
pub type Event = u64;

// CUDA driver API function pointer types.
type CuInit = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuPrimaryCtxRetain = unsafe extern "C" fn(*mut u64, i32) -> i32;
type CuCtxSetCurrent = unsafe extern "C" fn(u64) -> i32;
type CuCtxGetCurrent = unsafe extern "C" fn(*mut u64) -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type CuStreamDestroy = unsafe extern "C" fn(u64) -> i32;
type CuPrimaryCtxRelease = unsafe extern "C" fn(i32) -> i32;
type CuEventCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type CuEventDestroy = unsafe extern "C" fn(u64) -> i32;
type CuEventRecord = unsafe extern "C" fn(u64, u64) -> i32;
type CuStreamWaitEvent = unsafe extern "C" fn(u64, u64, u32) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize, u64) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize, u64) -> i32;
type CuStreamSync = unsafe extern "C" fn(u64) -> i32;
type CuCanAccessPeer = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
type CuEnablePeer = unsafe extern "C" fn(u64, u32) -> i32;
type CuMemcpyPeer = unsafe extern "C" fn(u64, u64, u64, u64, usize, u64) -> i32;

struct CudaApi {
    device_get: CuDeviceGet,
    primary_ctx_retain: CuPrimaryCtxRetain,
    primary_ctx_release: CuPrimaryCtxRelease,
    ctx_set_current: CuCtxSetCurrent,
    ctx_get_current: CuCtxGetCurrent,
    stream_create: CuStreamCreate,
    stream_destroy: CuStreamDestroy,
    event_create: CuEventCreate,
    event_destroy: CuEventDestroy,
    event_record: CuEventRecord,
    stream_wait_event: CuStreamWaitEvent,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    stream_sync: CuStreamSync,
    can_access_peer: CuCanAccessPeer,
    enable_peer: CuEnablePeer,
    memcpy_peer: CuMemcpyPeer,
    _lib: Library,
}

// HIP runtime API function pointer types.
type HipSetDevice = unsafe extern "C" fn(i32) -> i32;
type HipGetDevice = unsafe extern "C" fn(*mut i32) -> i32;
type HipStreamCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type HipStreamDestroy = unsafe extern "C" fn(u64) -> i32;
type HipEventCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type HipEventDestroy = unsafe extern "C" fn(u64) -> i32;
type HipEventRecord = unsafe extern "C" fn(u64, u64) -> i32;
type HipStreamWaitEvent = unsafe extern "C" fn(u64, u64, u32) -> i32;
type HipMemcpyHtoD = unsafe extern "C" fn(u64, *mut c_void, usize, u64) -> i32;
type HipMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize, u64) -> i32;
type HipStreamSync = unsafe extern "C" fn(u64) -> i32;
type HipCanAccessPeer = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
type HipEnablePeer = unsafe extern "C" fn(i32, u32) -> i32;
type HipMemcpyPeer = unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, usize, u64) -> i32;

struct HipApi {
    set_device: HipSetDevice,
    get_device: HipGetDevice,
    stream_create: HipStreamCreate,
    stream_destroy: HipStreamDestroy,
    event_create: HipEventCreate,
    event_destroy: HipEventDestroy,
    event_record: HipEventRecord,
    stream_wait_event: HipStreamWaitEvent,
    memcpy_htod: HipMemcpyHtoD,
    memcpy_dtoh: HipMemcpyDtoH,
    stream_sync: HipStreamSync,
    can_access_peer: HipCanAccessPeer,
    enable_peer: HipEnablePeer,
    memcpy_peer: HipMemcpyPeer,
    _lib: Library,
}

enum Backend {
    Cuda(CudaApi),
    Hip(HipApi),
}

// SAFETY: the API structs hold plain `extern "C"` function pointers (Copy, no
// interior state) plus the owning `Library`; the driver calls they wrap are
// thread-safe. Shared read-only behind a `OnceLock`.
unsafe impl Send for Backend {}
// SAFETY: see the Send impl above.
unsafe impl Sync for Backend {}

/// Loads one symbol as a raw function pointer, keeping it valid for the
/// `Library`'s lifetime (which we store alongside).
macro_rules! load_sym {
    ($lib:expr, $name:literal, $t:ty) => {{
        // SAFETY: the symbol is resolved from the just-opened vendor library and
        // matches the C signature of `$t`; the `Library` is stored in the same
        // struct so the pointer stays valid.
        let sym: Symbol<$t> = unsafe { $lib.get($name) }.map_err(|e| {
            ConnectorError::Config(format!("load {}: {e}", String::from_utf8_lossy($name)))
        })?;
        *sym
    }};
}

fn open_lib(names: &[&str]) -> Option<Library> {
    for name in names {
        // SAFETY: dlopen of a vendor GPU runtime by soname; no init runs beyond
        // the loader's own.
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Some(lib);
        }
    }
    None
}

fn load_cuda() -> Result<CudaApi> {
    let lib = open_lib(&["libcuda.so.1", "libcuda.so"])
        .ok_or_else(|| ConnectorError::Config("libcuda not found".into()))?;
    let init = load_sym!(lib, b"cuInit\0", CuInit);
    // SAFETY: cuInit(0) is the mandatory one-time driver init; idempotent.
    let rc = unsafe { init(0) };
    if rc != SUCCESS {
        return Err(ConnectorError::Config(format!("cuInit failed: {rc}")));
    }
    Ok(CudaApi {
        device_get: load_sym!(lib, b"cuDeviceGet\0", CuDeviceGet),
        primary_ctx_retain: load_sym!(lib, b"cuDevicePrimaryCtxRetain\0", CuPrimaryCtxRetain),
        primary_ctx_release: load_sym!(lib, b"cuDevicePrimaryCtxRelease_v2\0", CuPrimaryCtxRelease),
        ctx_set_current: load_sym!(lib, b"cuCtxSetCurrent\0", CuCtxSetCurrent),
        ctx_get_current: load_sym!(lib, b"cuCtxGetCurrent\0", CuCtxGetCurrent),
        stream_create: load_sym!(lib, b"cuStreamCreate\0", CuStreamCreate),
        stream_destroy: load_sym!(lib, b"cuStreamDestroy_v2\0", CuStreamDestroy),
        event_create: load_sym!(lib, b"cuEventCreate\0", CuEventCreate),
        event_destroy: load_sym!(lib, b"cuEventDestroy_v2\0", CuEventDestroy),
        event_record: load_sym!(lib, b"cuEventRecord\0", CuEventRecord),
        stream_wait_event: load_sym!(lib, b"cuStreamWaitEvent\0", CuStreamWaitEvent),
        memcpy_htod: load_sym!(lib, b"cuMemcpyHtoDAsync_v2\0", CuMemcpyHtoD),
        memcpy_dtoh: load_sym!(lib, b"cuMemcpyDtoHAsync_v2\0", CuMemcpyDtoH),
        stream_sync: load_sym!(lib, b"cuStreamSynchronize\0", CuStreamSync),
        can_access_peer: load_sym!(lib, b"cuDeviceCanAccessPeer\0", CuCanAccessPeer),
        enable_peer: load_sym!(lib, b"cuCtxEnablePeerAccess\0", CuEnablePeer),
        memcpy_peer: load_sym!(lib, b"cuMemcpyPeerAsync\0", CuMemcpyPeer),
        _lib: lib,
    })
}

fn load_hip() -> Result<HipApi> {
    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let candidate = format!("{rocm_path}/lib/libamdhip64.so");
    let lib = open_lib(&[
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
        &candidate,
    ])
    .ok_or_else(|| ConnectorError::Config("libamdhip64 not found".into()))?;
    // hipInit is optional (the runtime lazily initializes); binding happens via
    // hipSetDevice per op.
    Ok(HipApi {
        set_device: load_sym!(lib, b"hipSetDevice\0", HipSetDevice),
        get_device: load_sym!(lib, b"hipGetDevice\0", HipGetDevice),
        stream_create: load_sym!(lib, b"hipStreamCreateWithFlags\0", HipStreamCreate),
        stream_destroy: load_sym!(lib, b"hipStreamDestroy\0", HipStreamDestroy),
        event_create: load_sym!(lib, b"hipEventCreateWithFlags\0", HipEventCreate),
        event_destroy: load_sym!(lib, b"hipEventDestroy\0", HipEventDestroy),
        event_record: load_sym!(lib, b"hipEventRecord\0", HipEventRecord),
        stream_wait_event: load_sym!(lib, b"hipStreamWaitEvent\0", HipStreamWaitEvent),
        memcpy_htod: load_sym!(lib, b"hipMemcpyHtoDAsync\0", HipMemcpyHtoD),
        memcpy_dtoh: load_sym!(lib, b"hipMemcpyDtoHAsync\0", HipMemcpyDtoH),
        stream_sync: load_sym!(lib, b"hipStreamSynchronize\0", HipStreamSync),
        can_access_peer: load_sym!(lib, b"hipDeviceCanAccessPeer\0", HipCanAccessPeer),
        enable_peer: load_sym!(lib, b"hipDeviceEnablePeerAccess\0", HipEnablePeer),
        memcpy_peer: load_sym!(lib, b"hipMemcpyPeerAsync\0", HipMemcpyPeer),
        _lib: lib,
    })
}

/// The detected backend, loaded once. CUDA is preferred; HIP is the fallback.
fn backend() -> Option<&'static Backend> {
    static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            if let Ok(cuda) = load_cuda() {
                Some(Backend::Cuda(cuda))
            } else if let Ok(hip) = load_hip() {
                Some(Backend::Hip(hip))
            } else {
                None
            }
        })
        .as_ref()
}

fn require_backend() -> &'static Backend {
    backend().unwrap_or_else(|| fatal("no CUDA or HIP GPU runtime available"))
}

/// Whether a CUDA or HIP runtime could be loaded (probe used at setup).
#[must_use]
pub fn gpu_available() -> bool {
    backend().is_some()
}

/// Makes `ctx` current on the calling thread: `cuCtxSetCurrent` (CUDA) or
/// `hipSetDevice` (HIP). Aborts on failure (called on the hot path).
fn bind(ctx: &DeviceCtx) {
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: `ctx.ctx` is a live primary context from
            // `retain_primary_context`.
            unsafe { (a.ctx_set_current)(ctx.ctx) }
        }
        Backend::Hip(a) => {
            // SAFETY: `ctx.device_id` is a valid ordinal from
            // `retain_primary_context`.
            unsafe { (a.set_device)(ctx.device_id) }
        }
    };
    if rc != SUCCESS {
        fatal(format!("bind device {} failed: {rc}", ctx.device_id));
    }
}

/// The calling thread's context/device to restore: a `CUcontext` (`0` == none)
/// on CUDA, the current device ordinal on HIP.
enum SavedContext {
    Cuda(u64),
    Hip(i32),
}

/// RAII guard that restores the calling thread's current CUDA context (or HIP
/// current device) when dropped, so a `bind` on a thread the connector does not
/// own leaves that thread's context stack exactly as it found it.
///
/// This matters for in-process interop: `snapshot_compute`/`ComputeGate::drop`
/// and connector construction run on the caller's (scheduler / init) thread —
/// the same thread an external framework (PyTorch, or MAX itself) may use — and
/// `cuCtxSetCurrent` *replaces* the current context rather than pushing it, so
/// without this guard the connector would silently repoint that thread at its
/// own primary context. The connector's private lane threads do not need it:
/// they run only connector code, so leaving the context bound there is fine (and
/// avoids a needless rebind on the copy hot path).
#[must_use]
pub struct ContextGuard {
    prev: SavedContext,
}

impl ContextGuard {
    /// Captures the calling thread's current context/device WITHOUT changing it;
    /// restores it on drop. Use to fence a region that binds internally (e.g.
    /// [`CopyEngine::new`](crate::copy_engine::CopyEngine::new)).
    pub fn save() -> Self {
        let prev = match require_backend() {
            Backend::Cuda(a) => {
                let mut ctx: u64 = 0;
                // SAFETY: out-param initialized; `cuCtxGetCurrent` reads the
                // thread's current context (NULL -> 0 when none is current).
                unsafe { (a.ctx_get_current)(&mut ctx) };
                SavedContext::Cuda(ctx)
            }
            Backend::Hip(a) => {
                let mut dev: i32 = 0;
                // SAFETY: out-param initialized; reads the thread's current device.
                unsafe { (a.get_device)(&mut dev) };
                SavedContext::Hip(dev)
            }
        };
        ContextGuard { prev }
    }

    /// Captures the current context/device, makes `ctx` current, and restores
    /// the captured one on drop.
    fn make_current(ctx: &DeviceCtx) -> Self {
        let guard = Self::save();
        bind(ctx);
        guard
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        // Best-effort: a failed restore on a cleanup/hand-back path must not
        // abort the process, so warn rather than `fatal`.
        let rc = match (require_backend(), &self.prev) {
            // SAFETY: `ctx` was read from `cuCtxGetCurrent` (or 0 == detach);
            // both are valid arguments to `cuCtxSetCurrent`.
            (Backend::Cuda(a), SavedContext::Cuda(ctx)) => unsafe { (a.ctx_set_current)(*ctx) },
            // SAFETY: `dev` was read from `hipGetDevice`.
            (Backend::Hip(a), SavedContext::Hip(dev)) => unsafe { (a.set_device)(*dev) },
            // Backend cannot change after init; the arms above always match.
            _ => SUCCESS,
        };
        if rc != SUCCESS {
            tracing::warn!("restore previous GPU context failed: {rc}");
        }
    }
}

/// Retains the device's primary context (CUDA) / selects the device (HIP).
pub fn retain_primary_context(device_id: u64) -> Result<DeviceCtx> {
    let did = device_id as i32;
    match backend().ok_or_else(|| {
        ConnectorError::Config(format!(
            "no GPU runtime available to retain context for device {device_id}"
        ))
    })? {
        Backend::Cuda(a) => {
            let mut dev: i32 = 0;
            // SAFETY: valid ordinal; out-param initialized.
            let rc = unsafe { (a.device_get)(&mut dev, did) };
            if rc != SUCCESS {
                return Err(ConnectorError::Config(format!(
                    "cuDeviceGet({device_id}) failed: {rc}"
                )));
            }
            let mut ctx: u64 = 0;
            // SAFETY: `dev` is a valid CUdevice; out-param initialized.
            let rc = unsafe { (a.primary_ctx_retain)(&mut ctx, dev) };
            if rc != SUCCESS {
                return Err(ConnectorError::Config(format!(
                    "cuDevicePrimaryCtxRetain({device_id}) failed: {rc}"
                )));
            }
            Ok(DeviceCtx {
                device_id: dev,
                ctx,
            })
        }
        Backend::Hip(a) => {
            // SAFETY: valid ordinal; validates the device exists.
            let rc = unsafe { (a.set_device)(did) };
            if rc != SUCCESS {
                return Err(ConnectorError::Config(format!(
                    "hipSetDevice({device_id}) failed: {rc}"
                )));
            }
            Ok(DeviceCtx {
                device_id: did,
                ctx: 0,
            })
        }
    }
}

/// Creates a non-blocking auxiliary copy stream in `ctx`.
pub fn new_copy_stream(ctx: &DeviceCtx) -> Result<Stream> {
    bind(ctx);
    let mut stream: u64 = 0;
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: bound context is current; out-param initialized.
            unsafe { (a.stream_create)(&mut stream, STREAM_NON_BLOCKING) }
        }
        Backend::Hip(a) => {
            // SAFETY: bound device is current; out-param initialized.
            unsafe { (a.stream_create)(&mut stream, STREAM_NON_BLOCKING) }
        }
    };
    if rc != SUCCESS {
        return Err(ConnectorError::Config(format!(
            "create copy stream failed: {rc}"
        )));
    }
    Ok(stream)
}

// The helpers below run on the transfer critical path; a GPU failure there is
// unrecoverable, so they `fatal` (abort) rather than return `Result`.

/// Records a timing-disabled event on MAX's compute stream (`main_stream`) at
/// its current point, for an aux stream to wait on. MUST be called on the thread
/// that enqueues the compute stream. Aborts on failure.
pub fn record_compute_event(ctx: &DeviceCtx, main_stream: u64) -> Event {
    // Runs on the caller's scheduler thread, so restore its context on return.
    let _ctx_guard = ContextGuard::make_current(ctx);
    let mut event: u64 = 0;
    let rc_create = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: context current; out-param initialized.
            unsafe { (a.event_create)(&mut event, EVENT_DISABLE_TIMING) }
        }
        Backend::Hip(a) => {
            // SAFETY: device current; out-param initialized.
            unsafe { (a.event_create)(&mut event, EVENT_DISABLE_TIMING) }
        }
    };
    if rc_create != SUCCESS {
        fatal(format!("create event failed: {rc_create}"));
    }
    let rc_record = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: `event` just created; recording on MAX's compute stream.
            unsafe { (a.event_record)(event, main_stream) }
        }
        Backend::Hip(a) => {
            // SAFETY: `event` just created; recording on MAX's compute stream.
            unsafe { (a.event_record)(event, main_stream) }
        }
    };
    if rc_record != SUCCESS {
        fatal(format!(
            "record event on compute stream failed: {rc_record}"
        ));
    }
    event
}

/// Makes `aux` wait on `event` (an aux-stream enqueue). Aborts on failure.
pub fn stream_wait_event(aux: Stream, event: Event) {
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: valid stream + event handles; thread-safe enqueue.
            unsafe { (a.stream_wait_event)(aux, event, 0) }
        }
        Backend::Hip(a) => {
            // SAFETY: valid stream + event handles; thread-safe enqueue.
            unsafe { (a.stream_wait_event)(aux, event, 0) }
        }
    };
    if rc != SUCCESS {
        fatal(format!("stream wait event failed: {rc}"));
    }
}

/// Async host-to-device copy of `len` bytes on `aux`. Aborts on failure.
pub fn memcpy_htod_async(
    ctx: &DeviceCtx,
    aux: Stream,
    dst_device: u64,
    src_host: usize,
    len: usize,
) {
    bind(ctx);
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: dst_device has >= len bytes; src_host is a pinned region of
            // >= len bytes; ordered on aux and awaited before use.
            unsafe { (a.memcpy_htod)(dst_device, src_host as *const c_void, len, aux) }
        }
        Backend::Hip(a) => {
            // SAFETY: see the CUDA arm; HIP takes a non-const src pointer.
            unsafe { (a.memcpy_htod)(dst_device, src_host as *mut c_void, len, aux) }
        }
    };
    if rc != SUCCESS {
        fatal(format!(
            "memcpy H2D(dst={dst_device:#x}, len={len}) failed: {rc}"
        ));
    }
}

/// Async device-to-host copy of `len` bytes on `aux`. Aborts on failure.
pub fn memcpy_dtoh_async(
    ctx: &DeviceCtx,
    aux: Stream,
    dst_host: usize,
    src_device: u64,
    len: usize,
) {
    bind(ctx);
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: src_device has >= len bytes; dst_host is a pinned region of
            // >= len bytes; ordered after the main stream so writes are visible.
            unsafe { (a.memcpy_dtoh)(dst_host as *mut c_void, src_device, len, aux) }
        }
        Backend::Hip(a) => {
            // SAFETY: see the CUDA arm.
            unsafe { (a.memcpy_dtoh)(dst_host as *mut c_void, src_device, len, aux) }
        }
    };
    if rc != SUCCESS {
        fatal(format!(
            "memcpy D2H(src={src_device:#x}, len={len}) failed: {rc}"
        ));
    }
}

/// Blocks until all work queued on `aux` completes. Aborts on failure.
pub fn sync_stream(aux: Stream) {
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: valid stream handle.
            unsafe { (a.stream_sync)(aux) }
        }
        Backend::Hip(a) => {
            // SAFETY: valid stream handle.
            unsafe { (a.stream_sync)(aux) }
        }
    };
    if rc != SUCCESS {
        fatal(format!("stream synchronize failed: {rc}"));
    }
}

// The destroy/release helpers below run only on teardown paths (a `ComputeGate`
// drop or `CopyEngine::drop`), so they are best-effort: a failed cleanup warns
// rather than aborting, unlike the transfer-critical helpers above.

/// Destroys an event created by [`record_compute_event`]. `load`/`offload`
/// record one event per device per batch, so without this the driver-side event
/// handles accumulate unboundedly over the serve process's lifetime.
pub fn destroy_event(ctx: &DeviceCtx, event: Event) {
    // A `ComputeGate` can drop on the scheduler thread (if it holds the last
    // reference), so restore that thread's context on return.
    let _ctx_guard = ContextGuard::make_current(ctx);
    let rc = match require_backend() {
        // SAFETY: `event` came from `record_compute_event`; its owning
        // `ComputeGate` is the sole owner and drops once, so this destroys once.
        Backend::Cuda(a) => unsafe { (a.event_destroy)(event) },
        // SAFETY: see the CUDA arm.
        Backend::Hip(a) => unsafe { (a.event_destroy)(event) },
    };
    if rc != SUCCESS {
        tracing::warn!("destroy event failed: {rc}");
    }
}

/// Destroys an auxiliary copy stream created by [`new_copy_stream`]. Call after
/// [`sync_stream`] has drained it.
pub fn destroy_stream(stream: Stream) {
    let rc = match require_backend() {
        // SAFETY: `stream` came from `new_copy_stream` and is destroyed once at
        // teardown, after its in-flight copies were drained.
        Backend::Cuda(a) => unsafe { (a.stream_destroy)(stream) },
        // SAFETY: see the CUDA arm.
        Backend::Hip(a) => unsafe { (a.stream_destroy)(stream) },
    };
    if rc != SUCCESS {
        tracing::warn!("destroy stream failed: {rc}");
    }
}

/// Releases a primary context retained by [`retain_primary_context`], balancing
/// its retain. CUDA only: the HIP path never retained one (it just selects the
/// device), so there is nothing to release there.
pub fn release_primary_context(ctx: &DeviceCtx) {
    match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: `ctx.device_id` is the CUdevice whose primary context we
            // retained exactly once in `retain_primary_context`; this balances it
            // (MAX holds its own independent retains, so the context stays live).
            let rc = unsafe { (a.primary_ctx_release)(ctx.device_id) };
            if rc != SUCCESS {
                tracing::warn!("release primary context failed: {rc}");
            }
        }
        Backend::Hip(_) => {}
    }
}

/// Whether `from` can peer-access `to` (NVLink / xGMI / p2p).
pub fn can_access_peer(from: &DeviceCtx, to: &DeviceCtx) -> Result<bool> {
    let mut out: i32 = 0;
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: pure query over two valid device ordinals.
            unsafe { (a.can_access_peer)(&mut out, from.device_id, to.device_id) }
        }
        Backend::Hip(a) => {
            // SAFETY: pure query over two valid device ordinals.
            unsafe { (a.can_access_peer)(&mut out, from.device_id, to.device_id) }
        }
    };
    if rc != SUCCESS {
        return Err(ConnectorError::Config(format!(
            "can access peer failed: {rc}"
        )));
    }
    Ok(out != 0)
}

/// Enables `target` to peer-access `peer`'s memory (idempotent).
pub fn enable_peer_access(target: &DeviceCtx, peer: &DeviceCtx) -> Result<()> {
    bind(target);
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: target is current; peer.ctx is a valid context.
            unsafe { (a.enable_peer)(peer.ctx, 0) }
        }
        Backend::Hip(a) => {
            // SAFETY: target is current; peer.device_id is a valid ordinal.
            unsafe { (a.enable_peer)(peer.device_id, 0) }
        }
    };
    if rc != SUCCESS && rc != PEER_ACCESS_ALREADY_ENABLED {
        return Err(ConnectorError::Config(format!(
            "enable peer access failed: {rc}"
        )));
    }
    Ok(())
}

/// Async p2p copy of `len` bytes `src_device` (in `src`) -> `dst_device` (in
/// `dst`) on `stream`. Used for the MLA-replicated fan-out. Aborts on failure.
pub fn memcpy_peer_async(
    stream: Stream,
    dst_device: u64,
    dst: &DeviceCtx,
    src_device: u64,
    src: &DeviceCtx,
    len: usize,
) {
    // The copy runs on `stream`, which belongs to `src`'s device.
    bind(src);
    let rc = match require_backend() {
        Backend::Cuda(a) => {
            // SAFETY: both device pointers have >= len bytes in their owning
            // contexts; ordered on stream and drained before use.
            unsafe { (a.memcpy_peer)(dst_device, dst.ctx, src_device, src.ctx, len, stream) }
        }
        Backend::Hip(a) => {
            // SAFETY: see the CUDA arm; HIP identifies peers by device ordinal.
            unsafe {
                (a.memcpy_peer)(
                    dst_device as *mut c_void,
                    dst.device_id,
                    src_device as *const c_void,
                    src.device_id,
                    len,
                    stream,
                )
            }
        }
    };
    if rc != SUCCESS {
        fatal(format!(
            "memcpy peer(dst={dst_device:#x}, src={src_device:#x}, len={len}) failed: {rc}"
        ));
    }
}
