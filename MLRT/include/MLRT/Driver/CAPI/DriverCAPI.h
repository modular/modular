//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// The MAX driver's C interface, for consumers that cannot include the C++
// headers and reach the driver through a C ABI instead.
//
// This is the C-side mirror of the Python bindings in
// `max/python/max/_core/driver.pyi`: the same devices, streams, events,
// buffers, peer-access queries and pointer-level copies, with the same
// semantics, so a non-Python consumer can drive the same hardware through the
// same objects MAX itself uses.
//
// Design rules:
//
//  * Every entry point that can fail returns `M_DriverStatus`, the destroy
//    calls included. The queries that return a size report failure as 0 with
//    a message, and the version and the accelerator count never fail.
//    `M_DRIVER_OK` is success; anything else is a failure, and
//    `M_DriverLastErrorMessage` returns the message. No exception crosses the
//    boundary, because the library is built without exceptions, and no argument
//    is used before it is checked, so a bad argument is a status, not a crash.
//    That includes two handles from different backends, such as a CPU device
//    waited on from a CUDA stream, which is `M_DRIVER_INVALID_ARGUMENT`
//    wherever a call takes two.
//  * Handles are refcounted and owned by the caller. Every `...Create`,
//    `...From...` or `...Allocate` returns a handle the caller must destroy
//    with the matching `M_Driver...Destroy`, which refuses a null handle with
//    `M_DRIVER_INVALID_ARGUMENT`. A `const` handle argument never transfers
//    ownership and never destroys.
//  * Handles may be used from any thread, and the driver makes its own device
//    context current around each operation. Two threads must not act on the
//    same event at once, because recording an event on one thread while
//    another queries or waits on it races inside the backends. Callers need
//    not bind a context before calling in, but they may bind one with
//    `M_DriverDeviceBindToCurrentThread` for foreign code that expects the
//    device current on the calling thread, such as a transport registering
//    device memory.
//  * Byte sizes are `uint64_t` and always in bytes. Addresses and foreign
//    objects are `void *`, which says that the value is an identity to hand
//    around and not something to read through on the host. The entry points
//    that adopt an existing object by address say so, and an address that did
//    not come from this interface or from MAX is undefined behavior.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_CAPI_DRIVERCAPI_H
#define MLRT_DRIVER_CAPI_DRIVERCAPI_H

#include "Support/SymbolExport.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every entry point is exported from the driver's C shared library
// (//MLRT:Driver/CAPIShim), which is built with `-fvisibility=hidden`. Nothing
// inside the library references them, so without an export attribute the
// linker drops the definitions and a consumer that loads the library at run
// time sees an empty one. This is the visibility half of
// `MODULAR_DRIVER_EXPORT`; the `extern "C"` half is the block above.
#define M_DRIVER_API DRIVER_VISIBILITY_EXPORT

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

/// The version of this interface, in three parts. The major number changes
/// when an existing declaration changes shape or meaning. The minor number
/// grows when anything is added, whether an entry point, a status value or an
/// enum value. The patch number grows for a fix that changes no declaration,
/// so a consumer can tell two libraries with the same surface apart. A
/// consumer built against an older minor keeps working against a newer
/// library with the same major, but it can receive a status or enum value it
/// does not know, so it treats an unknown status as a failure and an unknown
/// enum value as unsupported rather than ignoring it.
#define M_DRIVER_CAPI_VERSION_MAJOR 1
#define M_DRIVER_CAPI_VERSION_MINOR 0
#define M_DRIVER_CAPI_VERSION_PATCH 0

/// Returns the interface version the library implements, packed as
/// `(major << 24) | (minor << 16) | patch`. A consumer that resolves the entry
/// points at run time resolves this one first, and then requires the major it
/// was built against and a minor at least as large as the one it was built
/// against before resolving anything else. Never fails.
M_DRIVER_API uint32_t M_DriverCApiVersion(void);

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

/// Status returned by the entry points in this header. `M_DRIVER_OK` is
/// success and the rest are failures. The codes say what kind of failure it
/// was and not whether a retry can help, because a capability gap this layer
/// cannot tell from a driver failure comes back as `M_DRIVER_RUNTIME_ERROR`
/// and an allocation that failed for any reason as `M_DRIVER_OUT_OF_MEMORY`,
/// so a caller that retries reads the message first.
typedef enum M_DriverStatus {
  /// The operation completed.
  M_DRIVER_OK = 0,
  /// A null handle, a null out-pointer, an out-of-range ordinal, or an
  /// unusable size.
  M_DRIVER_INVALID_ARGUMENT = 1,
  /// The device, driver, or build does not support this operation, or the
  /// handle was created in a mode that forbids it (e.g. a copy on a stream
  /// created from a native handle).
  M_DRIVER_NOT_SUPPORTED = 2,
  /// The driver reported a failure, the operation was rejected by the
  /// driver's own state (busy peer, mismatched device, etc.), or a backend
  /// lacks the operation where this layer cannot tell that apart from a
  /// driver failure, such as the allocation range on the CPU device or the
  /// elapsed time between two CPU events.
  M_DRIVER_RUNTIME_ERROR = 3,
  /// An allocation failed. The message says whether the device was out of
  /// memory, because a pending error taken from the shared context (see the
  /// Devices section) and a backend that cannot allocate the kind requested
  /// come back as this code too.
  M_DRIVER_OUT_OF_MEMORY = 4,
} M_DriverStatus;

/// Copies the message describing the most recent failed call on THIS thread
/// into `buffer`, NUL-terminated, and returns the size in bytes the message
/// needs INCLUDING the NUL terminator.
///
/// Copies nothing when `buffer` is null or `capacity` is too small; call with
/// a null buffer to size the allocation. The message is cleared by a
/// successful call, so read it before calling anything else on this thread.
///
/// @param buffer Destination, or null to query the required size.
/// @param capacity Bytes writable at `buffer`.
/// @returns Required size in bytes, including the NUL terminator; 0 when the
///   last call on this thread succeeded.
M_DRIVER_API size_t M_DriverLastErrorMessage(char *buffer, size_t capacity);

//===----------------------------------------------------------------------===//
// Handles
//===----------------------------------------------------------------------===//
//
// Opaque, refcounted handles, declared up front so every signature in this
// header can name them. The section that defines each handle's operations
// documents what it is and how it is created and destroyed.

typedef struct M_DriverDevice M_DriverDevice;
typedef struct M_DriverStream M_DriverStream;
typedef struct M_DriverEvent M_DriverEvent;
typedef struct M_DriverBuffer M_DriverBuffer;
typedef struct M_DriverMapping M_DriverMapping;

//===----------------------------------------------------------------------===//
// Devices
//===----------------------------------------------------------------------===//

/// A device and the driver context that executes its work. The context is the
/// process-wide one for that (api, id) pair, shared with every other consumer
/// in this process, including MAX itself (see `DeviceContext::getOrCreate`):
/// memory allocated through one handle is addressable through another.
///
/// Sharing the context shares its pending error slot: a call that checks it
/// first, such as `M_DriverDeviceSynchronize`, reports and clears an error
/// MAX recorded there. Events, the unordered peer copy's source and host
/// registration use a context this interface owns and leave that slot alone.
///
/// Create with `M_DriverAcceleratorDeviceCreate` or `M_DriverCPUDeviceCreate`;
/// destroy with `M_DriverDeviceDestroy`.

/// Number of accelerators the driver can see, or 0 when none are available
/// (no driver installed, no device, or an unsupported host). Never fails and
/// never aborts, so it is safe on a machine without a GPU.
///
/// Mirrors `max._core.driver.accelerator_count()`.
M_DRIVER_API int64_t M_DriverAcceleratorCount(void);

/// Retains the accelerator at `ordinal` and its context.
///
/// `ordinal` is in `[0, M_DriverAcceleratorCount())`; the context is
/// process-wide, so a handle obtained here shares memory and ordering with
/// MAX's own use of the same device.
///
/// @param ordinal Device ordinal.
/// @param out Receives the new handle; must be destroyed by the caller.
/// @returns `M_DRIVER_INVALID_ARGUMENT` for a null `out` or an out-of-range
///   ordinal, `M_DRIVER_RUNTIME_ERROR` when the driver cannot create the
///   device or its context.
///
/// Mirrors `max._core.driver.Accelerator(id=ordinal)`.
M_DRIVER_API M_DriverStatus
M_DriverAcceleratorDeviceCreate(int64_t ordinal, M_DriverDevice **out);

/// Retains the one CPU device and its context.
///
/// The CPU context is a host memory space: buffers allocated from it are host
/// memory, and copies through it stage host to host. It is this interface's
/// process-wide CPU context and not the one MAX's Python side builds for its
/// CPU device, so waiting on it or synchronizing it orders nothing against
/// MAX's host work.
///
/// @param id 0 names the one CPU device, and -1 is accepted as the default
///   for it; no other value names a CPU device.
/// @param out Receives the new handle; must be destroyed by the caller.
M_DRIVER_API M_DriverStatus M_DriverCPUDeviceCreate(int64_t id,
                                                    M_DriverDevice **out);

/// Releases a device handle and its context reference. A null handle is
/// `M_DRIVER_INVALID_ARGUMENT`. The driver keeps the underlying device alive
/// as long as any other consumer (MAX, another handle) uses it.
M_DRIVER_API M_DriverStatus M_DriverDeviceDestroy(M_DriverDevice *device);

/// Writes the device's ordinal to `*out`.
M_DRIVER_API M_DriverStatus M_DriverDeviceId(const M_DriverDevice *device,
                                             uint64_t *out);

/// Writes the device's API name to `buffer`: `"cuda"`, `"hip"`, `"metal"`,
/// `"cpu"`, or `"gpu"` for a generic accelerator handle whose concrete API was
/// not yet resolved. This is the portable way for a caller to branch on
/// vendor without linking a vendor library.
///
/// Returns the size the name needs including the NUL terminator; copies only
/// when `capacity` is sufficient.
///
/// Mirrors `max._core.driver.Device.api`.
M_DRIVER_API size_t M_DriverDeviceApi(const M_DriverDevice *device,
                                      char *buffer, size_t capacity);

/// Writes the device's architecture name to `buffer`: the CUDA compute
/// capability (`"sm_90"`), the AMD target (`"gfx950"`), or the Metal GPU
/// family. Use it to select precompiled kernels. The CPU device has no
/// architecture name and reports 0 with a message.
///
/// Returns the size the name needs including the NUL terminator; copies only
/// when `capacity` is sufficient.
///
/// Mirrors `max._core.driver.Device.architecture_name`.
M_DRIVER_API size_t M_DriverDeviceArchitectureName(const M_DriverDevice *device,
                                                   char *buffer,
                                                   size_t capacity);

/// Writes the device's marketing name to `buffer` (e.g. `"NVIDIA H100"`).
///
/// Returns the size the name needs including the NUL terminator; copies only
/// when `capacity` is sufficient.
///
/// Mirrors `max._core.driver.Device.model_name`.
M_DRIVER_API size_t M_DriverDeviceModelName(const M_DriverDevice *device,
                                            char *buffer, size_t capacity);

/// Writes the device's PCI bus ID to `buffer` (e.g. `"0000:65:00.0"`), the
/// identity a caller uses to look the device up in the host's topology -- for
/// example to read its NUMA node from sysfs before pinning host memory. The
/// spelling is the vendor's: HIP writes a four digit lowercase domain, while
/// NVIDIA's comes from NVML, so normalize before comparing across vendors.
///
/// Returns the size the identity needs including the NUL terminator, or 0 when
/// the backend cannot report it (the query needs NVML on NVIDIA): read
/// `M_DriverLastErrorMessage` to tell the two apart.
M_DRIVER_API size_t M_DriverDevicePciBusId(const M_DriverDevice *device,
                                           char *buffer, size_t capacity);

/// Writes the device's 16 byte UUID to `out_uuid`, the identity that names
/// the same device across processes and across the ordinal numbering of
/// different APIs. The bytes are the vendor's and are not formatted: CUDA
/// gives a binary UUID, while HIP gives 16 ASCII hex characters, so a caller
/// that renders them as a canonical UUID should not expect it to read as one
/// on every vendor. Compare them as bytes.
///
/// Returns `M_DRIVER_NOT_SUPPORTED` when the backend has no UUID, which
/// includes the CPU device and a vendor driver that predates the query (CUDA
/// 11.4 for `cuDeviceGetUuid_v2`, which gives each MIG instance its own).
M_DRIVER_API M_DriverStatus M_DriverDeviceUuid(const M_DriverDevice *device,
                                               uint8_t out_uuid[16]);

/// Writes whether `device` can access `peer`'s memory directly (NVLink, or
/// xGMI on AMD) to `*out`. Querying never enables anything and never fails for
/// two devices on the same host.
///
/// Mirrors `max._core.driver.Device.can_access`.
M_DRIVER_API M_DriverStatus M_DriverDeviceCanAccess(
    const M_DriverDevice *device, const M_DriverDevice *peer, bool *out);

/// Enables `device`'s access to `peer`'s memory: `device`'s context may then
/// address `peer`'s allocations. Enable each direction the caller needs.
///
/// Idempotent, so enabling an already-enabled pair succeeds. Use this rather
/// than `M_DriverEnableAllPeerAccess` when only some pairs of a multi-device
/// topology can reach each other -- enabling all at once fails as a whole if
/// any pair cannot.
M_DRIVER_API M_DriverStatus M_DriverDeviceEnablePeerAccess(
    const M_DriverDevice *device, const M_DriverDevice *peer);

/// Waits for all work enqueued on the device's context, then runs the
/// unregisters `M_DriverMappingDestroy` queued for this device and reports a
/// refused one as its status.
///
/// Mirrors `max._core.driver.Device.synchronize`.
M_DRIVER_API M_DriverStatus
M_DriverDeviceSynchronize(const M_DriverDevice *device);

/// The free byte count M_DriverDeviceMemoryInfo writes when the driver cannot
/// measure the device's free memory. A free count of 0 always means the device
/// is full.
#define M_DRIVER_MEMORY_UNKNOWN UINT64_MAX

/// Writes the total and free bytes of the device's memory to `*out_total` and
/// `*out_free`. Either out-pointer may be null. `*out_free` is
/// `M_DRIVER_MEMORY_UNKNOWN` when the driver cannot measure free memory.
///
/// Mirrors `max._core.driver.Device.stats`, whose free figure is `None` in
/// that case.
M_DRIVER_API M_DriverStatus M_DriverDeviceMemoryInfo(
    const M_DriverDevice *device, uint64_t *out_total, uint64_t *out_free);

/// Resolves the driver allocation that contains `address`: writes its base to
/// `*out_base` and its byte size to `*out_size`.
///
/// This is the extent a transport may register or export as one unit (CUDA:
/// `cuMemGetAddressRange`, HIP: `hipMemGetAddressRange`); the same boundaries
/// UCX exports. `address` must be inside an allocation of this device, and a
/// null one is `M_DRIVER_INVALID_ARGUMENT`. CUDA and HIP answer; a backend
/// without the query, such as the CPU device, returns `M_DRIVER_RUNTIME_ERROR`
/// with a message saying so.
M_DRIVER_API M_DriverStatus
M_DriverDeviceAllocationRange(const M_DriverDevice *device, const void *address,
                              void **out_base, uint64_t *out_size);

/// Enables bidirectional peer access between every pair of accelerators in the
/// process. Idempotent.
///
/// Mirrors `max._core.driver.enable_all_peer_access`.
M_DRIVER_API M_DriverStatus M_DriverEnableAllPeerAccess(void);

/// Writes whether peer access is enabled for every accelerator pair to `*out`.
M_DRIVER_API M_DriverStatus M_DriverAllPeerAccessEnabled(bool *out);

/// Returns the driver's own context object for `device` as an opaque address,
/// so MAX's Python side (or a handle from another MAX API) and this interface
/// can refer to the same context.
///
/// Opaque: valid only as an identity to hand back to MAX, never to dereference
/// or to pass to a vendor API.
M_DRIVER_API M_DriverStatus
M_DriverDeviceContextHandle(const M_DriverDevice *device, void **out);

/// Makes `device`'s context current on the calling thread, and keeps it so
/// until `M_DriverUnbindFromCurrentThread` or a bind of another device.
///
/// Nothing in this interface needs it, because every entry point makes its own
/// context current around the operation. It exists for foreign code called
/// from the same thread that expects the vendor's current device to be set,
/// such as a transport that registers device memory. Binding the device that
/// is already bound is a no-op.
M_DRIVER_API M_DriverStatus
M_DriverDeviceBindToCurrentThread(const M_DriverDevice *device);

/// Clears the binding `M_DriverDeviceBindToCurrentThread` made on the calling
/// thread. Safe to call when nothing is bound.
M_DRIVER_API void M_DriverUnbindFromCurrentThread(void);

//===----------------------------------------------------------------------===//
// Streams, queues and buffers
//===----------------------------------------------------------------------===//

/// An in-order submission queue: a context of its own on the device's shared
/// memory, so copies issued through it are ordered against each other and,
/// via events, against work enqueued through other streams.
///
/// Create with `M_DriverStreamCreate`, or adopt MAX's own stream with
/// `M_DriverStreamFromContextHandle`; destroy with `M_DriverStreamDestroy`.

/// Creates a stream: a context on `device` plus its submission queue. The
/// stream does not synchronize the device, and it does not record events
/// implicitly.
///
/// The queue does not synchronize with the vendor's legacy default stream, so
/// copies on it only wait for what the caller orders them after, which is what
/// an asynchronous copy engine needs. On CUDA and HIP that is a non blocking
/// queue from the context's pool, and a backend with no legacy default stream
/// has nothing to isolate from and hands out its usual queue. A library built
/// with the HAL device context hands out the context's base queue instead,
/// which may be blocking, because that build keeps no pool.
M_DRIVER_API M_DriverStatus M_DriverStreamCreate(const M_DriverDevice *device,
                                                 M_DriverStream **out);

/// Adopts a driver context, previously observed through
/// `M_DriverDeviceContextHandle` or `M_DriverStreamDeviceContextHandle`, as a
/// stream.
///
/// This is how a caller adopts a stream MAX already owns -- MAX's compute
/// stream for a device -- so its copies and event records are ordered against
/// MAX's work. The handle keeps the context alive.
///
/// Every operation is available on the returned stream, including copies into
/// memory that belongs to that context.
///
/// `context_handle` must be an address this interface or MAX produced. There
/// is no registry to validate it against, so any other value is undefined
/// behavior rather than a status, and only null is refused.
M_DRIVER_API M_DriverStatus
M_DriverStreamFromContextHandle(void *context_handle, M_DriverStream **out);

/// Adopts a vendor-native queue handle (a `CUstream` or `hipStream_t`) as a
/// stream.
///
/// The returned stream supports ordering operations only -- synchronization,
/// event record and event waits -- which is what a caller needs to order its
/// own copies against a queue it does not own. Copies through it return
/// `M_DRIVER_NOT_SUPPORTED`: a copy must be enqueued on the context that owns
/// the destination memory, which is why callers that need copies adopt MAX's
/// context with `M_DriverStreamFromContextHandle` instead.
///
/// Adopting creates a driver context of its own, with a vendor stream the
/// ordering operations never use, so it costs a stream creation each time.
/// Adopt a queue once and keep the handle rather than adopting it per call.
M_DRIVER_API M_DriverStatus M_DriverStreamFromNativeHandle(
    const M_DriverDevice *device, void *native_handle, M_DriverStream **out);

/// Releases a stream handle. A null handle is `M_DRIVER_INVALID_ARGUMENT`.
/// Work already enqueued still runs.
M_DRIVER_API M_DriverStatus M_DriverStreamDestroy(M_DriverStream *stream);

/// Writes the stream's native queue handle to `*out`: a `CUstream` or
/// `hipStream_t` value, or null when the backend exposes no such handle. This
/// is the escape hatch for passing the queue to an API that takes a raw
/// handle.
M_DRIVER_API M_DriverStatus
M_DriverStreamNativeHandle(const M_DriverStream *stream, void **out);

/// Writes the opaque handle of the stream's driver context to `*out`, so it
/// can be adopted elsewhere with `M_DriverStreamFromContextHandle`. See
/// `M_DriverDeviceContextHandle`.
///
/// A stream adopted from a native handle has no context to hand out and
/// returns `M_DRIVER_NOT_SUPPORTED`, because the context created beside that
/// queue is not ordered against it, so adopting it would give a stream whose
/// copies run out of order with the queue.
M_DRIVER_API M_DriverStatus
M_DriverStreamDeviceContextHandle(const M_DriverStream *stream, void **out);

/// Retains the device the stream runs on, as a handle on the device's
/// process-wide context, the same handle `M_DriverAcceleratorDeviceCreate`
/// gives for that device and not the stream's own context. The caller owns the
/// returned handle.
///
/// Mirrors `max._core.driver.DeviceQueue.device`.
M_DRIVER_API M_DriverStatus M_DriverStreamDevice(const M_DriverStream *stream,
                                                 M_DriverDevice **out);

/// Waits until all work enqueued on `stream` has completed.
///
/// Mirrors `max._core.driver.DeviceQueue.synchronize`.
M_DRIVER_API M_DriverStatus
M_DriverStreamSynchronize(const M_DriverStream *stream);

/// Orders all work after this call on `stream` to start only once `other`'s
/// work enqueued so far has completed. This is the vendor-agnostic way to
/// order one unit of work after another; it does not block the host, and it
/// does not synchronize either stream.
///
/// Waiting for the stream itself, through the same handle or another handle
/// for the same context, is harmless, because what is already enqueued there
/// is ordered ahead of it regardless. It is not free on every backend, since
/// the CPU and Metal backends return without enqueuing anything while CUDA
/// and HIP record an event and wait on it.
///
/// When either side is a vendor-native queue, the ordering is realized with an
/// event: the source queue records one and the destination waits on it, since a
/// native queue is not a context this interface can wait on directly. A wait
/// therefore appends work to both queues.
///
/// Mirrors `max._core.driver.DeviceQueue.wait_for`.
M_DRIVER_API M_DriverStatus M_DriverStreamWaitFor(const M_DriverStream *stream,
                                                  const M_DriverStream *other);

/// Like `M_DriverStreamWaitFor`, for all work enqueued on `device`'s context.
///
/// Mirrors `max._core.driver.DeviceQueue.wait_for(device)`.
M_DRIVER_API M_DriverStatus M_DriverStreamWaitForDevice(
    const M_DriverStream *stream, const M_DriverDevice *device);

/// Orders work after this call on `stream` to start only once `event` has been
/// recorded and its preceding work completed.
///
/// `event` may belong to another device: this is how a cross-device broadcast
/// orders its writes after the compute stream that produced their source.
M_DRIVER_API M_DriverStatus M_DriverStreamWaitForEvent(
    const M_DriverStream *stream, const M_DriverEvent *event);

/// Creates an event on `stream`'s device and records it on the stream,
/// returning the recording through `*out`.
///
/// The caller owns the returned event; destroying it after it has fired and
/// been waited on is safe.
///
/// Mirrors `max._core.driver.DeviceQueue.record_event()`.
M_DRIVER_API M_DriverStatus
M_DriverStreamRecordNewEvent(const M_DriverStream *stream, M_DriverEvent **out);

/// Records `event` on `stream`: the event fires once everything enqueued on
/// the stream so far has completed. Does not block the host.
///
/// Mirrors `max._core.driver.DeviceQueue.record_event(event)`.
M_DRIVER_API M_DriverStatus M_DriverStreamRecordEvent(
    const M_DriverStream *stream, const M_DriverEvent *event);

//===----------------------------------------------------------------------===//
// Events
//===----------------------------------------------------------------------===//

/// A point in a stream's timeline that the host, or another stream, can wait
/// on. Create with `M_DriverEventCreate`; destroy with
/// `M_DriverEventDestroy`.

/// Creates an event on `device`.
///
/// `enable_timing` asks the driver to timestamp the event so that
/// `M_DriverEventElapsedTimeMs` can measure between two of them. Timing is
/// fixed at creation: an event created without it cannot be used to measure,
/// and both events of a measurement need it. A backend that cannot timestamp
/// either refuses this call, as Metal does, or accepts it and fails the
/// measurement instead, as the CPU device does, so a caller checks the
/// measurement and not only the creation. Callers that only need ordering
/// pass false.
///
/// The event is created through the context this interface owns on `device`
/// rather than the shared one, so creating it neither reports nor clears an
/// error MAX left pending (see the Devices section). It records and waits on
/// any stream of the device all the same.
///
/// Mirrors `max._core.driver.DeviceEvent(device, enable_timing)`.
M_DRIVER_API M_DriverStatus M_DriverEventCreate(const M_DriverDevice *device,
                                                bool enable_timing,
                                                M_DriverEvent **out);

/// Releases an event handle. A null handle is `M_DRIVER_INVALID_ARGUMENT`.
M_DRIVER_API M_DriverStatus M_DriverEventDestroy(M_DriverEvent *event);

/// Blocks the host until `event` has fired, i.e. until the work recorded
/// before it on its stream has completed.
///
/// Mirrors `max._core.driver.DeviceEvent.synchronize`.
M_DRIVER_API M_DriverStatus
M_DriverEventSynchronize(const M_DriverEvent *event);

/// Writes whether `event` has already fired to `*out`. Never blocks.
///
/// Mirrors `max._core.driver.DeviceEvent.is_ready`.
M_DRIVER_API M_DriverStatus M_DriverEventIsReady(const M_DriverEvent *event,
                                                 bool *out);

/// Writes the milliseconds elapsed between `start` and `end` to `*out_ms`.
///
/// Both events must have been created with `enable_timing` and both must have
/// fired; this is the throughput measurement primitive. The result is the
/// vendor's signed difference of the two timestamps, so an `end` recorded
/// before `start` gives a negative value rather than a duration.
///
/// Mirrors `max._core.driver.DeviceEvent.elapsed_time`.
M_DRIVER_API M_DriverStatus M_DriverEventElapsedTimeMs(
    const M_DriverEvent *start, const M_DriverEvent *end, double *out_ms);

//===----------------------------------------------------------------------===//
// Buffers, host memory registration, and copies
//===----------------------------------------------------------------------===//

/// A range of memory that the driver can address: either allocated by the
/// driver, or wrapping memory the caller owns. Create with
/// `M_DriverBufferAllocate` or `M_DriverBufferFromPointer`; destroy with
/// `M_DriverBufferDestroy`.
///
/// Memory comes from the device's memory manager, which MAX and this interface
/// share. The handle does not order the memory against the streams that copy
/// through it, because the copies in this interface take raw addresses and
/// leave no record on the buffer. Two rules follow. Before a stream's first
/// use of a buffer, the caller orders the stream after the device with
/// `M_DriverStreamWaitForDevice`, because the memory manager can hand this
/// interface a block MAX released on its own stream while MAX's last use of
/// it is still in flight. Before destroying a buffer, the caller synchronizes
/// every stream that copied through it. A buffer wrapping caller memory never
/// frees that memory.

/// Allocates `byte_size` bytes on `device`.
///
/// With `pinned_host` false the memory is device memory; with `pinned_host`
/// true it is host memory that the device can DMA to and from directly, which
/// is the destination to use for asynchronous device-to-host copies. The
/// allocation is uninitialized, and comes from the device's memory manager, so
/// it is subject to the same budget and fragmentation behaviour as every other
/// device allocation in the process.
///
/// Mirrors `max._core.driver.Buffer(..., device=..., usage=Usage.STAGING)`
/// for `pinned_host`, and the plain device allocation otherwise.
M_DRIVER_API M_DriverStatus M_DriverBufferAllocate(const M_DriverDevice *device,
                                                   uint64_t byte_size,
                                                   bool pinned_host,
                                                   M_DriverBuffer **out);

/// Wraps memory the caller already owns and keeps alive, for use as a copy
/// source or destination, without copying or freeing it.
///
/// `is_host` selects the memory space: true for CPU-addressable memory (a
/// pinned host slab, or a host pointer the caller pinned separately), false
/// for memory already allocated on `device` -- for example a tensor MAX owns,
/// whose address the caller got from MAX.
///
/// The caller must keep the range valid, and must not free or reuse it, until
/// every copy out of it has been ordered and completed; the returned handle
/// does not extend the memory's lifetime.
///
/// Mirrors the borrowed-buffer path of `max._core.driver.Buffer._from_dlpack`.
M_DRIVER_API M_DriverStatus M_DriverBufferFromPointer(
    const M_DriverDevice *device, void *pointer, uint64_t byte_size,
    bool is_host, M_DriverBuffer **out);

/// Releases a buffer handle. A null handle is `M_DRIVER_INVALID_ARGUMENT`.
/// Memory the driver allocated returns to the device at once, so the caller
/// synchronizes every stream that copied through it first; memory the caller
/// owns is left untouched.
M_DRIVER_API M_DriverStatus M_DriverBufferDestroy(M_DriverBuffer *buffer);

/// Writes the address to use in copies with this buffer to `*out`. For device
/// memory it is an address on the device, which the host must not read
/// through; for host memory it is the host address.
///
/// For a wrapped range this may differ from the pointer passed to
/// `M_DriverBufferFromPointer`: a backend is free to allocate a device-visible
/// alias for host memory.
M_DRIVER_API M_DriverStatus M_DriverBufferPointer(const M_DriverBuffer *buffer,
                                                  void **out);

/// Writes the buffer's size in bytes to `*out`.
M_DRIVER_API M_DriverStatus M_DriverBufferByteSize(const M_DriverBuffer *buffer,
                                                   uint64_t *out);

/// Writes whether the buffer is host memory (as opposed to device memory) to
/// `*out`.
M_DRIVER_API M_DriverStatus M_DriverBufferIsHost(const M_DriverBuffer *buffer,
                                                 bool *out);

/// A host range the caller owns that has been made device-accessible for DMA,
/// for example a `MAP_SHARED` mapping of a memory file descriptor that a
/// remote peer also maps. Unregister with `M_DriverMappingDestroy`.
///
/// The driver grants access, not ownership: the caller keeps the pages, and
/// must unmap them only after every context that touched the range has been
/// synchronized.

/// Makes the host range [`pointer`, `pointer` + `byte_size`) DMA-accessible to
/// `device`, without allocating or copying it, and writes the resulting
/// mapping to `*out`.
///
/// This is the registration a transport needs before a peer can read a
/// locally mapped slab (CUDA: `cuMemHostRegister` with
/// `CU_MEMHOSTREGISTER_PORTABLE`, so one registration serves every context in
/// the process). Registration is a driver operation on a page-granular
/// resource: it can fail, it has a per-process budget, and the range
/// constraints are backend specific (Metal, for example, aliases whole VM
/// pages, so both the base and the size must be page multiples).
///
/// For addressability by device code, use `M_DriverMappingPointer`, which may
/// differ from `pointer`.
M_DRIVER_API M_DriverStatus M_DriverMemoryRegister(const M_DriverDevice *device,
                                                   void *pointer,
                                                   uint64_t byte_size,
                                                   M_DriverMapping **out);

/// Queues the unregister of a range registered by `M_DriverMemoryRegister`
/// and releases the handle. A null handle is `M_DRIVER_INVALID_ARGUMENT`.
///
/// Nothing orders the unregister behind streams that copied through the
/// range: synchronize them first, and unmap the pages after. The unregister
/// is queued; `M_DriverDeviceSynchronize` runs it and reports a refused one.
/// To register the same pages again at once, use
/// `M_DriverMappingDestroySync`.
M_DRIVER_API M_DriverStatus M_DriverMappingDestroy(M_DriverMapping *mapping);

/// Unregisters the range on the calling thread and releases the handle,
/// returning the vendor's unregister status, where `M_DriverMappingDestroy`
/// only queues the unregister. This is the call to make before unmapping the
/// pages or registering the same address again. Nothing orders it behind the
/// streams that copied through the range, so the caller synchronizes every
/// one of them first. The handle is released whether or not the unregister
/// succeeded. Unregisters `M_DriverMappingDestroy` queued earlier on the same
/// device run first and share the status; if one of them fails, this call's
/// own unregister stays queued for the next `M_DriverDeviceSynchronize`. A
/// null handle is `M_DRIVER_INVALID_ARGUMENT`.
M_DRIVER_API M_DriverStatus
M_DriverMappingDestroySync(M_DriverMapping *mapping);

/// Writes the device-accessible address of a registered range to `*out`. Use
/// this, not the pointer passed to `M_DriverMemoryRegister`, as a copy
/// destination or source.
M_DRIVER_API M_DriverStatus
M_DriverMappingPointer(const M_DriverMapping *mapping, void **out);

/// Writes the size in bytes of a registered range to `*out`.
M_DRIVER_API M_DriverStatus
M_DriverMappingByteSize(const M_DriverMapping *mapping, uint64_t *out);

/// Enqueues a host-to-device copy of `byte_size` bytes from `src_host` to
/// `dst_device`, on `stream`, and returns without waiting for it.
///
/// Addresses are raw: `dst_device` is a device address MAX or this interface
/// produced (`M_DriverBufferPointer`, a tensor, or a registered mapping), and
/// `src_host` is a host address (a mapped slab, or caller memory pinned with
/// `M_DriverMemoryRegister`). The copy runs on `stream`'s context, so it is
/// ordered against everything else enqueued there, and it is in place: the
/// destination must already be sized.
///
/// Neither range may overlap the other, and both must stay valid until the
/// copy has completed -- synchronize `stream`, or wait on an event recorded
/// after this call, before releasing or reusing either.
///
/// Mirrors `cuMemcpyHtoDAsync` / `Buffer.inplace_copy_from` on a stream.
M_DRIVER_API M_DriverStatus
M_DriverCopyHostToDevice(const M_DriverStream *stream, void *dst_device,
                         const void *src_host, uint64_t byte_size);

/// Enqueues a device-to-host copy of `byte_size` bytes from `src_device` to
/// `dst_host`, on `stream`, and returns without waiting for it.
///
/// The host destination must be DMA-accessible to the device
/// (`M_DriverMemoryRegister`, or a pinned buffer); a pageable host pointer
/// either fails or is staged through a bounce buffer, which the caller cannot
/// observe, so pin first.
///
/// Mirrors `cuMemcpyDtoHAsync`.
M_DRIVER_API M_DriverStatus
M_DriverCopyDeviceToHost(const M_DriverStream *stream, void *dst_host,
                         const void *src_device, uint64_t byte_size);

/// The direction every piece of an `M_DriverCopyBatch` shares.
typedef enum M_DriverCopyDirection {
  /// `dsts` are device addresses and `srcs` are host addresses.
  M_DRIVER_COPY_HOST_TO_DEVICE = 0,
  /// `dsts` are host addresses and `srcs` are device addresses.
  M_DRIVER_COPY_DEVICE_TO_HOST = 1,
} M_DriverCopyDirection;

/// Enqueues `count` copies in one direction, `sizes[i]` bytes from `srcs[i]`
/// to `dsts[i]`, on `stream`, and returns without waiting for them.
///
/// One vendor submission where the driver has a batched copy (CUDA 12.8+ and
/// recent HIP), otherwise the equivalent sequence of scalar copies. Each piece
/// has the requirements of `M_DriverCopyHostToDevice` / `DeviceToHost`: the
/// device address belongs to `stream`'s device, the host address is DMA
/// accessible, the two ranges do not overlap, and both stay valid until the
/// batch completes. Nothing is recorded on or awaited on any other stream.
///
/// A `count` of 0 succeeds. A null address, zero size, null array with a non
/// zero count, or unknown direction is `M_DRIVER_INVALID_ARGUMENT`, and
/// nothing is enqueued.
M_DRIVER_API M_DriverStatus M_DriverCopyBatch(const M_DriverStream *stream,
                                              M_DriverCopyDirection direction,
                                              void *const *dsts,
                                              const void *const *srcs,
                                              const uint64_t *sizes,
                                              size_t count);

/// Enqueues a device-to-device copy of `byte_size` bytes from `src_device` to
/// `dst_device`, on `stream`, and returns without waiting for it.
///
/// Both addresses must belong to `stream`'s own device and context; this is the
/// same-device path, which the driver issues as a direct device copy.
/// Cross-device copies use `M_DriverCopyPeerToPeer`.
///
/// Mirrors `Buffer.inplace_copy_from` and `cuMemcpyDtoDAsync`.
M_DRIVER_API M_DriverStatus
M_DriverCopyDeviceToDevice(const M_DriverStream *stream, void *dst_device,
                           const void *src_device, uint64_t byte_size);

/// Enqueues a copy of `byte_size` bytes from `src_device`+`src` to
/// `dst_device`+`dst`, on `stream`, and returns without waiting for it.
///
/// This is the cross-device path, issued as a peer copy (`cuMemcpyPeer` on
/// CUDA, the xGMI equivalent on AMD). It is direct only between devices that
/// can access each other (`M_DriverDeviceCanAccess`) with access enabled by
/// `M_DriverDeviceEnablePeerAccess`; on CUDA a pair without access still goes
/// through the peer copy, which the vendor then stages through host memory.
/// `M_DriverEnableAllPeerAccess` enables every pair at once but fails as a
/// whole on a host where any pair cannot reach each other.
///
/// The copy is enqueued on `stream`, which must run on the device that owns
/// `dst`, and the driver brackets it against the source device's shared
/// context, so `stream` first waits for everything enqueued there and that
/// context then waits for the copy before it runs anything else. Passing
/// `dst_device` explicitly lets the caller state the destination device, and
/// it is checked.
///
/// Mirrors `cuMemcpyPeerAsync` and the cross-device broadcast copy.
M_DRIVER_API M_DriverStatus M_DriverCopyPeerToPeer(
    const M_DriverStream *stream, const M_DriverDevice *dst_device, void *dst,
    const M_DriverDevice *src_device, const void *src, uint64_t byte_size);

/// Like `M_DriverCopyPeerToPeer`, except that the driver records nothing on
/// and waits for nothing on the source device's shared context.
/// `M_DriverCopyPeerToPeer` brackets the copy so `stream` waits for that
/// context's work and the context then waits for the copy, which also makes
/// MAX's compute stream on the source device stall behind the copy. This
/// variant leaves the ordering to the caller, who records an event on the
/// stream that produced `src` and waits for it on `stream` before this call,
/// and keeps `src` alive until an event recorded on `stream` after this call
/// has fired. Because it touches nothing on the source device's shared
/// context, it also neither reports nor clears an error MAX left pending
/// there, where the ordered variant does (see the Devices section).
M_DRIVER_API M_DriverStatus M_DriverCopyPeerToPeerUnordered(
    const M_DriverStream *stream, const M_DriverDevice *dst_device, void *dst,
    const M_DriverDevice *src_device, const void *src, uint64_t byte_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLRT_DRIVER_CAPI_DRIVERCAPI_H
