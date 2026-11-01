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
// Implementation of the driver C interface declared in
// MLRT/Driver/CAPI/DriverCAPI.h.
//
// This file is a translation layer and nothing else: it validates its
// arguments, forwards to `M::Driver::*`, and maps `ErrorOr` back to a status.
// All state lives in the driver, so a handle here is just a counted reference
// to a driver object -- which is what makes a caller's copies and events
// ordered against MAX's own work on the same device.
//
// Nothing here throws: the library is built without exceptions, so every
// failure is a status, and every argument is checked before it is used.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/CAPI/DriverCAPI.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"

#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace {

using M::ErrorOr;
using M::Driver::Device;
using M::Driver::DeviceBuffer;
using M::Driver::DeviceBufferRef;
using M::Driver::DeviceContext;
using M::Driver::DeviceContextRef;
using M::Driver::DeviceEvent;
using M::Driver::DeviceEventRef;
using M::Driver::DeviceStreamRef;

/// The message of the most recent failure on this thread. Thread-local, so
/// concurrent callers cannot read each other's errors; cleared by the next
/// successful call.
///
/// The string is leaked on purpose: a namespace-scope `std::string` would
/// need a global destructor, which this library is built without
/// (-Wglobal-constructors), and thread teardown can outlive the driver.
std::string &lastErrorMessage() {
  static thread_local std::string *message = new std::string();
  return *message;
}

M_DriverStatus succeed() {
  lastErrorMessage().clear();
  return M_DRIVER_OK;
}

M_DriverStatus fail(M_DriverStatus status, const std::string &message) {
  lastErrorMessage() = message;
  return status;
}

/// Reports `what` plus the driver's own message for a failure.
M_DriverStatus fail(const char *what, const std::string &detail,
                    M_DriverStatus status = M_DRIVER_RUNTIME_ERROR) {
  std::string message(what);
  message += " failed: ";
  message += detail;
  return fail(status, message);
}

/// Reports `what` plus the message carried by a failed `ErrorOr`.
template <class T>
M_DriverStatus fail(const ErrorOr<T> &err, const char *what,
                    M_DriverStatus status = M_DRIVER_RUNTIME_ERROR) {
  return fail(what, err.getError(), status);
}

/// Copies `value` into the caller's buffer, NUL-terminated, and returns the
/// bytes needed including the NUL terminator. See `M_DriverLastErrorMessage`.
size_t copyOut(const std::string &value, char *buffer, size_t capacity) {
  const size_t needed = value.size() + 1;
  if (buffer != nullptr && capacity >= needed) {
    std::memcpy(buffer, value.data(), value.size());
    buffer[value.size()] = '\0';
  }
  return needed;
}

/// Copies a device string out, reporting a failure through the error message
/// and a 0 return (the convention for the `size_t`-returning queries).
size_t copyOutDeviceString(const char *what, const std::string &value,
                           char *buffer, size_t capacity) {
  if (value.empty()) {
    fail(what, "the device has no name", M_DRIVER_RUNTIME_ERROR);
    return 0;
  }
  lastErrorMessage().clear();
  return copyOut(value, buffer, capacity);
}

M_DriverStatus notSupportedForExternalStream(const char *what) {
  std::string message(what);
  message +=
      ": the stream was created from a native handle, so it can only order "
      "work; enqueue copies on a stream created by M_DriverStreamCreate or "
      "adopted with M_DriverStreamFromContextHandle";
  return fail(M_DRIVER_NOT_SUPPORTED, message);
}

/// Reports `what` when the driver rejected an operation.
M_DriverStatus reportIfError(const M::ErrorOrSuccess &err, const char *what) {
  if (err.isError())
    return fail(what, err.getError());
  return succeed();
}

} // namespace

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

uint32_t M_DriverCApiVersion(void) {
  return (static_cast<uint32_t>(M_DRIVER_CAPI_VERSION_MAJOR) << 24) |
         (static_cast<uint32_t>(M_DRIVER_CAPI_VERSION_MINOR) << 16) |
         static_cast<uint32_t>(M_DRIVER_CAPI_VERSION_PATCH);
}

//===----------------------------------------------------------------------===//
// Handles
//===----------------------------------------------------------------------===//
//
// The driver owns every object; these structs hold a counted reference so the
// object outlives the handle even if MAX tears its own state down first.

struct M_DriverDevice {
  /// The process-wide context for this (api, id) pair. Shared with MAX and
  /// with every other handle for the same device, so memory allocated through
  /// either is addressable through both.
  M::Driver::DeviceContextRef ctx;
};

struct M_DriverStream {
  /// The context that executes this stream's work. Its active queue is the
  /// stream.
  M::Driver::DeviceContextRef ctx;
  /// Set when the stream adopts a vendor-native handle; empty otherwise.
  M::Driver::DeviceStreamRef external;
  /// The process-wide context of the device the stream runs on, which is
  /// what a device handle made from this stream names.
  M::Driver::DeviceContextRef device;
};

struct M_DriverEvent {
  M::Driver::DeviceEventRef event;
};

struct M_DriverBuffer {
  M::Driver::DeviceBufferRef buffer;
};

namespace {

/// A context of this interface's own, for calls that order nothing against
/// MAX, so their pending errors and queued unregisters never touch MAX's.
struct Service {
  DeviceContextRef ctx;
  /// Serializes every poll of `ctx`, so a drop on one thread cannot run the
  /// unregister another thread's synchronize is waiting on.
  std::mutex pollMutex;
};

} // namespace

struct M_DriverMapping {
  M::Driver::DeviceBufferRef buffer;
  Service *service;
};

namespace {

/// The context a device handle names.
DeviceContext &contextOf(const M_DriverDevice &device) { return *device.ctx; }

/// Never freed, so a mapping that outlives its device handle keeps its lock.
ErrorOr<Service *> serviceFor(DeviceContext &shared, bool create = true) {
  static std::mutex mutex;
  static std::map<DeviceContext *, Service *> services;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = services.find(&shared);
  if (found != services.end())
    return found->second;
  if (!create)
    return static_cast<Service *>(nullptr);
  ErrorOr<DeviceContextRef> created = shared.device().createContext();
  if (created.isError())
    return M::Error(created.getError());
  auto *service = new Service{created.takeValue(), {}};
  services.emplace(&shared, service);
  return service;
}

/// True when a stream adopts a vendor-native handle rather than a driver
/// context; such a stream supports ordering operations only.
bool isExternal(const M_DriverStream &stream) {
  return stream.external.getPointer() != nullptr;
}

/// True when both handles name the same device (same api and ordinal).
bool sameDevice(const M_DriverDevice &a, const M_DriverDevice &b) {
  return a.ctx->device().api().str() == b.ctx->device().api().str() &&
         a.ctx->device().id() == b.ctx->device().id();
}

/// True when both objects belong to the same backend. Every backend casts the
/// objects it is handed to its own types without checking, so a mix is
/// refused here, where it can be a status, rather than misread there.
bool sameApi(const DeviceContext &a, const DeviceContext &b) {
  return a.device().api().str() == b.device().api().str();
}

bool sameApi(const DeviceContext &context, const DeviceEvent &event) {
  return context.device().api().str() == event.device().api().str();
}

bool sameApi(const DeviceEvent &a, const DeviceEvent &b) {
  return a.device().api().str() == b.device().api().str();
}

M_DriverStatus mixedBackends(const char *what) {
  std::string message(what);
  message += ": the handles belong to different backends";
  return fail(M_DRIVER_INVALID_ARGUMENT, message);
}

} // namespace

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

size_t M_DriverLastErrorMessage(char *buffer, size_t capacity) {
  if (lastErrorMessage().empty())
    return 0;
  return copyOut(lastErrorMessage(), buffer, capacity);
}

//===----------------------------------------------------------------------===//
// Devices
//===----------------------------------------------------------------------===//

int64_t M_DriverAcceleratorCount(void) {
  // A machine without a driver reports zero rather than failing: `Device::
  // numberOfDevices` resolves through the backend table and returns 0 when
  // nothing can serve the query.
  return Device::numberOfDevices(Device::gpuAPI);
}

M_DriverStatus M_DriverAcceleratorDeviceCreate(int64_t ordinal,
                                               M_DriverDevice **out) {
  if (out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "out is null");
  *out = nullptr;
  // Checked against the count first, because the driver reports an ordinal
  // it cannot create as a runtime error, which reads as transient to a caller
  // that retries those.
  const int64_t count = M_DriverAcceleratorCount();
  if (ordinal < 0 || ordinal >= count)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "ordinal is out of range: expected 0 <= ordinal < " +
                    std::to_string(count));
  ErrorOr<DeviceContextRef> ctx =
      DeviceContext::getOrCreate(Device::gpuAPI, static_cast<int>(ordinal));
  if (ctx.isError())
    return fail(ctx, "M_DriverAcceleratorDeviceCreate");
  *out = new M_DriverDevice{ctx.takeValue()};
  return succeed();
}

M_DriverStatus M_DriverCPUDeviceCreate(int64_t id, M_DriverDevice **out) {
  if (out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "out is null");
  *out = nullptr;
  // There is exactly one CPU device, which `CpuDevice::get` serves for id 0,
  // and -1 names it as the default; nothing else is a CPU device.
  if (id < -1 || id > 0)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "id is out of range: the CPU device accepts 0 or -1");
  ErrorOr<DeviceContextRef> ctx =
      DeviceContext::getOrCreate(Device::cpuAPI, /*id=*/0);
  if (ctx.isError())
    return fail(ctx, "M_DriverCPUDeviceCreate");
  *out = new M_DriverDevice{ctx.takeValue()};
  return succeed();
}

M_DriverStatus M_DriverDeviceDestroy(M_DriverDevice *device) {
  if (device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
  delete device;
  return succeed();
}

M_DriverStatus M_DriverDeviceId(const M_DriverDevice *device, uint64_t *out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
  *out = static_cast<uint64_t>(contextOf(*device).device().id());
  return succeed();
}

size_t M_DriverDeviceApi(const M_DriverDevice *device, char *buffer,
                         size_t capacity) {
  if (device == nullptr) {
    fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
    return 0;
  }
  const std::string api = contextOf(*device).device().api().str();
  lastErrorMessage().clear();
  return copyOut(api, buffer, capacity);
}

size_t M_DriverDeviceArchitectureName(const M_DriverDevice *device,
                                      char *buffer, size_t capacity) {
  if (device == nullptr) {
    fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
    return 0;
  }
  ErrorOr<const char *> name = contextOf(*device).device().getArchName();
  if (name.isError()) {
    fail(name, "M_DriverDeviceArchitectureName");
    return 0;
  }
  return copyOutDeviceString("M_DriverDeviceArchitectureName", name.get(),
                             buffer, capacity);
}

size_t M_DriverDeviceModelName(const M_DriverDevice *device, char *buffer,
                               size_t capacity) {
  if (device == nullptr) {
    fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
    return 0;
  }
  return copyOutDeviceString("M_DriverDeviceModelName",
                             contextOf(*device).device().name(), buffer,
                             capacity);
}

size_t M_DriverDevicePciBusId(const M_DriverDevice *device, char *buffer,
                              size_t capacity) {
  if (device == nullptr) {
    fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
    return 0;
  }
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL shim has no PCI bus id query, so the caller reads the reason and
  // falls back to another placement hint.
  (void)buffer;
  (void)capacity;
  fail(M_DRIVER_NOT_SUPPORTED, "M_DriverDevicePciBusId is not available in a "
                               "HAL device context build");
  return 0;
#else
  ErrorOr<std::string> busId = contextOf(*device).device().getPciBusId();
  if (busId.isError()) {
    fail(busId, "M_DriverDevicePciBusId");
    return 0;
  }
  return copyOutDeviceString("M_DriverDevicePciBusId", busId.get(), buffer,
                             capacity);
#endif
}

M_DriverStatus M_DriverDeviceUuid(const M_DriverDevice *device,
                                  uint8_t out_uuid[16]) {
  if (device == nullptr || out_uuid == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out_uuid is null");
#if MODULAR_HAL_DEVICE_CONTEXT
  return fail(M_DRIVER_NOT_SUPPORTED,
              "M_DriverDeviceUuid is not available in a HAL device context "
              "build");
#else
  ErrorOr<std::optional<std::array<uint8_t, 16>>> uuid =
      contextOf(*device).device().getUuid();
  if (uuid.isError())
    return fail(uuid, "M_DriverDeviceUuid");
  // An empty answer is the backend lacking a UUID rather than a driver fault,
  // so the caller learns to key the device another way.
  if (!uuid.get().has_value())
    return fail(M_DRIVER_NOT_SUPPORTED,
                "M_DriverDeviceUuid: this device has no UUID to report");
  std::memcpy(out_uuid, uuid.get()->data(), uuid.get()->size());
  return succeed();
#endif
}

M_DriverStatus M_DriverDeviceEnablePeerAccess(const M_DriverDevice *device,
                                              const M_DriverDevice *peer) {
  if (device == nullptr || peer == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or peer is null");
  if (!sameApi(contextOf(*device), contextOf(*peer)))
    return mixedBackends("M_DriverDeviceEnablePeerAccess");
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL shim enables peer access for every pair at once and has no per
  // pair entry point, so callers use M_DriverEnableAllPeerAccess there.
  return fail(M_DRIVER_NOT_SUPPORTED,
              "M_DriverDeviceEnablePeerAccess is not available in a HAL "
              "device context build");
#else
  return reportIfError(
      contextOf(*device).device().enablePeerAccess(contextOf(*peer).device()),
      "M_DriverDeviceEnablePeerAccess");
#endif
}

M_DriverStatus M_DriverDeviceCanAccess(const M_DriverDevice *device,
                                       const M_DriverDevice *peer, bool *out) {
  if (device == nullptr || peer == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device, peer or out is null");
  if (!sameApi(contextOf(*device), contextOf(*peer)))
    return mixedBackends("M_DriverDeviceCanAccess");
  ErrorOr<bool> canAccess =
      contextOf(*device).device().canAccess(contextOf(*peer).device());
  if (canAccess.isError())
    return fail(canAccess, "M_DriverDeviceCanAccess");
  *out = canAccess.get();
  return succeed();
}

M_DriverStatus M_DriverDeviceSynchronize(const M_DriverDevice *device) {
  if (device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
  M_DriverStatus status = reportIfError(contextOf(*device).synchronize(),
                                        "M_DriverDeviceSynchronize");
  if (status != M_DRIVER_OK)
    return status;
  // Flushes the unregisters `M_DriverMappingDestroy` queued, as the header
  // promises.
  ErrorOr<Service *> service = serviceFor(contextOf(*device), /*create=*/false);
  if (service.isError() || *service == nullptr)
    return status;
  std::lock_guard<std::mutex> lock((*service)->pollMutex);
  return reportIfError((*service)->ctx->synchronize(),
                       "M_DriverDeviceSynchronize");
}

M_DriverStatus M_DriverDeviceMemoryInfo(const M_DriverDevice *device,
                                        uint64_t *out_total,
                                        uint64_t *out_free) {
  if (device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
  ErrorOr<Device::MemInfo> info = contextOf(*device).device().getMemoryInfo();
  if (info.isError())
    return fail(info, "M_DriverDeviceMemoryInfo");
  if (out_total != nullptr)
    *out_total = static_cast<uint64_t>(info.get().total);
  if (out_free != nullptr) {
    // A driver that cannot measure free memory leaves it unset.
    const std::optional<size_t> free = info.get().free;
    *out_free = free.has_value() ? static_cast<uint64_t>(*free)
                                 : M_DRIVER_MEMORY_UNKNOWN;
  }
  return succeed();
}

M_DriverStatus M_DriverDeviceAllocationRange(const M_DriverDevice *device,
                                             const void *address,
                                             void **out_base,
                                             uint64_t *out_size) {
  if (device == nullptr || out_base == nullptr || out_size == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "device, out_base or out_size is null");
  if (address == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "address is null");
  ErrorOr<Device::AllocationRange> range =
      contextOf(*device).device().allocationRange(const_cast<void *>(address));
  if (range.isError())
    return fail(range, "M_DriverDeviceAllocationRange");
  *out_base = range.get().base;
  *out_size = static_cast<uint64_t>(range.get().size);
  return succeed();
}

M_DriverStatus M_DriverEnableAllPeerAccess(void) {
  return reportIfError(Device::enableAllPeerAccess(),
                       "M_DriverEnableAllPeerAccess");
}

M_DriverStatus M_DriverAllPeerAccessEnabled(bool *out) {
  if (out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "out is null");
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL shim keeps no record of which pairs are enabled.
  *out = false;
  return fail(M_DRIVER_NOT_SUPPORTED,
              "M_DriverAllPeerAccessEnabled is not available in a HAL device "
              "context build");
#else
  ErrorOr<bool> enabled = Device::allPeerAccessEnabled();
  if (enabled.isError())
    return fail(enabled, "M_DriverAllPeerAccessEnabled");
  *out = enabled.get();
  return succeed();
#endif
}

M_DriverStatus M_DriverDeviceContextHandle(const M_DriverDevice *device,
                                           void **out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL context is a box owned by the Mojo side, with no stable address a
  // caller could hand back; identity has to come from the HAL itself.
  return fail(M_DRIVER_NOT_SUPPORTED,
              "M_DriverDeviceContextHandle is not available in a "
              "HAL device context build");
#else
  *out = device->ctx.getPointer();
  return succeed();
#endif
}

M_DriverStatus M_DriverDeviceBindToCurrentThread(const M_DriverDevice *device) {
  if (device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
  return reportIfError(contextOf(*device).bindToCurrentThread(),
                       "M_DriverDeviceBindToCurrentThread");
}

void M_DriverUnbindFromCurrentThread(void) {
  DeviceContext::unbindFromCurrentThread();
}

//===----------------------------------------------------------------------===//
// Streams
//===----------------------------------------------------------------------===//

M_DriverStatus M_DriverStreamCreate(const M_DriverDevice *device,
                                    M_DriverStream **out) {
  if (out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "out is null");
  // Cleared before any other check, so a refused call never leaves a stale
  // value a caller could mistake for a stream.
  *out = nullptr;
  if (device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device is null");
  // A fresh context, on the device's shared memory: copies through it are
  // its own queue's, but they land in the same allocator MAX uses.
  ErrorOr<DeviceContextRef> ctx = contextOf(*device).device().createContext();
  if (ctx.isError())
    return fail(ctx, "M_DriverStreamCreate");
  DeviceContextRef streamCtx = ctx.takeValue();
#if !MODULAR_HAL_DEVICE_CONTEXT
  // A CUDA/HIP context's base stream is blocking, but the pool streams
  // createStream adds are not, so take a view of a fresh one. The context is
  // private to this call, so the pool index cannot race. The HAL shim keeps
  // its base queue, as it did before every stream became non blocking.
  const Device::API api = streamCtx->device().api();
  if (api == Device::cudaAPI || api == Device::hipAPI) {
    ErrorOr<DeviceStreamRef> pooled = streamCtx->createStream();
    if (pooled.isError())
      return fail(pooled, "M_DriverStreamCreate");
    ErrorOr<DeviceContextRef> view =
        streamCtx->selectStream(streamCtx->numStreams() - 1);
    if (view.isError())
      return fail(view, "M_DriverStreamCreate");
    streamCtx = view.takeValue();
  }
#endif
  *out = new M_DriverStream{std::move(streamCtx), DeviceStreamRef(),
                            DeviceContextRef::copy(device->ctx.getPointer())};
  return succeed();
}

M_DriverStatus M_DriverStreamFromContextHandle(void *context_handle,
                                               M_DriverStream **out) {
  if (out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "out is null");
  *out = nullptr;
  if (context_handle == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "context_handle is null");
#if MODULAR_HAL_DEVICE_CONTEXT
  return fail(M_DRIVER_NOT_SUPPORTED,
              "M_DriverStreamFromContextHandle is not available in a "
              "HAL device context build");
#else
  auto *ctx = static_cast<DeviceContext *>(context_handle);
  // The adopted context may be a stream's private one, so the device handle a
  // caller later asks this stream for comes from the process-wide context of
  // its device, the one every other device handle names.
  ErrorOr<DeviceContextRef> shared =
      DeviceContext::getOrCreate(ctx->device().api(), ctx->device().id());
  if (shared.isError())
    return fail(shared, "M_DriverStreamFromContextHandle");
  *out = new M_DriverStream{DeviceContextRef::copy(ctx), DeviceStreamRef(),
                            shared.takeValue()};
  return succeed();
#endif
}

M_DriverStatus M_DriverStreamFromNativeHandle(const M_DriverDevice *device,
                                              void *native_handle,
                                              M_DriverStream **out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
  *out = nullptr;
  if (native_handle == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "native_handle is null");
  ErrorOr<DeviceContextRef> ctx = contextOf(*device).device().createContext();
  if (ctx.isError())
    return fail(ctx, "M_DriverStreamFromNativeHandle");
  DeviceContextRef context = ctx.takeValue();
  ErrorOr<DeviceStreamRef> stream =
      context->createExternalStream(native_handle);
  if (stream.isError())
    return fail(stream, "M_DriverStreamFromNativeHandle");
  *out = new M_DriverStream{std::move(context), stream.takeValue(),
                            DeviceContextRef::copy(device->ctx.getPointer())};
  return succeed();
}

M_DriverStatus M_DriverStreamDestroy(M_DriverStream *stream) {
  if (stream == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream is null");
  delete stream;
  return succeed();
}

M_DriverStatus M_DriverStreamNativeHandle(const M_DriverStream *stream,
                                          void **out) {
  if (stream == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or out is null");
  *out = nullptr;
  if (isExternal(*stream)) {
    *out = stream->external->ptr();
    return succeed();
  }
  ErrorOr<DeviceStreamRef> active = stream->ctx->stream();
  if (active.isError())
    return fail(active, "M_DriverStreamNativeHandle");
  *out = active.get()->ptr();
  return succeed();
}

M_DriverStatus M_DriverStreamDeviceContextHandle(const M_DriverStream *stream,
                                                 void **out) {
  if (stream == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or out is null");
  // The context created beside a native queue is not ordered against that
  // queue, so adopting it would give a stream whose copies run out of order.
  if (isExternal(*stream))
    return fail(M_DRIVER_NOT_SUPPORTED,
                "M_DriverStreamDeviceContextHandle: the stream was created "
                "from a native handle, and the context beside that queue is "
                "not ordered against it");
  M_DriverDevice device{DeviceContextRef::copy(stream->ctx.getPointer())};
  return M_DriverDeviceContextHandle(&device, out);
}

M_DriverStatus M_DriverStreamDevice(const M_DriverStream *stream,
                                    M_DriverDevice **out) {
  if (stream == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or out is null");
  *out =
      new M_DriverDevice{DeviceContextRef::copy(stream->device.getPointer())};
  return succeed();
}

M_DriverStatus M_DriverStreamSynchronize(const M_DriverStream *stream) {
  if (stream == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream is null");
  if (isExternal(*stream))
    return reportIfError(stream->external->synchronize(),
                         "M_DriverStreamSynchronize");
  return reportIfError(stream->ctx->synchronize(), "M_DriverStreamSynchronize");
}

/// Waits on `event` at `stream`, whichever kind of queue `stream` is.
M_DriverStatus waitOnEvent(const M_DriverStream &stream,
                           const DeviceEventRef &event, const char *what) {
  if (isExternal(stream))
    return reportIfError(stream.external->enqueueWaitForEvent(*event), what);
  // `enqueueWaitFor` predates const correctness and takes a mutable event; the
  // wait itself does not modify it.
  return reportIfError(
      stream.ctx->enqueueWaitFor(const_cast<DeviceEvent &>(*event)), what);
}

/// Orders `stream` after everything enqueued on `context`'s queue.
///
/// A vendor-native queue is not one of our contexts, so there is no context of
/// ours to wait *for*: the source has to record an event on itself and the
/// destination wait on that. This is the same shape the driver uses internally
/// for `enqueueWaitFor(DeviceContext&)`.
M_DriverStatus orderAfterContext(M_DriverStream &stream, DeviceContext &context,
                                 const char *what) {
  ErrorOr<DeviceEventRef> recorded = context.enqueueEvent();
  if (recorded.isError())
    return fail(recorded, what);
  DeviceEventRef event = recorded.takeValue();
  return waitOnEvent(stream, event, what);
}

/// Orders `stream` after everything enqueued on `other`, or after `other`'s own
/// queue when it has one that is not a context of ours.
///
/// A self-wait reaches the driver's own no-op
/// (`CpuDeviceContext::enqueueWaitFor` and its Metal counterpart return
/// immediately for their own context), so nothing here has to special-case it.
M_DriverStatus orderAfterStream(M_DriverStream &stream, M_DriverStream &other,
                                const char *what) {
  if (!isExternal(stream) && !isExternal(other))
    return reportIfError(stream.ctx->enqueueWaitFor(*other.ctx), what);
  if (isExternal(other)) {
    // The source is the vendor-native queue: record there, not on the context
    // this interface created alongside it.
    ErrorOr<DeviceEventRef> recorded = other.external->enqueueEvent();
    if (recorded.isError())
      return fail(recorded, what);
    DeviceEventRef event = recorded.takeValue();
    return waitOnEvent(stream, event, what);
  }
  return orderAfterContext(stream, *other.ctx, what);
}

M_DriverStatus M_DriverStreamWaitFor(const M_DriverStream *stream,
                                     const M_DriverStream *other) {
  if (stream == nullptr || other == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or other is null");
  if (!sameApi(*stream->ctx, *other->ctx))
    return mixedBackends("M_DriverStreamWaitFor");
  return orderAfterStream(*const_cast<M_DriverStream *>(stream),
                          *const_cast<M_DriverStream *>(other),
                          "M_DriverStreamWaitFor");
}

M_DriverStatus M_DriverStreamWaitForDevice(const M_DriverStream *stream,
                                           const M_DriverDevice *device) {
  if (stream == nullptr || device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or device is null");
  if (!sameApi(*stream->ctx, contextOf(*device)))
    return mixedBackends("M_DriverStreamWaitForDevice");
  return orderAfterContext(*const_cast<M_DriverStream *>(stream),
                           contextOf(*device), "M_DriverStreamWaitForDevice");
}

M_DriverStatus M_DriverStreamWaitForEvent(const M_DriverStream *stream,
                                          const M_DriverEvent *event) {
  if (stream == nullptr || event == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or event is null");
  if (!event->event.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "event is empty");
  if (!sameApi(*stream->ctx, *event->event))
    return mixedBackends("M_DriverStreamWaitForEvent");
  // `enqueueWaitFor` predates const correctness and takes a mutable event;
  // the wait itself does not modify it.
  DeviceEvent &target = const_cast<DeviceEvent &>(*event->event);
  if (isExternal(*stream))
    return reportIfError(stream->external->enqueueWaitForEvent(target),
                         "M_DriverStreamWaitForEvent");
  return reportIfError(stream->ctx->enqueueWaitFor(target),
                       "M_DriverStreamWaitForEvent");
}

M_DriverStatus M_DriverStreamRecordEvent(const M_DriverStream *stream,
                                         const M_DriverEvent *event) {
  if (stream == nullptr || event == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or event is null");
  if (!event->event.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "event is empty");
  if (!sameApi(*stream->ctx, *event->event))
    return mixedBackends("M_DriverStreamRecordEvent");
  const DeviceEvent &target = *event->event;
  if (isExternal(*stream)) {
#if MODULAR_HAL_DEVICE_CONTEXT
    // A HAL stream adopted from a native handle cannot record an event of its
    // own, so ordering against it goes through the context instead.
    return fail(M_DRIVER_NOT_SUPPORTED,
                "M_DriverStreamRecordEvent on a native handle stream is not "
                "available in a HAL device context build");
#else
    return reportIfError(stream->external->eventRecord(target),
                         "M_DriverStreamRecordEvent");
#endif
  }
  return reportIfError(stream->ctx->enqueueEvent(target),
                       "M_DriverStreamRecordEvent");
}

M_DriverStatus M_DriverStreamRecordNewEvent(const M_DriverStream *stream,
                                            M_DriverEvent **out) {
  if (stream == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream or out is null");
  *out = nullptr;
  // A native queue records on itself, as `M_DriverStreamRecordEvent` and the
  // waits do, rather than on the context created beside it.
  ErrorOr<DeviceEventRef> event = isExternal(*stream)
                                      ? stream->external->enqueueEvent()
                                      : stream->ctx->enqueueEvent();
  if (event.isError())
    return fail(event, "M_DriverStreamRecordNewEvent");
  *out = new M_DriverEvent{event.takeValue()};
  return succeed();
}

//===----------------------------------------------------------------------===//
// Events
//===----------------------------------------------------------------------===//

M_DriverStatus M_DriverEventCreate(const M_DriverDevice *device,
                                   bool enable_timing, M_DriverEvent **out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
  *out = nullptr;
  // Created on the context this interface owns, so that the pending error
  // check creation runs cannot take an error MAX recorded on the shared one.
  // The event records and waits on any stream of the device all the same.
  ErrorOr<Service *> service = serviceFor(contextOf(*device));
  if (service.isError())
    return fail(service, "M_DriverEventCreate");
  std::lock_guard<std::mutex> lock((*service)->pollMutex);
  ErrorOr<DeviceEventRef> event = (*service)->ctx->eventCreate(enable_timing);
  if (event.isError())
    return fail(event, "M_DriverEventCreate");
  *out = new M_DriverEvent{event.takeValue()};
  return succeed();
}

M_DriverStatus M_DriverEventDestroy(M_DriverEvent *event) {
  if (event == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "event is null");
  delete event;
  return succeed();
}

M_DriverStatus M_DriverEventSynchronize(const M_DriverEvent *event) {
  if (event == nullptr || !event->event.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "event is null or empty");
  return reportIfError(event->event->synchronize(), "M_DriverEventSynchronize");
}

M_DriverStatus M_DriverEventIsReady(const M_DriverEvent *event, bool *out) {
  if (event == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "event or out is null");
  ErrorOr<bool> ready = event->event->isReady();
  if (ready.isError())
    return fail(ready, "M_DriverEventIsReady");
  *out = ready.get();
  return succeed();
}

M_DriverStatus M_DriverEventElapsedTimeMs(const M_DriverEvent *start,
                                          const M_DriverEvent *end,
                                          double *out_ms) {
  if (start == nullptr || end == nullptr || out_ms == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "start, end or out_ms is null");
  if (!start->event.getPointer() || !end->event.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "start or end is empty");
  if (!sameApi(*start->event, *end->event))
    return mixedBackends("M_DriverEventElapsedTimeMs");
  ErrorOr<float> elapsed = start->event->elapsedTime(*end->event);
  if (elapsed.isError())
    return fail(elapsed, "M_DriverEventElapsedTimeMs");
  *out_ms = static_cast<double>(elapsed.get());
  return succeed();
}

//===----------------------------------------------------------------------===//
// Buffers, host memory registration, and copies
//===----------------------------------------------------------------------===//

M_DriverStatus M_DriverBufferAllocate(const M_DriverDevice *device,
                                      uint64_t byte_size, bool pinned_host,
                                      M_DriverBuffer **out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
  *out = nullptr;
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  ErrorOr<DeviceBufferRef> buffer =
      pinned_host ? contextOf(*device).createHostBuffer(byte_size)
                  : contextOf(*device).enqueueCreateBuffer(byte_size);
  if (buffer.isError())
    return fail(buffer, "M_DriverBufferAllocate", M_DRIVER_OUT_OF_MEMORY);
  *out = new M_DriverBuffer{buffer.takeValue()};
  return succeed();
}

M_DriverStatus M_DriverBufferFromPointer(const M_DriverDevice *device,
                                         void *pointer, uint64_t byte_size,
                                         bool is_host, M_DriverBuffer **out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
  *out = nullptr;
  if (pointer == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "pointer is null");
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  // `owning = false`: the memory belongs to the caller, and this handle must
  // never free it. It only orders the caller's own release against in-flight
  // copies.
  DeviceBufferRef buffer =
      is_host ? contextOf(*device).createHostBufferOwning(pointer, byte_size,
                                                          /*owning=*/false)
              : contextOf(*device).createBufferOwning(pointer, byte_size,
                                                      /*owning=*/false);
  *out = new M_DriverBuffer{std::move(buffer)};
  return succeed();
}

M_DriverStatus M_DriverBufferDestroy(M_DriverBuffer *buffer) {
  if (buffer == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "buffer is null");
  delete buffer;
  return succeed();
}

M_DriverStatus M_DriverBufferPointer(const M_DriverBuffer *buffer, void **out) {
  if (buffer == nullptr || out == nullptr || !buffer->buffer.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "buffer or out is null");
  *out = buffer->buffer->ptr();
  return succeed();
}

M_DriverStatus M_DriverBufferByteSize(const M_DriverBuffer *buffer,
                                      uint64_t *out) {
  if (buffer == nullptr || out == nullptr || !buffer->buffer.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "buffer or out is null");
  *out = static_cast<uint64_t>(buffer->buffer->byteSize());
  return succeed();
}

M_DriverStatus M_DriverBufferIsHost(const M_DriverBuffer *buffer, bool *out) {
  if (buffer == nullptr || out == nullptr || !buffer->buffer.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "buffer or out is null");
  *out = buffer->buffer->isHost();
  return succeed();
}

M_DriverStatus M_DriverMemoryRegister(const M_DriverDevice *device,
                                      void *pointer, uint64_t byte_size,
                                      M_DriverMapping **out) {
  if (device == nullptr || out == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "device or out is null");
  *out = nullptr;
  if (pointer == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "pointer is null");
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  ErrorOr<Service *> service = serviceFor(contextOf(*device));
  if (service.isError())
    return fail(service, "M_DriverMemoryRegister");
  // Registering reads the pending error a concurrent
  // `M_DriverMappingDestroySync` is about to report as its own.
  std::lock_guard<std::mutex> lock((*service)->pollMutex);
  ErrorOr<DeviceBufferRef> mapping =
      (*service)->ctx->wrapHostMemory(pointer, byte_size);
  if (mapping.isError())
    return fail(mapping, "M_DriverMemoryRegister");
  *out = new M_DriverMapping{mapping.takeValue(), *service};
  return succeed();
}

M_DriverStatus M_DriverMappingDestroy(M_DriverMapping *mapping) {
  if (mapping == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "mapping is null");
  // Dropping the buffer polls the context before it queues the unregister.
  std::lock_guard<std::mutex> lock(mapping->service->pollMutex);
  delete mapping;
  return succeed();
}

M_DriverStatus M_DriverMappingDestroySync(M_DriverMapping *mapping) {
  if (mapping == nullptr || !mapping->buffer.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "mapping is null");
  Service &service = *mapping->service;
  std::lock_guard<std::mutex> lock(service.pollMutex);
  // The handle goes first, so the caller holds nothing that could reach the
  // buffer again whatever the unregister reports. Dropping the buffer queues
  // the unregister behind an event on the service context's stream.
  delete mapping;
  // The first synchronize runs the queued unregister on this thread; a
  // refusal is recorded, not returned, so the second reports it.
  M_DriverStatus status =
      reportIfError(service.ctx->synchronize(), "M_DriverMappingDestroySync");
  if (status != M_DRIVER_OK)
    return status;
  return reportIfError(service.ctx->synchronize(),
                       "M_DriverMappingDestroySync");
}

M_DriverStatus M_DriverMappingPointer(const M_DriverMapping *mapping,
                                      void **out) {
  if (mapping == nullptr || out == nullptr || !mapping->buffer.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "mapping or out is null");
  *out = mapping->buffer->ptr();
  return succeed();
}

M_DriverStatus M_DriverMappingByteSize(const M_DriverMapping *mapping,
                                       uint64_t *out) {
  if (mapping == nullptr || out == nullptr || !mapping->buffer.getPointer())
    return fail(M_DRIVER_INVALID_ARGUMENT, "mapping or out is null");
  *out = static_cast<uint64_t>(mapping->buffer->byteSize());
  return succeed();
}

M_DriverStatus M_DriverCopyHostToDevice(const M_DriverStream *stream,
                                        void *dst_device, const void *src_host,
                                        uint64_t byte_size) {
  if (stream == nullptr || dst_device == nullptr || src_host == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "stream, dst_device or src_host is null");
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  if (isExternal(*stream))
    return notSupportedForExternalStream("M_DriverCopyHostToDevice");
  return reportIfError(
      stream->ctx->enqueueCopyToDevice(dst_device, src_host, byte_size),
      "M_DriverCopyHostToDevice");
}

M_DriverStatus M_DriverCopyDeviceToHost(const M_DriverStream *stream,
                                        void *dst_host, const void *src_device,
                                        uint64_t byte_size) {
  if (stream == nullptr || dst_host == nullptr || src_device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "stream, dst_host or src_device is null");
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  if (isExternal(*stream))
    return notSupportedForExternalStream("M_DriverCopyDeviceToHost");
  return reportIfError(
      stream->ctx->enqueueCopyFromDevice(dst_host, src_device, byte_size),
      "M_DriverCopyDeviceToHost");
}

M_DriverStatus M_DriverCopyBatch(const M_DriverStream *stream,
                                 M_DriverCopyDirection direction,
                                 void *const *dsts, const void *const *srcs,
                                 const uint64_t *sizes, size_t count) {
  if (stream == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT, "stream is null");
  if (direction != M_DRIVER_COPY_HOST_TO_DEVICE &&
      direction != M_DRIVER_COPY_DEVICE_TO_HOST)
    return fail(M_DRIVER_INVALID_ARGUMENT, "direction is not a copy direction");
  if (isExternal(*stream))
    return notSupportedForExternalStream("M_DriverCopyBatch");
  if (count == 0)
    return succeed();
  if (dsts == nullptr || srcs == nullptr || sizes == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "dsts, srcs or sizes is null with a non zero count");

  // Checked here so a bad piece is a status before anything is enqueued.
  for (size_t i = 0; i < count; ++i) {
    if (dsts[i] == nullptr || srcs[i] == nullptr)
      return fail(M_DRIVER_INVALID_ARGUMENT,
                  "piece " + std::to_string(i) + " has a null address");
    if (sizes[i] == 0)
      return fail(M_DRIVER_INVALID_ARGUMENT,
                  "piece " + std::to_string(i) + " has a size of 0");
  }

  // Borrowed wrappers, so dropping them neither frees nor unregisters the
  // caller's memory; on the stream's context, so the batch orders only there.
  const bool toDevice = direction == M_DRIVER_COPY_HOST_TO_DEVICE;
  DeviceContext &ctx = *stream->ctx;
  auto borrow = [&](const void *address, uint64_t size, bool host) {
    void *ptr = const_cast<void *>(address);
    return host ? ctx.createHostBufferOwning(ptr, size, /*owning=*/false)
                : ctx.createBufferOwning(ptr, size, /*owning=*/false);
  };
  llvm::SmallVector<DeviceBufferRef, 16> keepAlive;
  llvm::SmallVector<DeviceBuffer *, 16> dstBuffers;
  llvm::SmallVector<DeviceBuffer *, 16> srcBuffers;
  keepAlive.reserve(2 * count);
  dstBuffers.reserve(count);
  srcBuffers.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    DeviceBufferRef dst = borrow(dsts[i], sizes[i], /*host=*/!toDevice);
    DeviceBufferRef src = borrow(srcs[i], sizes[i], /*host=*/toDevice);
    if (!dst || !src)
      return fail(M_DRIVER_RUNTIME_ERROR,
                  "piece " + std::to_string(i) + " could not be wrapped");
    dstBuffers.push_back(dst.getPointer());
    srcBuffers.push_back(src.getPointer());
    keepAlive.push_back(std::move(dst));
    keepAlive.push_back(std::move(src));
  }
  return reportIfError(ctx.enqueueBatchCopy(dstBuffers, srcBuffers),
                       "M_DriverCopyBatch");
}

M_DriverStatus M_DriverCopyDeviceToDevice(const M_DriverStream *stream,
                                          void *dst_device,
                                          const void *src_device,
                                          uint64_t byte_size) {
  if (stream == nullptr || dst_device == nullptr || src_device == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "stream, dst_device or src_device is null");
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  if (isExternal(*stream))
    return notSupportedForExternalStream("M_DriverCopyDeviceToDevice");
  DeviceContext &ctx = *stream->ctx;
  // The copy is issued on the context that owns the destination, which is what
  // the driver checks; these handles only describe the two ranges.
  DeviceBufferRef dst = ctx.createBufferOwning(dst_device, byte_size,
                                               /*owning=*/false);
  DeviceBufferRef src = ctx.createBufferOwning(const_cast<void *>(src_device),
                                               byte_size, /*owning=*/false);
  return reportIfError(
      ctx.enqueueCopyDeviceToDevice(*dst, *src, /*srcByteOffset=*/0, byte_size,
                                    /*crossStreamSync=*/true),
      "M_DriverCopyDeviceToDevice");
}

namespace {

/// The peer copy behind `M_DriverCopyPeerToPeer` and its unordered variant,
/// which differ only in whether the driver brackets the copy with waits on the
/// source context's stream.
M_DriverStatus copyPeerToPeer(const char *what, const M_DriverStream *stream,
                              const M_DriverDevice *dst_device, void *dst,
                              const M_DriverDevice *src_device, const void *src,
                              uint64_t byte_size, bool crossStreamSync) {
  if (stream == nullptr || dst_device == nullptr || src_device == nullptr ||
      dst == nullptr || src == nullptr)
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "stream, device or address argument is null");
  if (byte_size == 0)
    return fail(M_DRIVER_INVALID_ARGUMENT, "byte_size is 0");
  if (isExternal(*stream))
    return notSupportedForExternalStream(what);
  // The copy is enqueued on the destination's context; a stream that runs on
  // another device would enqueue it against the wrong memory.
  M_DriverDevice streamDevice{DeviceContextRef::copy(stream->ctx.getPointer())};
  if (!sameDevice(streamDevice, *dst_device))
    return fail(M_DRIVER_INVALID_ARGUMENT,
                "the stream does not run on dst_device; a peer copy is "
                "enqueued on the destination's own context");
  if (!sameApi(*stream->ctx, contextOf(*src_device)))
    return mixedBackends(what);
  // The destination is stamped with the copy stream's own context, because
  // the driver waits for the source and records the copy's completion on the
  // destination buffer's context, so the device's shared context there would
  // bracket MAX's compute stream instead of this stream. The ordered variant
  // stamps the source with its device's shared context, which is the context
  // it waits on. The unordered variant orders nothing there, so it stamps the
  // source with the context this interface owns on that device and the
  // backend's pending error check on the source reads that slot instead of
  // MAX's. Either way the two contexts sit on different devices, which is
  // what makes the driver emit a direct peer copy rather than staging
  // through host memory.
  DeviceContextRef srcCtx =
      DeviceContextRef::copy(src_device->ctx.getPointer());
  if (!crossStreamSync) {
    ErrorOr<Service *> service = serviceFor(*src_device->ctx);
    if (service.isError())
      return fail(service, what);
    srcCtx = DeviceContextRef::copy((*service)->ctx.getPointer());
  }
  DeviceBufferRef dstBuffer =
      stream->ctx->createBufferOwning(dst, byte_size, /*owning=*/false);
  DeviceBufferRef srcBuffer = srcCtx->createBufferOwning(
      const_cast<void *>(src), byte_size, /*owning=*/false);
  return reportIfError(stream->ctx->enqueueCopyDeviceToDevice(
                           *dstBuffer, *srcBuffer, /*srcByteOffset=*/0,
                           byte_size, crossStreamSync),
                       what);
}

} // namespace

M_DriverStatus M_DriverCopyPeerToPeer(const M_DriverStream *stream,
                                      const M_DriverDevice *dst_device,
                                      void *dst,
                                      const M_DriverDevice *src_device,
                                      const void *src, uint64_t byte_size) {
  return copyPeerToPeer("M_DriverCopyPeerToPeer", stream, dst_device, dst,
                        src_device, src, byte_size, /*crossStreamSync=*/true);
}

M_DriverStatus M_DriverCopyPeerToPeerUnordered(
    const M_DriverStream *stream, const M_DriverDevice *dst_device, void *dst,
    const M_DriverDevice *src_device, const void *src, uint64_t byte_size) {
  return copyPeerToPeer("M_DriverCopyPeerToPeerUnordered", stream, dst_device,
                        dst, src_device, src, byte_size,
                        /*crossStreamSync=*/false);
}
