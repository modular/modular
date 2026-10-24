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

#ifndef SDK_DRIVERAPI_PYTHON_DEVICE_H
#define SDK_DRIVERAPI_PYTHON_DEVICE_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"
#include "Support/RCRef.h"
#include "_core/DLPack/Utils.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/FunctionExtras.h"
#include <cstdint>
#include <memory>

namespace nb = nanobind;

namespace M::Driver::Python {

struct CompletionFlag;
struct Device;
struct CPU;
struct Accelerator;
struct DeviceStream;
struct DeviceEvent;

namespace Detail {

/// Data the AsyncRT worker reads when it runs an async host func: the
/// Python callable, the completion flag the worker signals on
/// completion, and the 64-bit value to write to that flag. Intentionally
/// does NOT include the `CPUDeviceRef` the trampoline used to dispatch
/// — the worker doesn't need it.
struct AsyncHostFuncTaskData {
  nb::object fn;
  M::Driver::CompletionFlagRef flagRef;
  std::uint64_t signalValue;
};

/// Trampoline-scope data: the task payload the AsyncRT worker will
/// read, plus an MLRT `CPUDeviceRef` used solely to invoke `addTask`
/// on the right worker pool. The trampoline drops `cpuDeviceRef` once
/// dispatch returns; only `task` is moved into the worker closure.
/// Callers are expected to ensure their `Python::CPU` wrapper outlives
/// any in-flight task (in practice it is process-global for an
/// inference session).
struct AsyncHostFuncUserData {
  AsyncHostFuncTaskData task;
  M::AsyncRT::CPUDeviceRef cpuDeviceRef;
};

} // namespace Detail

using PyDeviceTy = std::shared_ptr<Device>;
using PyCPUTy = std::shared_ptr<CPU>;
using PyAcceleratorTy = std::shared_ptr<Accelerator>;
using PyDeviceStreamTy = std::shared_ptr<DeviceStream>;
using PyDeviceEventTy = std::shared_ptr<DeviceEvent>;

/// An abstraction over an MLRT device context to bind with python.
/// Here we are using std::shared_ptr instead of RCRef
/// because this library is built with exceptions enabled.
///
/// Holds an MLRT `DeviceContextRef` directly. `spec` is kept alongside
/// because the `cpu`/`gpu` label distinction is not recoverable from the
/// MLRT context's own `api()`, which reports the concrete backend
/// ("cuda", "hip", "metal", ...) rather than the label.
struct Device : public std::enable_shared_from_this<Device> {

  Device(M::DeviceSpec deviceSpec, M::Driver::DeviceContextRef ctx);
  virtual ~Device() = default;

  std::string getAsString() const;

  /// Allocates `spec`'s worth of memory on this device. Releases the GIL.
  /// Throws `std::invalid_argument` on failure.
  M::Driver::DeviceBufferRef allocate(const TensorSpec &spec,
                                      bool pinned = false);

  /// Same allocation as `allocate`, but does not touch the GIL and reports
  /// failure via `ErrorOr` instead of throwing. For callers that have
  /// already released the GIL themselves (e.g. `Buffer::copyTo`), to avoid
  /// a nested `gil_scoped_release`.
  ErrorOr<M::Driver::DeviceBufferRef> allocateBuffer(const TensorSpec &spec,
                                                     bool pinned = false) const;

  /// Wraps an externally-owned pointer (e.g. from a numpy array or a DLPack
  /// tensor) in a `DeviceBuffer` that runs `destructor` when the last
  /// reference is dropped, without copying the data.
  M::Driver::DeviceBufferRef
  wrapExternalBuffer(void *data, const TensorSpec &spec,
                     llvm::unique_function<void()> destructor) const;

  std::string getLabel() const;

  uint64_t getId() const;

  /// Returns whether or not the device is the host device.
  bool isHost() const;

  /// Returns whether device and host memory are one physical pool.
  bool isHostUnified() const;

  // Returns utilization statistics for device.
  std::string getStats() const;

  // Returns the largest single contiguous allocation the device supports.
  size_t getMaxSingleAllocSize() const;

  // Returns (base, size) of the driver allocation containing `address`.
  std::pair<uintptr_t, size_t> getAllocationRange(uintptr_t address) const;

  // Returns the programming API for the device.
  std::string getApi() const;

  // Returns the architecture name for the device.
  std::string getArchName() const;

  // Returns the model name of the device (e.g. "NVIDIA B200", "AMD MI300X").
  std::string getModelName() const;

  PyDeviceStreamTy defaultStream();

  /// Returns whether or not the device is compatible with MAX.
  bool isCompatible() const;

  /// Ensures all operations on the device complete before returning.
  void synchronize() const;

  /// Returns whether this device can directly access memory of `other`.
  bool canAccess(const Device &other) const;

  bool operator==(const Device &other) const;
  bool operator!=(const Device &other) const;

  /// Hash uniquely identifying the device.
  size_t hash() const;

  M::DeviceSpec deviceSpec;
  M::Driver::DeviceContextRef ctx;
};

struct CPU : public Device {
  CPU(M::DeviceSpec deviceSpec, M::Driver::DeviceContextRef ctx)
      : Device(std::move(deviceSpec), std::move(ctx)) {}

  /// Returns the AsyncRT CPUDevice that backs this CPU's MLRT worker
  /// pool. Used by the async-host-func trampoline to dispatch the
  /// Python callable onto an explicit, caller-supplied AsyncRT runtime
  /// rather than the global/TLS one.
  ///
  /// Raises ``RuntimeError`` if this Python::CPU's underlying device
  /// context is not a CpuDeviceContext (shouldn't happen in practice
  /// since `Python::CPU` is only constructed by `getCpuDevice`, which
  /// creates the context against `Device::cpuAPI`).
  M::AsyncRT::CPUDeviceRef getAsyncRTCPUDevice() const;

  /// Packages a callable + completion flag + signal value into a
  /// heap-allocated user-data block paired with this CPU's AsyncRT
  /// `CPUDeviceRef`, ready for the async-host-func trampoline to
  /// dispatch. Encapsulates the `CPUDeviceRef` so call sites in the
  /// binding layer don't need to plumb it through themselves.
  std::unique_ptr<Detail::AsyncHostFuncUserData>
  makeAsyncHostFuncUserData(nb::object fn, const CompletionFlag &flag,
                            std::uint64_t value) const;
};

struct Accelerator : public Device {
  Accelerator(M::DeviceSpec deviceSpec, M::Driver::DeviceContextRef ctx)
      : Device(std::move(deviceSpec), std::move(ctx)) {}
};

struct DeviceEvent : public std::enable_shared_from_this<DeviceEvent> {
  DeviceEvent(M::Driver::DeviceEventRef ev);
  DeviceEvent(Device &device, bool enable_timing = false);
  ~DeviceEvent() = default;

  std::string getAsString() const;

  void synchronize() const;
  bool isReady() const;

  float elapsedTime(const DeviceEvent &end) const;

  bool operator==(const DeviceEvent &other) const;

  M::Driver::DeviceEventRef ev;

private:
  static M::Driver::DeviceEventRef createEventFromDevice(Device &device,
                                                         bool enable_timing);
};

struct DeviceStream : public std::enable_shared_from_this<DeviceStream> {
  DeviceStream(PyDeviceTy device);

  DeviceStream(Device &device, M::Driver::DeviceContextRef context);
  ~DeviceStream();

  std::string getAsString() const;

  void synchronize() const;

  void waitFor(const DeviceStream &stream) const;

  void waitForDevice(const Device &device) const;

  /// Native stream handle (CUDA `CUstream` / HIP `hipStream_t`) as an integer,
  /// or 0 if the stream has no native handle (e.g. a CPU device). Lets native
  /// code (outside MLRT) order its own work against this stream — e.g. record a
  /// CUDA event on it. Read-only; the handle stays owned by this stream.
  uintptr_t nativeStreamHandle() const;

  PyDeviceTy getDevice() const;

  PyDeviceEventTy enqueueEvent();
  void enqueueEvent(const DeviceEvent &event) const;

  void enqueueHostFunc(nb::callable &&fn) const;

  /// Async variant of `enqueueHostFunc`: the kickoff host node dispatches
  /// `fn` onto the caller-supplied `cpu`'s AsyncRT worker pool and
  /// returns immediately, so the stream proceeds to the next node
  /// concurrently with `fn` running on an AsyncRT worker. When `fn`
  /// finishes, the worker atomic-stores `value` (release ordering) to
  /// `flag`'s 8-byte memory. Pair with `waitForHostValue(flag, value)`
  /// later on the stream to gate the consumer.
  ///
  /// The trampoline holds a `CompletionFlagRef` for the duration of the
  /// AsyncRT task so the flag allocation cannot be released out from
  /// under the in-flight worker. The trampoline does NOT hold a
  /// `CPUDeviceRef` past dispatch; callers must ensure the `cpu`
  /// wrapper outlives any in-flight task (in practice
  /// `max.driver.CPU` is process-global for an inference session).
  void enqueueAsyncHostFunc(nb::callable &&fn, const CompletionFlag &flag,
                            uint64_t value, const CPU &cpu) const;

  /// Wraps MLRT's `DeviceStream::enqueueWaitOnHostValue`. Stalls the stream
  /// until the 64-bit value at `flag.devicePtr()` equals `value`.
  /// Typically paired with `enqueueAsyncHostFunc` to gate the GPU stream
  /// on a CPU task dispatched by the kickoff callback.
  void waitForHostValue(const CompletionFlag &flag, uint64_t value) const;

  static PyDeviceStreamTy createStream(Device &device);

  bool operator==(const DeviceStream &other) const;

  PyDeviceTy dev;
  PyDeviceTy stream; // A container for the different DeviceContext.
};

/// Helper to return the Python device object from dlpack device type and id.
std::tuple<PyDeviceTy, std::optional<intptr_t>>
getPyDevice(DLDeviceType deviceType, ssize_t deviceID);

/// Helper to return the Python device object from a dlpack array, along with a
/// stream ID (or none if on host).
std::tuple<PyDeviceTy, std::optional<intptr_t>>
getPyDevice(const nb::object &arrayOrTensor);

PyCPUTy getCpuDevice(ssize_t numaId);
PyAcceleratorTy getAccelerator(ssize_t gpuId);
// Returns number of accelerators available.
int acceleratorCount();

/// Packs `fn` into a heap-allocated trampoline payload compatible with
/// `MLRT::DeviceStream::enqueueHostFunc`. Returns a pair of opaque
/// pointer-sized integers: the C trampoline function pointer and a raw
/// pointer to the user-data block. Ownership of the user data transfers
/// to the caller; the caller must ensure it is eventually consumed by
/// exactly one `enqueueHostFunc` invocation (after which the trampoline
/// frees it). Leaking the user-data pointer leaks the underlying Python
/// callable.
std::tuple<uintptr_t, uintptr_t> packHostFunc(nb::callable fn);

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_DEVICE_H
