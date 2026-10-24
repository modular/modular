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

#include "_core/Driver/Device.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#if MODULAR_HAL_DEVICE_CONTEXT
#include "AsyncRT/Runtime/HostSystem.h"
#else
#include "MLRT/Driver/DeviceContext/CPU/CpuDeviceContext.h"
#endif
#include "_core/DLPack/Utils.h"
#include "_core/Driver/CompletionFlag.h"
#include "nanobind/nanobind.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

using namespace M;
using namespace Driver;
using namespace std::string_literals;

Python::Device::Device(M::DeviceSpec deviceSpec,
                       M::Driver::DeviceContextRef ctx)
    : deviceSpec(std::move(deviceSpec)), ctx(std::move(ctx)) {}

std::string Python::Device::getAsString() const {
  std::string str;
  llvm::raw_string_ostream oss(str);
  oss << "Device(type=" << deviceSpec.ref.label << ",";
  oss << "id=" << deviceSpec.ref.id << ")";
  return str;
}

ErrorOr<M::Driver::DeviceBufferRef>
Python::Device::allocateBuffer(const TensorSpec &spec, bool pinned) const {
  return M::GraphCompiler::Driver::allocateDeviceBuffer(ctx, spec, pinned);
}

M::Driver::DeviceBufferRef Python::Device::allocate(const TensorSpec &spec,
                                                    bool pinned) {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::DeviceBufferRef> bufOr = allocateBuffer(spec, pinned);
  if (bufOr.isError())
    throw std::invalid_argument(bufOr.getError());
  return bufOr.takeValue();
}

M::Driver::DeviceBufferRef Python::Device::wrapExternalBuffer(
    void *data, const TensorSpec &spec,
    llvm::unique_function<void()> destructor) const {
  M::Driver::DeviceBufferRef buf =
      ctx->createBufferOwning(data, spec.getSizeInBytes(), /*owning=*/true);
  buf->setDestructorFn(std::move(destructor));
  return buf;
}

std::string Python::Device::getLabel() const { return deviceSpec.ref.label; }

uint64_t Python::Device::getId() const { return deviceSpec.ref.id; }

bool Python::Device::isHost() const {
  return ctx->device().api() == M::Driver::Device::cpuAPI;
}

M::AsyncRT::CPUDeviceRef Python::CPU::getAsyncRTCPUDevice() const {
  // `Python::CPU` is only constructed by `getCpuDevice`, which creates the
  // context against `Device::cpuAPI`, so it is always a `CpuDeviceContext`.
  // The downcast helper asserts this contract.
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL DeviceContext has no CpuDeviceContext subclass to reach through.
  // Use the same process-wide AsyncRT CPUDevice that CpuDeviceContext creates
  // (same source and options), so all consumers share one CPUDevice.
  M::AsyncRT::CPUDeviceRef cpuDevice = M::AsyncRT::getOrCreateCPUDevice(
      M::AsyncRT::CPUDeviceSource::CPUDeviceContext,
      M::AsyncRT::CPUDeviceOptions().withMainWillNotDonate(),
      /*allowUsingExistingOptions=*/true);
#else
  const M::Driver::CpuDeviceContext &cpuCtx =
      M::Driver::CpuDeviceContext::cast(*ctx);
  M::AsyncRT::CPUDeviceRef cpuDevice = cpuCtx.getCPUDevice();
#endif
  if (!cpuDevice) {
    throw std::runtime_error(
        "max.driver.CPU has no associated AsyncRT CPUDevice");
  }
  return cpuDevice;
}

std::unique_ptr<Python::Detail::AsyncHostFuncUserData>
Python::CPU::makeAsyncHostFuncUserData(nb::object fn,
                                       const CompletionFlag &flag,
                                       std::uint64_t value) const {
  // `flag.ref().copy()` bumps the RCRef; the AsyncRT worker holds it
  // for the duration of the task so the MLRT allocation outlives the
  // Python wrapper's GC state.
  return std::make_unique<Detail::AsyncHostFuncUserData>(
      Detail::AsyncHostFuncUserData{
          /*task=*/{std::move(fn), flag.ref().copy(), value},
          /*cpuDeviceRef=*/getAsyncRTCPUDevice(),
      });
}

std::string Python::Device::getStats() const {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::Device::MemInfo> infoOr = ctx->getMemoryInfo();
  if (infoOr.isError()) {
    std::string errorMessage =
        "failed to get device stats: "s + infoOr.getError();
    throw std::runtime_error(errorMessage);
  }
  M::Driver::Device::MemInfo info = infoOr.get();

  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::json::OStream j(os);

  j.object([&] {
    j.attribute("free_memory", info.free);
    j.attribute("total_memory", info.total);
    ErrorOr<M::Driver::Device::GraphMemInfo> graphMemResult =
        ctx->device().getGraphMemoryInfo();
    if (!graphMemResult.isError()) {
      M::Driver::Device::GraphMemInfo graphMem = graphMemResult.takeValue();
      j.attribute("graph_mem_reserved", graphMem.reserved);
      j.attribute("graph_mem_used", graphMem.used);
    }
  });

  os.flush();
  return s;
}

size_t Python::Device::getMaxSingleAllocSize() const {
  nb::gil_scoped_release release;
  ErrorOr<size_t> sizeOr = ctx->device().maxSingleAllocationSize();
  if (sizeOr.isError()) {
    std::string errorMessage =
        "failed to get device max single allocation size: "s +
        sizeOr.getError();
    throw std::runtime_error(errorMessage);
  }
  return sizeOr.get();
}

std::pair<uintptr_t, size_t>
Python::Device::getAllocationRange(uintptr_t address) const {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::Device::AllocationRange> rangeOr =
      ctx->device().allocationRange(reinterpret_cast<void *>(address));
  if (rangeOr.isError())
    throw std::runtime_error("failed to get the allocation range: "s +
                             rangeOr.getError());
  return {reinterpret_cast<uintptr_t>(rangeOr->base), rangeOr->size};
}

bool Python::Device::isHostUnified() const {
  nb::gil_scoped_release release;
  ErrorOr<bool> unifiedOr = ctx->device().isHostUnified();
  if (unifiedOr.isError()) {
    std::string errorMessage =
        "failed to query whether device memory is unified with the host: "s +
        unifiedOr.getError();
    throw std::runtime_error(errorMessage);
  }
  return unifiedOr.get();
}

std::string Python::Device::getApi() const {
  nb::gil_scoped_release release;
  return ctx->device().api().str();
}

std::string Python::Device::getArchName() const {
  nb::gil_scoped_release release;
  ErrorOr<const char *> archNameOr = ctx->device().getArchName();
  if (archNameOr.isError()) {
    std::string errorMessage =
        "failed to get device architecture name: "s + archNameOr.getError();
    throw std::invalid_argument(errorMessage);
  }
  return archNameOr.get();
}

std::string Python::Device::getModelName() const {
  nb::gil_scoped_release release;
  return ctx->device().name();
}

Python::PyDeviceStreamTy Python::Device::defaultStream() {
  return std::make_unique<Python::DeviceStream>(*this, this->ctx);
}

bool Python::Device::isCompatible() const {
  nb::gil_scoped_release release;
  ErrorOrSuccess compatibleOr = ctx->device().isCompatible();
  if (compatibleOr.isError()) {
    throw std::runtime_error(compatibleOr.getError());
  }
  return true;
}

void Python::Device::synchronize() const {
  // Release the GIL so host callbacks enqueued via `enqueue_host_func`
  // (which need to acquire the GIL to invoke Python) can make progress
  // while we wait for the device to drain.
  nb::gil_scoped_release release;
  if (auto errOr = ctx->synchronize(); errOr.isError())
    throw std::runtime_error(errOr.getError());
}

bool Python::Device::canAccess(const Device &other) const {
  ErrorOr<bool> canAccessOr = ctx->device().canAccess(other.ctx->device());
  if (canAccessOr.isError())
    throw std::runtime_error(canAccessOr.getError());
  return *canAccessOr;
}

bool Python::Device::operator==(const Device &other) const {
  return this->deviceSpec.ref == other.deviceSpec.ref;
}

bool Python::Device::operator!=(const Device &other) const {
  return !(*this == other);
}

Python::DeviceEvent::DeviceEvent(M::Driver::DeviceEventRef ev) : ev(ev) {}

Python::DeviceEvent::DeviceEvent(Device &device, bool enable_timing)
    : ev(createEventFromDevice(device, enable_timing)) {}

M::Driver::DeviceEventRef
Python::DeviceEvent::createEventFromDevice(Device &device, bool enable_timing) {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::DeviceEventRef> eventOr =
      device.ctx->eventCreate(enable_timing);
  if (eventOr.isError())
    throw std::runtime_error(eventOr.getError());
  return *eventOr;
}

std::string Python::DeviceEvent::getAsString() const {
  std::stringstream ss;
  ss << ev->device().api().str() << "[" << ev->device().id() << "]:event@"
     << static_cast<const void *>(ev.getPointer());
  return ss.str();
}

void Python::DeviceEvent::synchronize() const {
  nb::gil_scoped_release release;
  ErrorOrSuccess err = ev->synchronize();
  if (err.isError())
    throw std::runtime_error(err.getError());
}

bool Python::DeviceEvent::isReady() const {
  nb::gil_scoped_release release;
  ErrorOr<bool> err = ev->isReady();
  if (err.isError())
    throw std::runtime_error(err.getError());
  return *err;
}

float Python::DeviceEvent::elapsedTime(const DeviceEvent &end) const {
  nb::gil_scoped_release release;
  ErrorOr<float> millis = ev->elapsedTime(*end.ev);
  if (millis.isError())
    throw std::runtime_error(millis.getError());
  return *millis;
}

bool Python::DeviceEvent::operator==(const DeviceEvent &other) const {
  // Events are equal if they refer to the same underlying event object
  return ev.getPointer() == other.ev.getPointer();
}

Python::DeviceStream::DeviceStream(Python::PyDeviceTy device)
    : dev(device), stream(std::move(device)) {}

Python::DeviceStream::DeviceStream(Device &device,
                                   M::Driver::DeviceContextRef context)
    : dev(device.shared_from_this()),
      stream(std::make_shared<Device>(dev->deviceSpec, std::move(context))) {}

Python::DeviceStream::~DeviceStream() {}

std::string Python::DeviceStream::getAsString() const {
  void *streamPtr = nullptr;
  if (*(stream->ctx) != *(dev->ctx)) {
    auto err = stream->ctx->stream();
    if (!err.isError())
      streamPtr = (*err)->ptr();
  }
  std::stringstream ss;
  if (streamPtr == nullptr) {
    ss << stream->ctx->device().api().str() << "[" << stream->ctx->device().id()
       << "]";
  } else {
    ss << stream->ctx->device().api().str() << "[" << stream->ctx->device().id()
       << ", " << streamPtr << "]";
  }
  return ss.str();
}

void Python::DeviceStream::synchronize() const {
  nb::gil_scoped_release release;
  ErrorOrSuccess err = stream->ctx->synchronize();
  if (err.isError())
    throw std::runtime_error(err.getError());
}

void Python::DeviceStream::waitFor(const DeviceStream &stream) const {
  nb::gil_scoped_release release;
  ErrorOrSuccess err = this->stream->ctx->enqueueWaitFor(*stream.stream->ctx);
  if (err.isError())
    throw std::runtime_error(err.getError());
}

Python::PyDeviceEventTy Python::DeviceStream::enqueueEvent() {
  nb::gil_scoped_release release;
  return std::make_unique<Python::DeviceEvent>(
      *(this->stream->ctx->enqueueEvent()));
}

void Python::DeviceStream::enqueueEvent(const DeviceEvent &event) const {
  nb::gil_scoped_release release;
  ErrorOrSuccess err = this->stream->ctx->enqueueEvent(*event.ev);
  if (err.isError())
    throw std::runtime_error(err.getError());
}

namespace {

// Owns the Python callable whose refcount must be managed under the GIL.
struct HostFuncUserData {
  nb::object fn;
};

// Invoked by the CUDA driver thread after all preceding stream work has
// completed. No GIL is held on entry.
//
// The GIL must be acquired before taking ownership of `rawUserData`. The
// `unique_ptr` destructor decrements the contained `nb::object`'s Python
// refcount, which is only legal with the GIL held. Declaring `userData`
// after `gil` guarantees the correct destruction order.
void hostFuncTrampoline(void *rawUserData) {
  nb::gil_scoped_acquire gil;
  std::unique_ptr<HostFuncUserData> userData(
      static_cast<HostFuncUserData *>(rawUserData));
  try {
    userData->fn();
  } catch (nb::python_error &e) {
    e.restore();
    PyErr_Print();
  } catch (const std::exception &e) {
    PySys_WriteStderr("host callback raised C++ exception: %s\n", e.what());
  }
}

} // namespace

void Python::DeviceStream::enqueueHostFunc(nb::callable &&fn) const {
  auto userData =
      std::make_unique<HostFuncUserData>(HostFuncUserData{std::move(fn)});

  ErrorOr<M::Driver::DeviceStreamRef> mlrtStream = stream->ctx->stream();
  if (mlrtStream.isError())
    throw std::runtime_error(mlrtStream.getError());

  ErrorOrSuccess err = [&] {
    nb::gil_scoped_release release;
    return (*mlrtStream)->enqueueHostFunc(&hostFuncTrampoline, userData.get());
  }();
  if (err.isError())
    throw std::runtime_error(err.getError());

  (void)userData.release();
}

std::tuple<uintptr_t, uintptr_t> Python::packHostFunc(nb::callable fn) {
  auto *userData = new HostFuncUserData{std::move(fn)};
  return {reinterpret_cast<uintptr_t>(&hostFuncTrampoline),
          reinterpret_cast<uintptr_t>(userData)};
}

namespace {

// CUDA host function. Runs on a CUDA driver thread once the stream
// reaches this node. No GIL on entry. Must return promptly so the
// stream can continue. Dispatches the Python callable onto the
// explicit AsyncRT CPUDevice carried in `userData`, then returns.
//
// The AsyncRT worker re-acquires the GIL, invokes the callable,
// releases the GIL, then signals the flag with `signalValue` (release
// ordering, encapsulated in `CompletionFlag::signal`). A
// `cuStreamWaitValue64` node placed later on the same stream observes
// the value and lets downstream work proceed.
//
// Lifetime split: the outer `AsyncHostFuncUserData` block (which
// includes the `CPUDeviceRef` we dispatch through) is owned by this
// trampoline's stack and freed before returning to CUDA. The inner
// `AsyncHostFuncTaskData` is moved into a separate heap allocation
// owned by the worker closure, so the worker carries only the data it
// actually reads.
void asyncHostFuncTrampoline(void *rawUserData) {
  // Take ownership of the outer block with RAII.
  std::unique_ptr<Python::Detail::AsyncHostFuncUserData> userData(
      static_cast<Python::Detail::AsyncHostFuncUserData *>(rawUserData));

  // Clear the flag before the worker runs. cuStreamWaitValue64 sees 0
  // until the worker calls `signal`.
  userData->task.flagRef->reset();

  // Move the task data into its own heap allocation owned by a
  // shared_ptr with a GIL-aware deleter so the `nb::object`'s
  // Py_DECREF runs under the GIL when the worker finishes.
  auto taskData = std::shared_ptr<Python::Detail::AsyncHostFuncTaskData>(
      new Python::Detail::AsyncHostFuncTaskData(std::move(userData->task)),
      [](Python::Detail::AsyncHostFuncTaskData *p) {
        nb::gil_scoped_acquire gil;
        delete p;
      });

  // Dispatch onto a non-device worker so the callback doesn't contend
  // with the per-GPU command-dispatch worker threads. Outside of cuGraph
  // capture this matters most: a callback that lands on a device-pinned
  // worker can stall GPU command submission. See
  // `MLRT::taskIdForNonDevice`.
  M::AsyncRT::CPUDevice &cpuDevice = *userData->cpuDeviceRef;
  std::size_t numWorkers = cpuDevice.getWorkQueue()->getParallelismLevel();
  M::AsyncRT::addTask(
      cpuDevice,
      [taskData = std::move(taskData)] {
        std::uint64_t value = taskData->signalValue;
        {
          nb::gil_scoped_acquire gil;
          try {
            taskData->fn();
          } catch (nb::python_error &e) {
            e.restore();
            PyErr_Print();
          } catch (const std::exception &e) {
            PySys_WriteStderr("async host callback raised C++ exception: %s\n",
                              e.what());
          }
        } // GIL released here before the signal store.
        // signal() does a release-ordered store: result stores in `fn`
        // are visible before the GPU-side `cuStreamWaitValue64`
        // observes the value.
        taskData->flagRef->signal(value);
      },
      M::MLRT::taskIdForNonDevice(0, numWorkers));
}

} // namespace

void Python::DeviceStream::enqueueAsyncHostFunc(nb::callable &&fn,
                                                const CompletionFlag &flag,
                                                uint64_t value,
                                                const CPU &cpu) const {
  // Eagerly construct the user-data block at enqueue time; the factory
  // pulls `cpu`'s `CPUDeviceRef` and raises a clear Python error if
  // `cpu` is malformed (rather than crashing on a null deref in the
  // trampoline).
  auto userData = cpu.makeAsyncHostFuncUserData(std::move(fn), flag, value);

  ErrorOr<M::Driver::DeviceStreamRef> mlrtStream = stream->ctx->stream();
  if (mlrtStream.isError())
    throw std::runtime_error(mlrtStream.getError());

  ErrorOrSuccess err = [&] {
    nb::gil_scoped_release release;
    return (*mlrtStream)
        ->enqueueHostFunc(&asyncHostFuncTrampoline, userData.get());
  }();
  if (err.isError())
    throw std::runtime_error(err.getError());

  (void)userData.release();
}

void Python::DeviceStream::waitForHostValue(const CompletionFlag &flag,
                                            uint64_t value) const {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::DeviceStreamRef> mlrtStream = stream->ctx->stream();
  if (mlrtStream.isError())
    throw std::runtime_error(mlrtStream.getError());

  ErrorOrSuccess err =
      (*mlrtStream)->enqueueWaitOnHostValue(*flag.ref(), value);
  if (err.isError())
    throw std::runtime_error(err.getError());
}

void Python::DeviceStream::waitForDevice(const Device &device) const {
  nb::gil_scoped_release release;
  ErrorOrSuccess err = stream->ctx->enqueueWaitFor(*device.ctx);
  if (err.isError())
    throw std::runtime_error(err.getError());
}

uintptr_t Python::DeviceStream::nativeStreamHandle() const {
  // Mirrors getAsString's extraction: ask the device context for its native
  // stream and return the raw pointer. 0 when there is none (e.g. a CPU
  // device), which native callers treat as "no stream". Non-zero means a
  // CUstream / hipStream_t to callers, so other backends report 0.
  M::Driver::Device::API api = stream->ctx->device().api();
  if (api != M::Driver::Device::cudaAPI && api != M::Driver::Device::hipAPI)
    return 0;
  auto err = stream->ctx->stream();
  if (err.isError())
    return 0;
  return reinterpret_cast<uintptr_t>((*err)->ptr());
}

Python::PyDeviceTy Python::DeviceStream::getDevice() const { return dev; }

Python::PyDeviceStreamTy Python::DeviceStream::createStream(Device &device) {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::DeviceContextRef> newCtx =
      device.ctx->device().createContext();
  if (newCtx.isError())
    throw std::runtime_error(newCtx.getError());

  return std::make_unique<DeviceStream>(device, *newCtx);
}

bool Python::DeviceStream::operator==(const DeviceStream &other) const {
  return *this->stream->ctx == *other.stream->ctx;
}

/// Helper to return the Python device object from dlpack device type and id.
std::tuple<std::shared_ptr<Python::Device>, std::optional<intptr_t>>
Python::getPyDevice(DLDeviceType deviceType, ssize_t deviceID) {
  if (deviceType == DLDeviceType::kDLCPU)
    return {Python::getCpuDevice(deviceID), std::nullopt};

  std::shared_ptr<Device> device = Python::getAccelerator(deviceID);
  auto &ctxDev = device->ctx->device();

  if (static_cast<DLDeviceType>(ctxDev.getDLDevice(false).device_type) !=
      deviceType)
    throw std::runtime_error(
        "unsupported device type in dlpack implementation");

  return {device, device->ctx->getDLPackStream()};
}

/// Helper to return the Python device object from a dlpack array.
std::tuple<std::shared_ptr<Python::Device>, std::optional<intptr_t>>
Python::getPyDevice(const nb::object &arrayOrTensor) {
  auto [deviceType, deviceID] = DLPack::callDLPackDevice(arrayOrTensor);
  return getPyDevice(deviceType, deviceID);
}

namespace {
// The accelerator cache stores `weak_ptr`s so that GPU devices can be properly
// released. Otherwise, some resources owned by the device are held too long,
// and can cause segfaults if multiple models are run one after another.
template <class DeviceType>
class WeakDeviceCache {
public:
  std::shared_ptr<DeviceType> find(ssize_t key) {
    // If we have a cache entry that hasn't been freed yet, we just return it.
    if (auto it = map.find(key); it != map.end())
      if (std::shared_ptr<DeviceType> ptr = it->second.lock())
        return ptr;
    return nullptr;
  }

  std::shared_ptr<DeviceType> insert(ssize_t key, M::DeviceSpec spec,
                                     M::Driver::DeviceContextRef ctx) {
    auto shared = std::make_shared<DeviceType>(std::move(spec), std::move(ctx));
    // We use insert_or_assign to overwrite potential freed entries.
    map.insert_or_assign(key, shared);
    return shared;
  }

private:
  llvm::SmallDenseMap<ssize_t, std::weak_ptr<DeviceType>> map;
};

// The CPU cache stores `shared_ptr`s so that CPU devices (and the AsyncRT
// cpuDevice they hold) remain alive for the lifetime of the process.
template <class DeviceType>
class StrongDeviceCache {
public:
  std::shared_ptr<DeviceType> find(ssize_t key) {
    if (auto it = map.find(key); it != map.end())
      return it->second;
    return nullptr;
  }

  std::shared_ptr<DeviceType> insert(ssize_t key, M::DeviceSpec spec,
                                     M::Driver::DeviceContextRef ctx) {
    auto shared = std::make_shared<DeviceType>(std::move(spec), std::move(ctx));
    map.insert_or_assign(key, shared);
    return shared;
  }

private:
  llvm::SmallDenseMap<ssize_t, std::shared_ptr<DeviceType>> map;
};

using CPUCache = StrongDeviceCache<Python::CPU>;
using AcceleratorCache = WeakDeviceCache<Python::Accelerator>;
} // namespace

/// Protects the cache returned by `getCPUCache`.
static std::mutex &getCPUCacheMutex() {
  static std::mutex cacheMutex;
  return cacheMutex;
}

/// Users must acquire the `CPUCacheMutex` first;
static CPUCache &getCPUCache() {
  // Intentionally leaked: the CPU cache holds `shared_ptr<Python::CPU>`, which
  // in turn keeps the AsyncRT cpuDevice alive. At process exit, static-local
  // destructors across translation units run in reverse-of-construction order.
  // The global cpuDevice mutex (in libAsyncRTRuntimeGlobals.dylib) is
  // constructed *after* this cache (on first CpuDeviceContext creation), so it
  // is destroyed *first*. If the cache were then destroyed normally, releasing
  // its shared refs would run `CPUDevice::~Runtime` →
  // `clearGlobalCPUDevicePointerIfEquals`, which locks the already-destroyed
  // mutex — on macOS that returns EINVAL and std::mutex throws, terminating the
  // process (GEX-3583). Leaking the cache matches the stated intent that CPU
  // devices live for the lifetime of the process.
  static CPUCache *cache = new CPUCache{};
  return *cache;
}

namespace {
using DeviceContextPair = std::pair<M::DeviceSpec, M::Driver::DeviceContextRef>;

/// Builds the CPU (spec, context) pair `M_createCPUDevice` used to produce.
/// The spec's id stays 0 for every numa node, matching the wrapper; only the
/// cache keys on the requested id.
ErrorOr<DeviceContextPair> makeCPUDeviceContext() {
  ErrorOr<TargetInfo> targetOr = M::getHostTargetInfo();
  if (targetOr.isError())
    return targetOr.takeError();

  ErrorOr<M::Driver::DeviceContextRef> ctxOr =
      M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
  if (ctxOr.isError())
    return ctxOr.takeError();

  M::DeviceSpec spec;
  spec.ref.id = 0;
  spec.ref.label = kCPULabel;
  spec.target = targetOr.takeValue();
  return DeviceContextPair{std::move(spec), ctxOr.takeValue()};
}

/// Builds the accelerator (spec, context) pair `M_createAcceleratorDevice`
/// used to produce.
ErrorOr<DeviceContextPair> makeAcceleratorDeviceContext(ssize_t deviceId) {
  ErrorOr<M::Driver::DeviceContextRef> ctxOr =
      M::Driver::DeviceContext::getOrCreate(M::Driver::Device::gpuAPI,
                                            deviceId >= 0 ? deviceId : 0);
  if (ctxOr.isError())
    return ctxOr.takeError();

  M::DeviceSpec spec;
  spec.ref.label = kGPULabel;
  spec.ref.id = deviceId;
  return DeviceContextPair{std::move(spec), ctxOr.takeValue()};
}
} // namespace

std::shared_ptr<Python::CPU> Python::getCpuDevice(ssize_t numaId) {
  numaId = numaId >= 0 ? numaId : 0;
  CPUCache &cache = getCPUCache();
  {
    std::lock_guard<std::mutex> lock(getCPUCacheMutex());
    if (auto maybeDevice = cache.find(numaId))
      return maybeDevice;
  }

  ErrorOr<DeviceContextPair> deviceOr = [&] {
    nb::gil_scoped_release release;
    return makeCPUDeviceContext();
  }();
  if (deviceOr.isError()) {
    std::string errorMessage =
        "failed to create device: "s + deviceOr.getError();
    throw std::invalid_argument(errorMessage);
  }
  DeviceContextPair created = deviceOr.takeValue();

  ErrorOrSuccess compatibleOr = [&] {
    nb::gil_scoped_release release;
    return created.second->device().isCompatible();
  }();
  if (compatibleOr.isError())
    throw std::runtime_error(compatibleOr.getError());

  std::lock_guard<std::mutex> lock(getCPUCacheMutex());
  return cache.insert(numaId, std::move(created.first),
                      std::move(created.second));
}

/// Protects the cache returned by `getAcceleratorCache`.
static std::mutex &getAcceleratorCacheMutex() {
  static std::mutex cacheMutex;
  return cacheMutex;
}

/// Users must acquire the `acceleratorCacheMutex` first;
static AcceleratorCache &getAcceleratorCache() {
  static AcceleratorCache cache{};
  return cache;
}

std::shared_ptr<Python::Accelerator> Python::getAccelerator(ssize_t gpuId) {
  gpuId = gpuId >= 0 ? gpuId : 0;
  AcceleratorCache &cache = getAcceleratorCache();
  {
    std::lock_guard<std::mutex> lock(getAcceleratorCacheMutex());
    if (auto maybeDevice = cache.find(gpuId))
      return maybeDevice;
  }

  ErrorOr<DeviceContextPair> deviceOr = [&] {
    nb::gil_scoped_release release;
    return makeAcceleratorDeviceContext(gpuId);
  }();
  if (deviceOr.isError()) {
    std::string errorMessage =
        "failed to create device: "s + deviceOr.getError();
    throw std::invalid_argument(errorMessage);
  }
  DeviceContextPair created = deviceOr.takeValue();

  ErrorOrSuccess compatibleOr = [&] {
    nb::gil_scoped_release release;
    return created.second->device().isCompatible();
  }();
  if (compatibleOr.isError())
    throw std::runtime_error(compatibleOr.getError());

  std::lock_guard<std::mutex> lock(getAcceleratorCacheMutex());
  return cache.insert(gpuId, std::move(created.first),
                      std::move(created.second));
}

// Returns number of accelerators available.
int Python::acceleratorCount() {
  nb::gil_scoped_release release;
  return M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);
}

size_t Python::Device::hash() const {
  std::size_t h1 = std::hash<std::string>{}(this->getLabel());
  std::size_t h2 = std::hash<uint64_t>{}(this->getId());
  return h1 ^ (h2 << 1);
}
