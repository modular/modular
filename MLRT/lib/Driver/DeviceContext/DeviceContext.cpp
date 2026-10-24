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
// This file implements the CPU device context.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

// If the -U ever stops beating Driver/DeviceContextAPI's propagated define, the
// legacy headers below fail with a cascade of unrelated-looking errors. Fail
// first, naming the cause.
static_assert(
    !MODULAR_HAL_DEVICE_CONTEXT,
    "the legacy DeviceContext implementation must compile against the "
    "legacy declarations: -UMODULAR_HAL_DEVICE_CONTEXT in this "
    "target's copts must override Driver/DeviceContextAPI's define");

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "MLRT/Driver/DeviceContext/CPU/CpuDeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "CUDA/CUDACompilationDevice.h"
#include "CUDA/CUDADeviceContext.h"
#include "HIP/HIPCompilationDevice.h"
#include "HIP/HIPDeviceContext.h"
#include "MLRT/Driver/DeviceContext/CPU/CpuDeviceContext.h"
#include "MemoryManager.h"
#include "Metal/MetalDeviceContext.h"
#include "Support/AlignedAlloc.h"
#include "Support/AssertStream.h"
#include "Support/ErrorOr.h"
#include "Support/HashUtils.h"
#include "Support/SymbolExport.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "Target/DriverPlugin.h"
#include "Virtual/VirtualDeviceContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string.h>
#include <utility>

namespace M::Driver {

// Thread-local pointer to the currently bound device context.
static thread_local const DeviceContext *currentlyBoundContext = nullptr;

// Fast-path flag for allPeerAccessEnabled(). Set to non-zero when all
// peer access is confirmed enabled, avoiding the need to re-query
// every device on subsequent calls.
static std::atomic<bool> g_allPeerAccessEnabled{false};

// Forwarded, not stored: this TU is statically linked into several images
// (libmax, libAsyncRTMojoBindings), and a string held here would exist once
// per image under Mach-O two-level binding. See VirtualDeviceGlobals.h.
MODULAR_CXX_EXPORT void setVirtualCpuTarget(StringRef cpu) {
  AsyncRT::setVirtualCpuTarget(cpu);
}

MODULAR_CXX_EXPORT std::string getVirtualCpuTarget() {
  return AsyncRT::getVirtualCpuTarget();
}

DeviceContextScope::DeviceContextScope(const DeviceContext &ctx)
    : prevCtx_(nullptr), err_(success()) {
  Device &dev = ctx.device();
  ErrorOr<void *> errOrCtx = dev.getCurrentDriverContext();
  if (errOrCtx.isError()) {
    err_ = errOrCtx.takeError();
    return;
  }

  void *prevCtx = errOrCtx.takeValue();
  void *newCtx = ctx.driverContext();

  if (prevCtx == newCtx) {
    // Desired context is already current. Nothing to be done on restore, so we
    // keep prevCtx_ and device_ uninitialized.
    return;
  }

  ErrorOrSuccess err = dev.setCurrentDriverContext(newCtx);
  if (err.isError()) {
    err_ = err.takeError();
    return;
  }

  // Remember values to be restored in destructor.
  device_ = DeviceRef::copy(&dev);
  prevCtx_ = prevCtx;
}

DeviceContextScope::~DeviceContextScope() {
  if (device_.getPointer() == nullptr) {
    assert(prevCtx_ == nullptr);
    return;
  }
  ErrorOrSuccess ignore = device_->setCurrentDriverContext(prevCtx_);
}

ErrorOrSuccess DeviceContextScope::err() const {
  if (!err_.isError())
    return success();
  return err_.copy().takeError();
}

DeviceFunctionCacheEntry::DeviceFunctionCacheEntry(const std::string name)
    : name_(std::move(name)) {}

const std::string &DeviceFunctionCacheEntry::name() const { return name_; }

Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  static bool syncModeEnabled = DeviceContext::syncModeFromConfig();

  llvm::StringLiteral kErrorMessage = "";

  if (!syncModeEnabled)
    kErrorMessage = "\nTo get more accurate error information, set "
                    "MODULAR_DEBUG=device-sync-mode.";

  static bool logVerbose = DeviceContext::configAsBool("log_verbose", false);
  if (logVerbose) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " failed: " + msg + kErrorMessage);
  }
  return Error(msg + kErrorMessage);
}

// TODO: Remove once users have been updated to use
// `Device::preferredAlignment`.
MODULAR_CXX_EXPORT size_t getPreferredAlignment(std::string label) {
  return Device::preferredAlignment(label);
}

//
// ===== Device ===== //
//
Device::Device(API api, std::string name, int id, MemInfo memInfo,
               size_t allocAlignment)
    : name_(std::move(name)), id_(id), api_(api), memInfo_(memInfo),
      allocAlignment_(allocAlignment),
      hostMemoryManager_(MemoryManager::onHost(*this, allocAlignment)) {}

Device::~Device() {
  // Destroy the function map before destroying the context.
  funcMap_.clear();
}

MODULAR_CXX_EXPORT const std::string &Device::name() const { return name_; }

ErrorOrSuccess evaluateGPUHealth(const GPUHealthData &data) {
  // 1. Check hardware throttling.
  if (data.throttleQueried) {
    constexpr unsigned long long hwThrottleMask =
        // NVIDIA
        GPUHealthData::THROTTLE_HW_SLOWDOWN |
        GPUHealthData::THROTTLE_HW_THERMAL_SLOWDOWN |
        GPUHealthData::THROTTLE_HW_POWER_BRAKE |
        // AMD
        GPUHealthData::THROTTLE_PROCHOT |
        GPUHealthData::THROTTLE_SOCKET_THERMAL |
        GPUHealthData::THROTTLE_VR_THERMAL |
        GPUHealthData::THROTTLE_HBM_THERMAL;
    if (data.throttleReasons & hwThrottleMask) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "GPU " << data.deviceId << " (" << data.deviceName
         << ") is hardware-throttled:";
      // NVIDIA throttle reasons.
      if (data.throttleReasons & GPUHealthData::THROTTLE_HW_SLOWDOWN)
        os << " [HW Slowdown]";
      if (data.throttleReasons & GPUHealthData::THROTTLE_HW_THERMAL_SLOWDOWN)
        os << " [HW Thermal Slowdown]";
      if (data.throttleReasons & GPUHealthData::THROTTLE_HW_POWER_BRAKE)
        os << " [HW Power Brake]";
      // AMD throttle reasons.
      if (data.throttleReasons & GPUHealthData::THROTTLE_PROCHOT)
        os << " [PROCHOT]";
      if (data.throttleReasons & GPUHealthData::THROTTLE_SOCKET_THERMAL)
        os << " [Socket Thermal]";
      if (data.throttleReasons & GPUHealthData::THROTTLE_VR_THERMAL)
        os << " [VR Thermal]";
      if (data.throttleReasons & GPUHealthData::THROTTLE_HBM_THERMAL)
        os << " [HBM Thermal]";
      os << ". Set MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK=true to bypass.";
      RETURN_ERROR(msg);
    }
  }

  // 2. Check uncorrectable ECC errors.
  if (data.eccQueried && data.eccErrors > 0) {
    RETURN_ERROR("GPU " + std::to_string(data.deviceId) + " (" +
                 data.deviceName + ") has " + std::to_string(data.eccErrors) +
                 " uncorrectable ECC error(s). "
                 "Set MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK=true to "
                 "bypass.");
  }

  return success();
}

ErrorOr<const char *> Device::getArchName() const {
  return Error(Twine("Not implemented for device: ") + Twine(name_));
}

MODULAR_CXX_EXPORT int Device::id() const { return id_; }

MODULAR_CXX_EXPORT Device::API Device::api() const { return api_; }

ErrorOr<bool> Device::supportsMulticast() const { return false; }

// Unanswered backends report the topology no caller can get wrong.
ErrorOr<bool> Device::isHostUnified() const { return false; }

ErrorOr<Device::GraphMemInfo> Device::getGraphMemoryInfo() const {
  return GraphMemInfo{};
}

ErrorOr<size_t> Device::maxSingleAllocationSize() const {
  DEFINE_OR_RETURN_ERROR(Device::MemInfo, info, getMemoryInfo());
  return info.total;
}

ErrorOr<Device::AllocationRange> Device::allocationRange(void *) const {
  return Error("allocation range queries are not supported on this device");
}

// Default VMM implementations: unsupported. Backends that support VMM (CUDA)
// override `vmmSupported()` and these methods. The MemoryManager never calls
// the methods below unless `vmmSupported()` returned true.
static Error vmmUnsupportedError() {
  return Error("Virtual Memory Management (VMM) is not supported on this "
               "device");
}
ErrorOr<size_t> Device::vmmGranularity() const { return vmmUnsupportedError(); }
ErrorOr<void *> Device::vmmAddressReserve(size_t) {
  return vmmUnsupportedError();
}
ErrorOrSuccess Device::vmmAddressFree(void *, size_t) {
  return vmmUnsupportedError();
}
ErrorOr<uint64_t> Device::vmmCreate(size_t) { return vmmUnsupportedError(); }
ErrorOrSuccess Device::vmmRelease(uint64_t) { return vmmUnsupportedError(); }
ErrorOrSuccess Device::vmmMapAndSetAccess(void *, size_t, uint64_t) {
  return vmmUnsupportedError();
}
ErrorOrSuccess Device::vmmUnmap(void *, size_t) {
  return vmmUnsupportedError();
}

MODULAR_CXX_EXPORT size_t Device::allocationAlignment() const {
  return memoryManager().allocationAlignment();
}

MODULAR_CXX_EXPORT size_t Device::preferredAlignment(StringRef api) {
  if (api == gpuAPI) {
    // TODO(GEX-1416): Remove special hard-coded value to pass tests.
    return 256;
  }
  ErrorOr<DeviceRef> device = create(api, 0);
  if (device.isError()) {
    return M::kPreferredMemoryAlignment;
  }
  return device.get()->allocationAlignment();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceRef> Device::create(StringRef api, int id) {
  // Check if virtual device mode is enabled. If so, create a VirtualDevice
  // instead of a real hardware device.
  if (AsyncRT::isVirtualDeviceMode() &&
      (api == gpuAPI || api == cudaAPI || api == hipAPI || api == metalAPI)) {
    // In virtual device mode, create a lightweight compilation device
    // that can generate code without requiring physical GPU hardware.

    // Get the configured API and target architecture from global config
    std::string virtualAPI = AsyncRT::getVirtualDeviceAPI();
    std::string targetArch = AsyncRT::getVirtualDeviceTargetArch();

    // Validate that both have been set
    if (virtualAPI.empty()) {
      return Error(
          "Virtual device API not set. "
          "Call setVirtualDeviceAPI() with target API "
          "(e.g., 'cuda', 'hip', 'metal') before creating virtual devices.");
    }
    if (targetArch.empty()) {
      return Error(
          "Virtual device target architecture not set. "
          "Call setVirtualDeviceTargetArch() with target GPU architecture "
          "(e.g., 'sm_80', 'sm_90') before creating virtual devices.");
    }

    ErrorOr<std::unique_ptr<CompilationDevice>> compilationDevice =
        CompilationDevice::create(virtualAPI, targetArch);
    if (compilationDevice.isError())
      return compilationDevice.takeError();

    // Create VirtualDevice with the compilation device
    VirtualDeviceRef vref = VirtualDeviceRef::create(
        id, "VirtualGPU-" + std::to_string(id), std::move(*compilationDevice));
    return DeviceRef::copy(vref.getPointer());
  }

  // Only zero and positive values are allowed as device ids.
  if (id < 0) {
    RETURN_ERROR("Device id must be positive. Got " + llvm::Twine(id));
  }
  // Make sure that we are not requesting a device that is out of range. If
  // numberOfDevices(api) returned a zero, then this is an unsupported API and
  // we let the code below issue the correct error.
  int numDevices = numberOfDevices(api);
  if ((numDevices != 0) && (id >= numDevices)) {
    RETURN_ERROR("Device(api: " + api + ", id: " + llvm::Twine(id) +
                 ") is not available. Maximum number of supported devices is " +
                 llvm::Twine(numDevices) + ".");
  }

  // TODO(iposva): Implement dynamic registration mechanism.
  std::string cudaError;
  if ((api == gpuAPI) || (api == cudaAPI)) {
    ErrorOr<CUDADevice &> device = CUDADevice::get(id);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    } else if (api == cudaAPI)
      return device.takeError();
    cudaError = device.getError();
  }

  std::string hipError;
  if ((api == gpuAPI) || (api == hipAPI)) {
    ErrorOr<HIPDevice &> device = HIPDevice::get(id);
    if (!device.isError()) {
      // This disables full buffering when piping to a file if running
      // on AMD GPU. Without setting this `FileCheck` will have CPU
      // prints and GPU kernel prints mixed together, causing the tests
      // to fail.
      setbuf(stdout, nullptr);
      return DeviceRef::copy(&device.get());
    } else if (api == hipAPI)
      return device.takeError();
    hipError = device.getError();
  }

  std::string metalError;
  if ((api == gpuAPI) || (api == metalAPI)) {
#if MODULAR_MACOS

    ErrorOr<MetalDevice &> device = MetalDevice::get(id);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    } else if (api == metalAPI)
      return device.takeError();
    metalError = device.getError();
#endif
  }

  // Generic plugin-based device discovery (no hardcoded API names).
  // Plugins self-identify via property_string("api").
  {
    Plugin::ensurePluginsLoaded();
    if (Plugin::hasPluginForApi(api)) {
      auto deviceOr = Plugin::getDevice(api, id, /*deviceMemoryLimit=*/-1);
      if (!deviceOr.isError())
        return deviceOr;
      else if (api != gpuAPI)
        return deviceOr.takeError();
      // For gpuAPI, fall through to try other backends.
    }
  }

  if (api == cpuAPI) {
    ErrorOr<CpuDevice &> device = CpuDevice::get(id);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  }

  if (api == gpuAPI) {
    return Error(llvm::Twine("No supported \"gpu\" device available.\n") +
                 "  CUDA information: " + cudaError + "\n" +
                 "  HIP information:  " + hipError + "\n" +
                 "  Metal information: " + metalError);
  }
  return Error(llvm::Twine("Unsupported device api: \"") + api + "\"");
}

std::string Device::FunctionCacheKey::moduleCacheName(StringRef moduleName,
                                                      StringRef code) {
  if (!moduleName.empty())
    return moduleName.str();
  // No module name: key on a short stable hash of the code instead of the blob
  // itself. xxhash is non-cryptographic but the inputs are trusted compiler
  // output and the rest of the key (function name, debug/opt/shared params)
  // further disambiguates, so collision risk is negligible.
  M::raw_xxhash128_stream hasher;
  hasher << code;
  return hasher.hashString();
}

DeviceFunctionCacheEntryRef
Device::findInFunctionCache(const FunctionCacheKey &key) const {
  TRACY_ZONE_SCOPED_NC("Device::findInFunctionCache", TRACY_COLOR_PURPLE);
  std::scoped_lock<std::mutex> lock(mu_);
  return funcMap_.lookup(key);
}

DeviceFunctionCacheEntryRef
Device::insertInFunctionCache(const FunctionCacheKey &key,
                              const DeviceFunctionCacheEntry &entry) {
  TRACY_ZONE_SCOPED_NC("Device::insertInFunctionCache", TRACY_COLOR_PURPLE);
  {
    std::scoped_lock<std::mutex> lock(mu_);
    auto iter = funcMap_.find(key);
    if (iter != funcMap_.end()) {
      return iter->second;
    }
    funcMap_.insert({key, DeviceFunctionCacheEntryRef::copy(&entry)});
    return DeviceFunctionCacheEntryRef::copy(&entry);
  }
}

void Device::clearFunctionCache() const {
  std::scoped_lock<std::mutex> lock(mu_);
  funcMap_.clear();
}

constexpr void *buffer_pointer_for(void *ptr, size_t byteSize) {
  if (byteSize == 0) {
    return DeviceBuffer::emptyBufferPointer();
  }
  return ptr;
}

namespace {
// Invokes `freeFn` once, from its destructor, when its last reference drops. A
// host buffer's destructor holds a shared_ptr to this token inside one no-op
// event handler per stream that used the buffer, so the free runs only after
// every such stream's event has fired and released its handler. Templated on
// the callable to avoid `std::function` type erasure; build via
// `makeDeferredHostFree`.
template <typename FreeFn>
struct DeferredHostFree {
  explicit DeferredHostFree(FreeFn freeFn) : freeFn(std::move(freeFn)) {}
  ~DeferredHostFree() { freeFn(); }

  FreeFn freeFn;
};

template <typename FreeFn>
std::shared_ptr<DeferredHostFree<FreeFn>> makeDeferredHostFree(FreeFn freeFn) {
  return std::make_shared<DeferredHostFree<FreeFn>>(std::move(freeFn));
}
} // namespace

//
// ===== DeviceBuffer ===== //
//
// `WRAPPED` only ever arrives through the constructor taking an explicit kind.
static constexpr Device::FreeKind freeKindFor(bool isHost) {
  return isHost ? Device::FreeKind::HOST : Device::FreeKind::DEVICE;
}

DeviceBuffer::DeviceBuffer(DeviceContext &context, void *ptr, size_t byteSize,
                           bool isHost)
    : context_(DeviceContextRef::copy(&context)),
      ptr_(buffer_pointer_for(ptr, byteSize)), byteSize_(byteSize),
      isHost_(isHost), freeKind_(freeKindFor(isHost)) {}
DeviceBuffer::DeviceBuffer(DeviceContext &context, DeviceBuffer &parent,
                           void *ptr, size_t byteSize, bool isHost)
    : context_(DeviceContextRef::copy(&context)),
      parent_(DeviceBufferRef::copy(&parent)),
      ptr_(buffer_pointer_for(ptr, byteSize)), byteSize_(byteSize),
      isHost_(isHost), freeKind_(freeKindFor(isHost)) {}
// TODO: Remove the V1 compatibility.
DeviceBuffer::DeviceBuffer(DeviceContext &context, void *ptr, size_t byteSize,
                           bool isHost, bool owning)
    : context_(DeviceContextRef::copy(&context)),
      ptr_(buffer_pointer_for(ptr, byteSize)), byteSize_(byteSize),
      isHost_(isHost), freeKind_(freeKindFor(isHost)), owning_(owning) {}
DeviceBuffer::DeviceBuffer(DeviceContext &context, void *ptr, size_t byteSize,
                           Device::FreeKind kind)
    : context_(DeviceContextRef::copy(&context)),
      ptr_(buffer_pointer_for(ptr, byteSize)), byteSize_(byteSize),
      isHost_(kind == Device::FreeKind::HOST ||
              kind == Device::FreeKind::WRAPPED),
      freeKind_(kind) {}

MODULAR_CXX_EXPORT DeviceBuffer::~DeviceBuffer() {
  if (!owning_) {
    // TODO: Remove the V1 compatibility.
    // This is a device buffer created by pointer in Mojo while ownership is
    // maintained elsewhere.
    return;
  }
  if ((parent() != nullptr) || (ptr_ == nullptr)) {
    // Parent owns the memory.
    return;
  }
  // Call custom destructor if set.
  if (destructorFn_) {
    (*destructorFn_)();
    ptr_ = nullptr;
    return;
  }
  // If byte size is 0, we don't need to call free
  if (byteSize_ == 0) {
    return;
  }
  // Release the GPU memory now or enqueue an event based release for host
  // memory. Host memory can only be released once the stream of execution on
  // the corresponding device has completed all currently pending events.
  DeviceContext &ctx = context();
  // DEBUGGING AID
  // RECORD_PENDING_ERROR(ctx, ctx.enqueueSetMemory(*this, 0xf3, 1));
  // \DEBUGGING AID
  RECORD_PENDING_ERROR(ctx, ctx.pollPendingEvents());
  if (!isHost()) {
    RECORD_PENDING_ERROR(ctx, ctx.freeList().free(ptr_));
  } else {
    // Host memory can only be released once every stream that DMA'd into or
    // out of it has finished, not just the owning stream: a pinned staging
    // buffer is routinely the source/destination of async copies on several
    // GPU streams (e.g. multi-GPU KV-cache offload). Freeing against the owning
    // stream alone races copies still in flight on the others.
    //
    // Following PyTorch's caching pinned allocator, enqueue a no-op event
    // handler on the owning context and every other context the buffer was
    // used on; each holds a reference to `deferredFree`, which runs the free
    // from its destructor once the last handler completes.
    //
    // A wrapped range needs the same ordering, but the host free list never
    // saw these pages, so it unregisters instead.
    //
    // TODO(pprovins): opt-in precise wait. Queueing makes the caller
    // synchronize every context the range was copied on; blocking on the events
    // below would wait exactly. Do it once a wrap drops in a loop.
    auto deferredFree = makeDeferredHostFree([contextRef =
                                                  DeviceContextRef::copy(&ctx),
                                              ptr = ptr_, kind = freeKind_]() {
      if (kind == Device::FreeKind::WRAPPED) {
        RECORD_PENDING_ERROR(*contextRef, contextRef->memFree(ptr, kind));
        return;
      }
      RECORD_PENDING_ERROR(*contextRef, contextRef->hostFreeList().free(ptr));
    });
    // Refcount is zero here, so no thread can record a new used context
    // concurrently; the lock only publishes `recordUsedOnContext`'s writes.
    // That function takes only `usedContextsMu_`, never `pendingWorkMutex_`, so
    // enqueuing under the lock cannot deadlock.
    std::lock_guard<std::mutex> lock(usedContextsMu_);
    // Order the free against the owning context...
    RECORD_PENDING_ERROR(ctx, ctx.enqueueEventHandler([deferredFree]() {}));
    // ...and every other stream the buffer was used on.
    for (const DeviceContextRef &usedCtx : usedContexts_) {
      RECORD_PENDING_ERROR(*usedCtx,
                           usedCtx->enqueueEventHandler([deferredFree]() {}));
    }
  }
}

void DeviceBuffer::recordUsedOnContext(DeviceContext &ctx) {
  // Only host/pinned buffers use the deferred, event-ordered host free that
  // must account for foreign streams; device buffers free synchronously.
  if (!isHost_)
    return;
  // Sub-buffers alias their root parent's backing memory and their destructor
  // returns early -- only the root owner runs the deferred host free. Forward
  // the recording up the parent chain so the root orders its free against
  // `ctx` too; recording it here would be lost when the sub-buffer is
  // destroyed.
  if (DeviceBuffer *parent = parent_.getPointer()) {
    parent->recordUsedOnContext(ctx);
    return;
  }
  // The owning context is always ordered against in the destructor.
  if (&ctx == context_.getPointer())
    return;
  std::lock_guard<std::mutex> lock(usedContextsMu_);
  // The set of distinct contexts is tiny (bounded by the device count), so a
  // linear de-dup keeps the vector from growing under repeated copies.
  for (const DeviceContextRef &existing : usedContexts_)
    if (existing.getPointer() == &ctx)
      return;
  usedContexts_.push_back(DeviceContextRef::copy(&ctx));
}

MODULAR_CXX_EXPORT DeviceContext &DeviceBuffer::context() { return *context_; }

MODULAR_CXX_EXPORT const DeviceContext &DeviceBuffer::context() const {
  return *context_;
}

const DeviceBuffer *DeviceBuffer::parent() const {
  return parent_.getPointer();
}

MODULAR_CXX_EXPORT void *DeviceBuffer::ptr() const { return ptr_; }

// TODO: Remove the V1 compatibility.
MODULAR_CXX_EXPORT void *DeviceBuffer::releasePtr() {
  auto tmp = ptr_;
  ptr_ = nullptr;
  owning_ = false;
  return tmp;
}

MODULAR_CXX_EXPORT size_t DeviceBuffer::byteSize() const { return byteSize_; }

MODULAR_CXX_EXPORT bool DeviceBuffer::isHost() const { return isHost_; }

void *DeviceBuffer::emptyBufferPointer() {
  return reinterpret_cast<void *>(-1);
}

void DeviceBuffer::setDestructorFn(llvm::unique_function<void()> destructor) {
  destructorFn_ = std::move(destructor);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::copyTo(DeviceBuffer &dst,
                                                       size_t srcByteOffset,
                                                       size_t srcByteSize) {
  DeviceBuffer &src = *this;
  return dst.context().enqueueCopyDeviceToDevice(dst, src, srcByteOffset,
                                                 srcByteSize);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::copyTo(void *dst) {
  DeviceBuffer &src = *this;
  return src.context().enqueueCopyFromDevice(dst, src);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::copyFrom(const void *src) {
  DeviceBuffer &dst = *this;
  return dst.context().enqueueCopyToDevice(dst, src);
}

MODULAR_CXX_EXPORT void DeviceBuffer::setContext(DeviceContext &ctx) {
  context_ = DeviceContextRef::copy(&ctx);
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceBuffer::createSubBuffer(size_t byteOffset, size_t byteSize) {
  char *origPtr = static_cast<char *>(this->ptr());
  size_t origByteSize = this->byteSize();

  if ((byteSize > origByteSize) || (origByteSize - byteSize < byteOffset)) {
    RETURN_ERROR("offset, len and elem_size are out of range");
  }
  // Defaults to `origPtr + byteOffset`; backends whose runtime only accepts
  // registered pointers return the registered slice address instead.
  DEFINE_OR_RETURN_ERROR(
      void *, subPtr,
      context().device().registerBufferSlice(origPtr, byteOffset, byteSize));
  return DeviceBufferRef::take(
      new DeviceBuffer(context(), *this, subPtr, byteSize, isHost()));
}

ErrorOr<void *> Device::registerBufferSlice(void *parentPtr, size_t byteOffset,
                                            size_t /*byteSize*/) {
  return static_cast<char *>(parentPtr) + byteOffset;
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceBuffer::synchronize(void *external_stream) {
  // Regular synchronization
  if (!external_stream) {
    return context_->synchronize();
  }

  // Synchronization with an external stream
  DEFINE_OR_RETURN_ERROR(DeviceStreamRef, extStream,
                         context_->createExternalStream(external_stream));
  // Create an event on the buffer's stream
  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, context_->enqueueEvent());
  // Enqueue a wait for the event on the external stream
  RETURN_IF_ERROR(extStream->enqueueWaitForEvent(*event));

  return success();
}

//
// ===== DeviceMulticastBuffer ===== //
//
DeviceMulticastBuffer::DeviceMulticastBuffer(size_t byteSize)
    : byteSize_(byteSize) {}

DeviceMulticastBuffer::~DeviceMulticastBuffer() {}

size_t DeviceMulticastBuffer::byteSize() const { return byteSize_; }

ErrorOr<DeviceMulticastBufferRef>
DeviceMulticastBuffer::allocate(const std::vector<DeviceContextRef> &contexts,
                                size_t byteSize) {
  std::vector<const Device *> devices;
  for (const auto &ctx : contexts) {
    if (!devices.empty()) {
      if (devices[0]->api() != ctx->device().api()) {
        RETURN_ERROR("All contexts need to use the same API.");
      }
    }
    // Check for duplicate entries.
    for (const auto &dev : devices) {
      if (dev->id() == ctx->device().id()) {
        RETURN_ERROR("Duplicate context on a device.");
      }
    }
    // Check that multicast is supported on this particular device.
    DEFINE_OR_RETURN_ERROR(bool, multicastSupported,
                           ctx->device().supportsMulticast());
    if (!multicastSupported)
      RETURN_ERROR("Multicast buffer not supported.");

    // Add to the set of devices participating in this multicast allocation.
    devices.emplace_back(&ctx->device());
  }
  if (devices.size() < 2) {
    RETURN_ERROR("At least two devices necessary.");
  }
  if (devices[0]->api() == CUDADevice::api) {
    return CUDADeviceMulticastBuffer::allocate(devices, byteSize);
  }
  RETURN_ERROR("Multicast only supported on CUDA devices.");
}

//
// ===== SymmetricPool ===== //
//
namespace {
/// Sorted device ids, so any ordering of the same devices finds one pool.
/// Plain ids suffice because allocate() only accepts all-CUDA context lists.
using PoolKey = std::vector<int>;

PoolKey makePoolKey(const std::vector<DeviceContextRef> &contexts) {
  PoolKey key;
  key.reserve(contexts.size());
  for (const auto &ctx : contexts)
    key.push_back(ctx->device().id());
  llvm::sort(key);
  return key;
}

std::mutex &poolRegistryMutex() {
  static std::mutex mu;
  return mu;
}

std::map<PoolKey, DeviceMulticastBufferRef> &poolRegistry() {
  static std::map<PoolKey, DeviceMulticastBufferRef> registry;
  return registry;
}
} // namespace

ErrorOr<DeviceMulticastBufferRef>
SymmetricPool::create(const std::vector<DeviceContextRef> &contexts,
                      size_t byteSize) {
  if (byteSize == 0)
    RETURN_ERROR("Symmetric pool capacity must be non-zero.");

  PoolKey key = makePoolKey(contexts);
  std::lock_guard<std::mutex> lock(poolRegistryMutex());
  if (poolRegistry().find(key) != poolRegistry().end())
    RETURN_ERROR("Symmetric pool already created for this device set.");

  DEFINE_OR_RETURN_ERROR(DeviceMulticastBufferRef, pool,
                         DeviceMulticastBuffer::allocate(contexts, byteSize));

  poolRegistry().emplace(std::move(key), pool);
  return pool;
}

DeviceMulticastBufferRef
SymmetricPool::lookup(const std::vector<DeviceContextRef> &contexts) {
  PoolKey key = makePoolKey(contexts);
  std::lock_guard<std::mutex> lock(poolRegistryMutex());
  auto it = poolRegistry().find(key);
  if (it == poolRegistry().end())
    return {};
  return it->second;
}

//
// ===== DeviceEvent ===== //
//
DeviceEvent::DeviceEvent(Device &device) : device_(DeviceRef::copy(&device)) {}

MODULAR_CXX_EXPORT Device &DeviceEvent::device() const { return *device_; }

ErrorOr<float> DeviceEvent::elapsedTime(const DeviceEvent &end) const {
  RETURN_ERROR("event elapsed time is not supported on this device");
}

//
// ===== DeviceGraphCollection ===== //
//

void DeviceGraphCollection::setOutputs(
    ArrayRef<AsyncRT::AnyAsyncValueRef> values) {
  ASSERT_STREAM(outputs.empty(), "DeviceGraphCollection already has values");
  outputs = llvm::map_to_vector(values, [](auto &ref) { return ref.copy(); });
}

ArrayRef<AsyncRT::AnyAsyncValueRef> DeviceGraphCollection::getOutputs() const {
  return outputs;
}

ErrorOrSuccess DeviceGraphCollection::replay() const {
  for (const AsyncDeviceGraphRef &graph : graphs) {
    if (auto err = AsyncRT::awaitOrError(graph))
      return err.takeError();
    if (auto err = (*graph)->replay())
      return err;
  }
  return success();
}

int64_t DeviceGraphCollection::numGraphs() const {
  return static_cast<int64_t>(graphs.size());
}

/// RAII type for managing the capture state of a stream.
class DeviceGraphCaptureContext {
public:
  DeviceGraphCaptureContext(const DeviceGraphCaptureContext &other) = delete;
  DeviceGraphCaptureContext &
  operator=(const DeviceGraphCaptureContext &) = delete;
  DeviceGraphCaptureContext(DeviceGraphCaptureContext &&other) = default;
  DeviceGraphCaptureContext &operator=(DeviceGraphCaptureContext &&) = default;

  /// Initiate capture on a particular stream.
  static ErrorOr<DeviceGraphCaptureContext>
  beginCapture(const DeviceStreamRef &stream) {
    if (auto err = stream->beginCapture())
      return err.takeError();
    stream->beginLaunchTrace();
    return DeviceGraphCaptureContext(stream);
  }

  static DeviceGraphCaptureContext beginTrace(const DeviceStreamRef &stream) {
    stream->beginLaunchTrace();
    DeviceGraphCaptureContext ctx(stream);
    ctx.isCapture = false;
    return ctx;
  }

  /// End capture and return the resulting device graph.
  ErrorOr<DeviceGraphRef> endCapture() {
    ErrorOr<DeviceGraphRef> result = stream->endCapture();

    if (result.isError()) {
      (void)stream->takeLaunchTrace();
      stream.reset();
      return result.takeError();
    }

    // Take trace BEFORE moving stream (otherwise stream is null after move)
    std::vector<KernelLaunchMetadata> trace = stream->takeLaunchTrace();
    result.get()->setStream(std::move(stream));
    result.get()->setLaunchTrace(std::move(trace));
    return result;
  }

  std::vector<KernelLaunchMetadata> endTrace() {
    auto result = stream->takeLaunchTrace();
    stream.reset();
    return result;
  }

  ~DeviceGraphCaptureContext() {
    if (stream) {
      if (isCapture)
        (void)stream->endCapture();
      (void)stream->takeLaunchTrace();
    }
  }

private:
  DeviceGraphCaptureContext(const DeviceStreamRef &ref) : stream(ref.copy()) {}

  /// The stream on which capture is occurring (can be nullptr).
  DeviceStreamRef stream;

  /// Are we performing a capture or just tracing execution?
  bool isCapture = true;
};

std::string LaunchTraceDiff::formatErrorMessage(size_t graphIdx) const {
  std::string errorMsg;
  llvm::raw_string_ostream os(errorMsg);

  os << "Graph " << graphIdx << " launch trace mismatch:\n";

  if (!missingOps.empty()) {
    os << "  Missing operations (in baseline but not in current):\n";
    for (const auto &[op, count] : missingOps) {
      os << op.format(count);
    }
  }

  if (!unexpectedOps.empty()) {
    os << "  Unexpected operations (in current but not in baseline):\n";
    for (const auto &[op, count] : unexpectedOps) {
      os << op.format(count);
    }
  }

  return os.str();
}

llvm::json::Value LaunchTraceDiff::formatJSON(size_t graphIdx) const {
  llvm::json::Object json;

  json["graphIdx"] = static_cast<int64_t>(graphIdx);
  json["hasMismatch"] = !empty();

  // Format missing operations
  llvm::json::Array missingArray;
  for (const auto &[op, count] : missingOps) {
    llvm::json::Object opJson;
    opJson["operation"] = op.formatJSON();
    opJson["count"] = count;
    missingArray.push_back(std::move(opJson));
  }
  json["missingOps"] = std::move(missingArray);

  // Format unexpected operations
  llvm::json::Array unexpectedArray;
  for (const auto &[op, count] : unexpectedOps) {
    llvm::json::Object opJson;
    opJson["operation"] = op.formatJSON();
    opJson["count"] = count;
    unexpectedArray.push_back(std::move(opJson));
  }
  json["unexpectedOps"] = std::move(unexpectedArray);

  return llvm::json::Value(std::move(json));
}

/// Compute the difference between two launch traces.
/// Counts occurrences of each operation and reports differences in counts.
static LaunchTraceDiff
computeLaunchTraceDiff(ArrayRef<KernelLaunchMetadata> currentTrace,
                       ArrayRef<KernelLaunchMetadata> baseline) {
  LaunchTraceDiff diff;

  // Using std::map to ensure ordering.
  // KernelLaunchMetadata orders based on kernel name.

  // Count occurrences in baseline
  std::map<KernelLaunchMetadata, int> baselineCounts;
  for (const KernelLaunchMetadata &op : baseline)
    baselineCounts[op]++;

  // Count occurrences in current
  std::map<KernelLaunchMetadata, int> currentCounts;
  for (const KernelLaunchMetadata &op : currentTrace)
    currentCounts[op]++;

  // Find operations with different counts
  // First, check for operations in baseline that are missing or fewer in
  // current
  for (const auto &[op, count] : baselineCounts) {
    auto it = currentCounts.find(op);
    int currentCount = (it != currentCounts.end()) ? it->second : 0;
    if (currentCount < count)
      diff.missingOps[op] = count - currentCount;
  }

  // Then, check for operations in current that exceed baseline
  for (const auto &[op, count] : currentCounts) {
    auto it = baselineCounts.find(op);
    int baselineCount = (it != baselineCounts.end()) ? it->second : 0;
    if (count > baselineCount)
      diff.unexpectedOps[op] = count - baselineCount;
  }

  return diff;
}

ErrorOrSuccess DeviceGraphCollection::debugVerifyReplay(
    llvm::function_ref<ErrorOrSuccess()> execute) const {
  // Step 1: Validate graphs and begin tracing on all streams
  SmallVector<DeviceGraphCaptureContext> traceContexts;
  for (const AsyncDeviceGraphRef &graph : graphs) {
    if (auto err = AsyncRT::awaitOrError(graph))
      return err.takeError();
    if (!(*graph)->hasCaptureStream())
      RETURN_ERROR("No capture stream stored for graph");
    if (!(*graph)->hasBaselineLaunchTrace())
      RETURN_ERROR("No baseline launch trace for graph");

    traceContexts.push_back(
        DeviceGraphCaptureContext::beginTrace((*graph)->captureStreamRef()));
  }

  // Step 2: Execute once for all devices
  ErrorOrSuccess execResult = execute();

  if (execResult.isError())
    return execResult.takeError();

  // Step 3: Extract traces and compare to baseline for each graph
  for (size_t graphIdx : llvm::seq(graphs.size())) {
    std::vector<KernelLaunchMetadata> currentTrace =
        traceContexts[graphIdx].endTrace();

    // Step 4: Compare with baseline (as sorted permutations to handle
    // non-deterministic execution order)
    ArrayRef<KernelLaunchMetadata> baseline =
        (*graphs[graphIdx])->baselineLaunchTrace();

    // Compute the diff between current trace and baseline
    LaunchTraceDiff diff = computeLaunchTraceDiff(currentTrace, baseline);

    // If there are any differences, report them
    if (!diff.empty()) {
      return Error(diff.formatErrorMessage(graphIdx));
    }
  }

  if (execResult.isError())
    return execResult.takeError();

  return success();
}

//
// ===== DeviceGraphMemoryPool ===== //
//

DeviceGraphMemoryPool::DeviceGraphMemoryPool(DeviceContextRef context)
    : context(context) {
  Device &device = context->device();

  static std::atomic<int64_t> counter = 0;

  int64_t id = counter.fetch_add(1);
  deviceFreeList = std::make_unique<MemoryManagerFreeList>(
      context.getPointer(), device.memoryManager(),
      device.memoryManager().freeList(),
      /*disableFlush=*/true,
      /*label=*/
      (Twine("DeviceGraphMemoryPool(id=") + Twine(id) + ", device)").str());

  hostFreeList = std::make_unique<MemoryManagerFreeList>(
      context.getPointer(), device.hostMemoryManager(),
      device.hostMemoryManager().freeList(),
      /*disableFlush=*/true,
      /*label=*/
      (Twine("DeviceGraphMemoryPool(id=") + Twine(id) + ", host)").str());
}

DeviceGraphMemoryPool::~DeviceGraphMemoryPool() = default;

Device &DeviceGraphMemoryPool::device() const { return context->device(); }

ErrorOr<void *> DeviceGraphMemoryPool::allocate(size_t numBytes, bool isHost) {
  // The pool's allocator rejects zero-size requests, and a Mojo `DeviceBuffer`
  // cannot hold a null device pointer.
  if (numBytes == 0)
    return nullptr;

  if (isHost)
    return hostFreeList->allocate(numBytes, Device::AllocMode::HOST);
  return deviceFreeList->allocate(numBytes, Device::AllocMode::SYNC);
}

void DeviceGraphMemoryPool::free(void *ptr, bool isHost) {
  MemoryManagerFreeList &pool = isHost ? *hostFreeList : *deviceFreeList;
  // Callers are `DeviceBuffer` destructors, which cannot propagate a failure,
  // so surface it on the owning context like the non-pool free path does.
  RECORD_PENDING_ERROR(*context, pool.free(ptr));
}

//
// ===== DeviceGraph ===== //
//
/// Shared helper: begin capture on all streams and run the execute lambda.
/// Populates \p captures with per-stream capture contexts.
static ErrorOrSuccess
beginCaptureAndExecute(ArrayRef<DeviceStreamRef> streams,
                       llvm::function_ref<ErrorOrSuccess()> execute,
                       SmallVectorImpl<DeviceGraphCaptureContext> &captures) {
  captures.reserve(streams.size());

  // Single-arena VMM: physically back the free budget BEFORE capture begins.
  // VMM map/unmap is illegal during capture, so allocations made inside the
  // capture window must hit already-mapped granules. This is a no-op for
  // non-VMM managers and idempotent across capture sessions; the over-mapped
  // slack is reclaimed on demand by later eager allocations.
  for (const DeviceStreamRef &stream : streams) {
    if (ErrorOrSuccess err =
            stream->device().memoryManager().preBackForCapture())
      return err.takeError();
  }

  for (const DeviceStreamRef &stream : streams) {
    auto capture = DeviceGraphCaptureContext::beginCapture(stream);
    if (capture.isError())
      return capture.takeError();
    captures.push_back(capture.takeValue());
  }

  return execute();
}

ErrorOr<DeviceGraphCollection>
DeviceGraph::capture(CPUDevice &cpuDevice, ArrayRef<DeviceStreamRef> streams,
                     llvm::function_ref<ErrorOrSuccess()> execute) {
  SmallVector<DeviceGraphCaptureContext, 8> captures;
  if (ErrorOrSuccess err = beginCaptureAndExecute(streams, execute, captures))
    return err.takeError();

  // Phase 1: End stream capture SYNCHRONOUSLY for all streams.
  // cuStreamEndCapture / hipStreamEndCapture must complete before a new
  // capture session can begin on the same stream, so this cannot be deferred
  // to async tasks. Each call is cheap (no GPU graph instantiation here).
  SmallVector<DeviceGraphRef> uninstantiatedGraphs;
  uninstantiatedGraphs.reserve(captures.size());
  for (DeviceGraphCaptureContext &ctx : captures) {
    ErrorOr<DeviceGraphRef> graphOr = ctx.endCapture();
    if (graphOr.isError())
      return graphOr.takeError();
    uninstantiatedGraphs.push_back(graphOr.takeValue());
  }

  // Phase 2: Instantiate graphs in PARALLEL using AsyncRT.
  // cuGraphInstantiateWithParams / hipGraphInstantiateWithFlags are the
  // expensive operations. Each graph is independent, so concurrent
  // instantiation is safe (modulo the profiler mutex in the CUDA backend).
  //
  // launchNonDeviceCollective routes tasks to threads that are NOT
  // per-device-affinitized (reserved for GPU command dispatch), avoiding
  // contention with ongoing kernel launches. Consumer copies of each
  // AsyncDeviceGraphRef go into DeviceGraphCollection; producer copies are
  // moved into the captured vectors so they live until instantiation completes.
  // The returned chain is discarded; per-graph completion is tracked via the
  // individual AsyncDeviceGraphRef values awaited lazily by callers.
  size_t numGraphs = uninstantiatedGraphs.size();

  SmallVector<AsyncDeviceGraphRef> asyncGraphs;
  asyncGraphs.reserve(numGraphs);
  for (auto _ : llvm::seq(numGraphs))
    asyncGraphs.push_back(AsyncDeviceGraphRef::allocate(cpuDevice));

  SmallVector<AsyncDeviceGraphRef> producers;
  producers.reserve(numGraphs);
  for (AsyncDeviceGraphRef &av : asyncGraphs)
    producers.push_back(av.copy());

  MLRT::launchNonDeviceCollective(
      cpuDevice, numGraphs,
      [graphs = std::move(uninstantiatedGraphs),
       results = std::move(producers)](size_t i) mutable {
        if (ErrorOrSuccess err = graphs[i]->instantiate()) {
          std::move(results[i])
              .setToError(AsyncRT::UnknownLocationDecoder::getDiagnostic(
                  err.takeError()));
        } else {
          std::move(results[i]).emplace(std::move(graphs[i]));
        }
      });

  return DeviceGraphCollection(std::move(asyncGraphs));
}

void DeviceGraph::setStream(DeviceStreamRef stream) {
  captureStream_ = std::move(stream);
}

void DeviceGraph::setLaunchTrace(
    std::vector<KernelLaunchMetadata> baselineLaunchTrace) {
  baselineLaunchTrace_ = std::move(baselineLaunchTrace);
  hasBaselineLaunchTrace_ = true;
}

Device &DeviceGraph::device() const { return context_->device(); }

DeviceContext &DeviceGraph::context() const { return *context_; }

ErrorOrSuccess DeviceGraph::patchStableAddresses() const {
  for (const StableAssignment &assignment : stableAssignments_) {
    // The source is this graph's own storage, so it outlives the copy.
    RETURN_IF_ERROR(context_->enqueueCopyToDevice(
        assignment.stableAddress, &assignment.ptr, sizeof(assignment.ptr)));
  }
  return success();
}

//
// ===== DeviceStream ===== //
//
DeviceStream::DeviceStream(DeviceContext &context)
    : ctx_(DeviceContextRef::copy(&context)) {}

Device &DeviceStream::device() const { return ctx_->device(); }

ErrorOrSuccess DeviceStream::beginCapture() const {
  RETURN_ERROR("graph capture not supported on this device");
}

ErrorOr<DeviceGraphRef> DeviceStream::endCapture() {
  RETURN_ERROR("graph capture not supported on this device");
}

void DeviceStream::beginLaunchTrace() const {}

std::vector<KernelLaunchMetadata> DeviceStream::takeLaunchTrace() const {
  return {};
}

//
// ===== DeviceFunction ===== //
//
DeviceFunction::DeviceFunction(DeviceContext &context)
    : context_(DeviceContextRef::copy(&context)) {}

DeviceContext &DeviceFunction::context() { return *context_; }

const DeviceContext &DeviceFunction::context() const { return *context_; }

//
// ===== DeviceTimer ===== //
//
DeviceTimer::DeviceTimer(DeviceContext &context)
    : context_(DeviceContextRef::copy(&context)) {}

const DeviceContext &DeviceTimer::context() const { return *context_; }

//
// ===== CompletionFlag ===== //
//
MODULAR_CXX_EXPORT CompletionFlag::CompletionFlag(DeviceContext &context,
                                                  void *hostPtr,
                                                  uint64_t devicePtr)
    : context_(DeviceContextRef::copy(&context)), hostPtr_(hostPtr),
      devicePtr_(devicePtr) {}

MODULAR_CXX_EXPORT CompletionFlag::~CompletionFlag() {
  if (hostPtr_ == nullptr)
    return;
  // Best-effort free. Destructor must not throw; record errors on the
  // context as pending errors so they're surfaced on the next driver call.
  if (auto err = context_->freeCompletionFlag(hostPtr_); err.isError()) {
    RECORD_PENDING_ERROR(*context_, std::move(err));
  }
  hostPtr_ = nullptr;
}

MODULAR_CXX_EXPORT void CompletionFlag::reset() {
  // Relaxed: the GPU's cuStreamWaitValue64 uses CUDA-level memory fencing.
  // Stream-ordered enqueue of the wait happens-after this store, so no
  // observer can race here. The release-ordering for the *signal* side
  // lives in `signal()` below.
  reinterpret_cast<std::atomic<uint64_t> *>(hostPtr_)->store(
      0, std::memory_order_relaxed);
}

MODULAR_CXX_EXPORT void CompletionFlag::signal(uint64_t value) {
  // Release pairs with the GPU-side `cuStreamWaitValue64` (which under
  // CUDA semantics performs the equivalent of an acquire when the
  // observed value satisfies the predicate) and with `load()`'s acquire
  // for host-side observers. Anything the caller did before `signal()`
  // is visible to whichever side first sees `value`.
  reinterpret_cast<std::atomic<uint64_t> *>(hostPtr_)->store(
      value, std::memory_order_release);
}

MODULAR_CXX_EXPORT uint64_t CompletionFlag::load() const {
  return reinterpret_cast<std::atomic<uint64_t> *>(hostPtr_)->load(
      std::memory_order_acquire);
}

//
// ===== DeviceContext ===== //
//

// Mutex to protect shared, static data. E.g. config dictionary
static std::mutex &getGlobalMutex() {
  static std::mutex mu;
  return mu;
}

DeviceContext::DeviceContext(Device &device, TypeID typeID)
    : device_(DeviceRef::copy(&device)), typeID(typeID),
      freeList_(std::make_unique<MemoryManagerFreeList>(
          this, device.memoryManager(), device.memoryManager().freeList(),
          /*disableFlush=*/false, /*label=*/"freeList")),
      hostFreeList_(std::make_unique<MemoryManagerFreeList>(
          this, device.hostMemoryManager(),
          device.hostMemoryManager().freeList(),
          /*disableFlush=*/false, /*label=*/"hostFreeList")),
      graphFreeList_(std::make_unique<MemoryManagerFreeList>(
          this, device.memoryManager(), device.memoryManager().freeList(),
          /*disableFlush=*/true, /*label=*/"graphFreeList")),
      syncMode_(0) {
  if (syncModeFromConfig()) {
    incSyncMode();
  }
}

DeviceContext::~DeviceContext() {
  // Key function for vtable generation.
}

MODULAR_CXX_EXPORT Device &DeviceContext::device() const { return *device_; }

ErrorOrSuccess DeviceContext::setAsCurrent() const {
  // If this context is already bound, the driver context is already set.
  if (currentlyBoundContext == this)
    return success();
  // If a different context is currently bound, unbind it since we're
  // explicitly changing the current context.
  if (currentlyBoundContext != nullptr)
    unbindFromCurrentThread();
  return device().setCurrentDriverContext(driverContext());
}

MODULAR_CXX_EXPORT bool DeviceContext::isBoundToCurrentThread() const {
  return currentlyBoundContext == this;
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::bindToCurrentThread() const {
  // No-op if device context is already bound to current thread.
  if (currentlyBoundContext == this) {
    return success();
  }

  // Check if the driver context is already the current context.
  ErrorOr<void *> currentCtx = device().getCurrentDriverContext();
  if (currentCtx.isError()) {
    return currentCtx.takeError();
  }
  if (*currentCtx != driverContext()) {
    // Context is not the current context, so set it.
    ErrorOrSuccess err = device().setCurrentDriverContext(driverContext());
    if (err.isError()) {
      return err;
    }
  }

  currentlyBoundContext = this;
  return success();
}

MODULAR_CXX_EXPORT void DeviceContext::unbindFromCurrentThread() {
  currentlyBoundContext = nullptr;
}

MODULAR_CXX_EXPORT ErrorOr<int>
DeviceContext::getAttribute(DeviceContext::Attribute attr) const {
  RETURN_ERROR("Unsupported DeviceContext attribute");
}

MODULAR_CXX_EXPORT ErrorOr<Device::MemInfo>
DeviceContext::getMemoryInfo() const {
  // If we are using the memory manager, get memory information from it.
  MemoryManager &cache = device().memoryManager();
  if (cache.isEnabled()) {
    return cache.getMemoryInfo();
  }

  // Otherwise, go directly to the device.
  return getDeviceMemoryInfo();
}

void DeviceContext::setPendingError(Error err) {
  {
    std::lock_guard<std::mutex> lock(pendingErrorMutex_);
    // Do not overwrite an already pending error.
    if (pendingError_.isError())
      return;

    pendingError_ = std::move(err);
  }
  // After releasing pendingErrorMutex_ also set the atomic value to true.
  hasPendingError_.store(1, std::memory_order_release);
}

class DeviceContext::PendingEventHandler {
public:
  PendingEventHandler(DeviceContextRef ctx, DeviceEventRef event,
                      std::function<void()> handler)
      : next_(nullptr), ctx_(ctx), event_(std::move(event)),
        handler_(std::move(handler)) {}

  // Returns true if the event is ready (or if there's no event).
  ErrorOr<bool> isReady() {
    if (!event_)
      return true; // No event means immediately ready
    return event_->isReady();
  }

  void runHandler() { handler_(); }

private:
  friend class DeviceContext;

  PendingEventHandler *next_;
  // We keep a reference to the context to ensure that we do not release the
  // context while we still have pending events in the queue.
  DeviceContextRef ctx_;
  DeviceEventRef event_;
  std::function<void()> handler_;
};

void DeviceContext::setPendingEventHandler(PendingEventHandler *h) {
  {
    std::lock_guard<std::mutex> lock(pendingWorkMutex_);
    if (pendingEventsLast_ != nullptr) {
      // Enqueue at end.
      assert(pendingEventsLast_->next_ == nullptr);
      pendingEventsLast_->next_ = h;
      pendingEventsLast_ = h;
    } else {
      // Enqueue as first item.
      pendingEvents_ = h;
      pendingEventsLast_ = h;
    }
  }
  // After releasing pendingWorkMutex_ also set the atomic value to true.
  hasPendingWork_.store(1, std::memory_order_release);
}

ErrorOrSuccess DeviceContext::checkPendingError() {
  TRACY_ZONE_SCOPED_NC("DeviceContext::checkPendingError", TRACY_COLOR_PURPLE);

  // Quickly check if we should check for a pending error.
  if (hasPendingError_.load(std::memory_order_acquire) == 0) {
    return success();
  }

  // We can safely remove the flag indicating that the pending error should be
  // checked. We will unconditionally acquire the lock below and check the
  // pending error while protected by the lock.
  hasPendingError_.store(0, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(pendingErrorMutex_);

    // Check if we have a pending error recorded.
    if (!pendingError_.isError())
      return success();

    // Return the error and reset.
    Error result = pendingError_.takeError();
    pendingError_ = success();
    return result;
  }
}

ErrorOrSuccess DeviceContext::pollPendingEvents() {
  TRACY_ZONE_SCOPED_NC("DeviceContext::pollPendingEvents", TRACY_COLOR_PURPLE);

  // During graph capture, events aren't executing - they're graph nodes.
  // cuEventQuery is prohibited during capture, so skip polling. Handlers
  // will run after capture ends and the graph executes.
  if (isGraphCapturing())
    return success();

  // Recycle this context's own completed frees. A failed event query is
  // reported here rather than swallowed; the record it belongs to stays queued,
  // so the next poll picks up whatever this one could not.
  RETURN_IF_ERROR(
      device().memoryManager().drainCompletedFrees(freeList_.get()));
  RETURN_IF_ERROR(
      device().hostMemoryManager().drainCompletedFrees(hostFreeList_.get()));

  // Quickly check if we should check for a pending work.
  if (hasPendingWork_.load(std::memory_order_acquire) == 0) {
    return success();
  }

  // We can safely remove the flag indicating that the pending work should be
  // checked. We will unconditionally acquire the lock below and check the
  // pending event handlers while protected by the lock.
  hasPendingWork_.store(0, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(pendingWorkMutex_);

    // Check if the head of the queue of pending events exists and its event is
    // ready.
    while (pendingEvents_ != nullptr) {
      ErrorOr<bool> isReadyOr = pendingEvents_->isReady();
      if (isReadyOr.isError()) {
        // Since there is at least one pending event handler in the queue we
        // also ensure that we do continue checking for the events to be ready.
        hasPendingWork_.store(1, std::memory_order_release);
        return isReadyOr.takeError();
      }
      if (!*isReadyOr) {
        // First element in the queue is not ready, so we can abort the
        // walk through the pending events.
        // Since there is at least one pending event handler in the queue we
        // also ensure that we do continue checking for the events to be ready.
        hasPendingWork_.store(1, std::memory_order_release);
        break;
      }
      // The handler at the head of the queue is ready to run. Remove the head
      // of the queue from the list of pending handlers.
      // Important: The head of the queue needs to be removed before the lock is
      // released below to avoid racing threads triggering the current handler
      // multiple times.
      PendingEventHandler *curHandler = pendingEvents_;
      pendingEvents_ = pendingEvents_->next_;
      if (pendingEvents_ == nullptr) {
        // We reached the end of the queue.
        pendingEventsLast_ = nullptr;
      }

      // Before calling the handler we need to release the lock in case there
      // are pending errors to be enqueued or there are other effects by the
      // handler code (e.g. causing dead lock or contention).
      pendingWorkMutex_.unlock();
      curHandler->runHandler();
      delete curHandler;
      pendingWorkMutex_.lock();
    }
  }

  return success();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::enqueueCreateBuffer(size_t byteSize) {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, /*owning=*/true);
  return allocBuffer(byteSize, Device::AllocMode::ASYNC);
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::createHostBuffer(size_t byteSize) {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, true);
  return allocBuffer(byteSize, Device::AllocMode::HOST);
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::wrapHostMemory(void *ptr, size_t byteSize) {
  if (ptr == nullptr)
    RETURN_ERROR("wrapHostMemory: pointer is null");
  // Unlike `memAlloc`, not a no-op: a zero length means a miscomputed range.
  if (byteSize == 0)
    RETURN_ERROR("wrapHostMemory: byteSize is zero");

  void *devPtr = nullptr;
  RETURN_IF_ERROR(memWrap(&devPtr, ptr, byteSize));
  // `isHost` stays true: the deferred release and `recordUsedOnContext`
  // ordering both key off it. Only the address differs.
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, devPtr, byteSize, Device::FreeKind::WRAPPED));
}

MODULAR_CXX_EXPORT DeviceBufferRef
DeviceContext::createBufferOwning(void *ptr, size_t byteSize, bool owning) {
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, ptr, byteSize, /*isHost=*/false, owning));
}

MODULAR_CXX_EXPORT DeviceBufferRef
DeviceContext::createHostBufferOwning(void *ptr, size_t byteSize, bool owning) {
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, ptr, byteSize, /*isHost=*/true, owning));
}

ErrorOr<DeviceBufferRef> DeviceContext::allocBuffer(size_t byteSize,
                                                    Device::AllocMode mode) {
  assert(byteSize != 0);

  if (mode == Device::AllocMode::ASYNC)
    mode = Device::AllocMode::SYNC;

  // Select the free list to allocate from:
  //  - graph capture + device: graphFreeList() avoids capturing stream ops as
  //                            graph nodes; allocatedBy_ tracking routes the
  //                            free back to graphFreeList_ via the default
  //                            destructor.
  //  - non-capture + device:   freeList()
  //  - host:                   hostFreeList() (capture-safe: no GPU stream ops)
  MemoryManagerFreeList &allocator = (mode == Device::AllocMode::HOST)
                                         ? hostFreeList()
                                     : isGraphCapturing() ? graphFreeList()
                                                          : freeList();

  ErrorOr<void *> errPtr = allocator.allocate(byteSize, mode);
  if (errPtr.isError())
    return errPtr.takeError();

  return DeviceBufferRef::take(new DeviceBuffer(
      *this, errPtr.takeValue(), byteSize, mode == Device::AllocMode::HOST));
}

// The CPU worker runs after this returns, but a buffer's free doesn't wait:
// a wrapped buffer frees on release, an owned one on its own queue. The refs
// drop on poll, not on the worker, since a numpy release needs the GIL.
//
// TODO(DRIV-579): remove once resource lifetime management lands; until then
// a queue dropped with no later poll leaks these (DRIV-555).
static ErrorOrSuccess
enqueueOnCpuKeepingAlive(DeviceContext &ctx,
                         std::initializer_list<const DeviceBuffer *> buffers,
                         llvm::function_ref<ErrorOrSuccess()> enqueue) {
  // Without this, a loop of enqueues holds every buffer until a synchronize.
  RETURN_IF_ERROR(ctx.pollPendingEvents());
  RETURN_IF_ERROR(enqueue());
  auto keepAlive = std::make_shared<llvm::SmallVector<DeviceBufferRef, 2>>();
  for (const DeviceBuffer *buffer : buffers)
    keepAlive->push_back(
        DeviceBufferRef::copy(const_cast<DeviceBuffer *>(buffer)));
  return ctx.enqueueEventHandler([keepAlive = std::move(keepAlive)]() {});
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::enqueueSetMemory(
    const DeviceBuffer &dst, uint64_t val, size_t valSize) {
  if (dst.context().device().api() != device().api())
    RETURN_ERROR("API mismatch between buffer and context");
  auto fill = [&] {
    return enqueueSetMemory(dst.ptr(), dst.byteSize(), val, valSize);
  };
  if (device().api() != Device::cpuAPI)
    return fill();
  return enqueueOnCpuKeepingAlive(*this, {&dst}, fill);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::enqueueCopyDeviceToDevice(
    DeviceBuffer &dst, DeviceBuffer &src, size_t srcByteOffset,
    size_t srcByteSize, bool crossStreamSync) {
  const Device::API srcApi = src.context().device().api();
  const Device::API dstApi = dst.context().device().api();
  if (srcByteOffset + srcByteSize > src.byteSize())
    RETURN_ERROR("not enough data in src");
  if (srcByteSize != dst.byteSize())
    RETURN_ERROR("size mismatch with dst");

  // Treat host memory and memory owned by a CPU device context as DtoH
  // or HtoD. If the host memory is owned by a CPU device context,
  // synchronize to avoid conflicts with pending events (we cannot wait on
  // events between different device types).
  if (src.isHost() || (srcApi == Device::cpuAPI)) {
    if (srcApi == Device::cpuAPI && dstApi != Device::cpuAPI)
      RETURN_IF_ERROR(src.context().synchronize());

    // The host source is DMA'd on dst's stream, which may differ from src's
    // owning context. Record it so src's host free is ordered against this
    // stream too. If dst is also host memory (pinned -> pinned), it needs no
    // recording here: the copy runs on dst's own owning context, and dst's
    // destructor already orders its free against that context.
    src.recordUsedOnContext(dst.context());
    if (dstApi == Device::cpuAPI)
      return enqueueOnCpuKeepingAlive(dst.context(), {&src, &dst}, [&] {
        return dst.context().enqueueCopyToDevice(
            dst.ptr(), (uint8_t *)src.ptr() + srcByteOffset, srcByteSize);
      });
    return dst.context().enqueueCopyToDevice(
        dst.ptr(), (uint8_t *)src.ptr() + srcByteOffset, srcByteSize);
  } else if (dst.isHost() || (dstApi == Device::cpuAPI)) {
    if (dstApi == Device::cpuAPI && srcApi != Device::cpuAPI)
      RETURN_IF_ERROR(dst.context().synchronize());

    // The host destination is DMA'd on src's stream; record it so dst's host
    // free is ordered against this stream too.
    dst.recordUsedOnContext(src.context());
    return src.context().enqueueCopyFromDevice(
        dst.ptr(), (uint8_t *)src.ptr() + srcByteOffset, srcByteSize);
  } else if (srcApi == dstApi) {
    return this->memCopyDtoD(dst, src, srcByteOffset, srcByteSize,
                             crossStreamSync);
  }
  return Error("Device API mismatch copying from \"" + srcApi + "\" to \"" +
               dstApi + "\"");
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::enqueueBatchCopy(
    llvm::ArrayRef<DeviceBuffer *> dsts, llvm::ArrayRef<DeviceBuffer *> srcs) {
  if (dsts.size() != srcs.size())
    RETURN_ERROR("batch copy requires dsts and srcs of equal length");
  if (dsts.empty())
    return success();

  for (size_t i = 0, e = dsts.size(); i < e; ++i) {
    if (dsts[i]->byteSize() != srcs[i]->byteSize())
      RETURN_ERROR("batch copy requires matching byte sizes for each pair");
    if (&dsts[i]->context() != this)
      RETURN_ERROR("batch copy requires every destination on this context");
  }
  return memBatchCopy(dsts, srcs);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::memBatchCopy(
    llvm::ArrayRef<DeviceBuffer *> dsts, llvm::ArrayRef<DeviceBuffer *> srcs) {
  for (size_t i = 0, e = dsts.size(); i < e; ++i)
    RETURN_IF_ERROR(srcs[i]->copyTo(*dsts[i]));
  return success();
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceContext::enqueueEventHandler(std::function<void()> handler) {
  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, this->enqueueEvent());
  PendingEventHandler *eventHandler = new PendingEventHandler(
      DeviceContextRef::copy(this), std::move(event), std::move(handler));
  setPendingEventHandler(eventHandler);
  return success();
}

MODULAR_CXX_EXPORT
ErrorOr<Driver::DeviceBufferRef>
DeviceContext::memmapFile(const std::filesystem::path &filename) {
  auto result = this->memmapFileImpl(filename);
  if (result.isError())
    return result.takeError();
  auto [ptr, size, mmap] = result.takeValue();

  auto buffer = RCRef<Driver::DeviceBuffer>::take(
      new Driver::DeviceBuffer(*this, ptr, size, /*isHost=*/false,
                               /*owning*/ true));

  // Set destructor on the buffer to keep the mmap alive.
  if (mmap) {
    buffer->setDestructorFn(
        [mmap = std::make_shared<llvm::sys::fs::mapped_file_region>(
             std::move(*mmap))]() { (void)mmap; });
  }

  return buffer;
}

MODULAR_CXX_EXPORT ErrorOr<DeviceContextRef>
DeviceContext::create(StringRef api, int id) {
  ErrorOr<DeviceRef> device = Device::create(api, id);
  if (device.isError()) {
    return device.takeError();
  }

  // Check device compatibility before creating the context. This ensures clear
  // error messages for incompatible drivers or hardware, matching the behavior
  // of the Python Accelerator() constructor.
  ErrorOrSuccess compatible = (*device)->isCompatible();
  if (compatible.isError()) {
    return compatible.takeError();
  }

  return (*device)->createContext();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceContextRef>
DeviceContext::getOrCreate(StringRef api, int id) {
  static std::mutex mu;
  static llvm::StringMap<DeviceContext *> contextMap;

  ErrorOr<DeviceRef> errDev = Device::create(api, id);
  if (errDev.isError())
    return errDev.takeError();

  // Check device compatibility before creating the context.
  ErrorOrSuccess compatible = (*errDev)->isCompatible();
  if (compatible.isError())
    return compatible.takeError();

  // Create the cache key in such a way that for example an `api` of "gpu" and
  // "cuda" are equivalent on an CUDA based system.
  std::string cacheKey = ((*errDev)->api() + "[" + llvm::Twine(id) + "]").str();
  {
    std::lock_guard<std::mutex> lock(mu);
    auto iter = contextMap.find(cacheKey);
    if (iter != contextMap.end()) {
      DeviceContext *cachedCtx = iter->second;
      return DeviceContextRef::copy(cachedCtx);
    }
    ErrorOr<DeviceContextRef> errCtx = (*errDev)->createContext();
    if (errCtx.isError())
      return errCtx.takeError();

    DeviceContext *ctx = errCtx.takeValue().release();
    contextMap.insert({cacheKey, ctx});
    return DeviceContextRef::copy(ctx);
  }
}

MODULAR_CXX_EXPORT bool
DeviceContext::operator==(const DeviceContext &other) const {
  if (this == &other)
    return true;

  ErrorOr<DeviceStreamRef> errStream = stream();
  if (errStream.isError())
    return false;
  void *streamThis = (*errStream)->ptr();

  errStream = other.stream();
  if (errStream.isError())
    return false;
  void *streamOther = (*errStream)->ptr();

  return streamThis == streamOther;
}

MODULAR_CXX_EXPORT bool
DeviceContext::operator!=(const DeviceContext &other) const {
  return !(*this == other);
}

unsigned DeviceContext::numStreams() const { return 1; }

ErrorOr<DeviceContextRef> DeviceContext::selectStream(unsigned streamId) {
  // No multi-stream model: only the single default stream (id 0) exists, so any
  // other stream id is a programming error. Stream 0 resolves to this context.
  if (streamId != 0)
    return Error("invalid stream id: " + llvm::Twine(streamId));
  return DeviceContextRef::copy(this);
}

bool DeviceContext::flushIdleStreamViews() { return false; }

MemoryManagerFreeList &DeviceContext::freeList() const { return *freeList_; }
MemoryManagerFreeList &DeviceContext::hostFreeList() const {
  return *hostFreeList_;
}
MemoryManagerFreeList &DeviceContext::graphFreeList() const {
  return *graphFreeList_;
}

//
// ===== Device ===== //
//
MODULAR_CXX_EXPORT int Device::numberOfDevices(StringRef api) {
  // In virtual device mode, return the virtual device count for GPU APIs.
  // This allows validation checks to pass even when requesting more GPUs
  // than are physically available.
  int virtualCount = AsyncRT::getVirtualDeviceCount();
  if (virtualCount > 0 && api == gpuAPI) {
    return virtualCount;
  }

  // TODO(iposva): Implement dynamic registration mechanism.
  if ((api == gpuAPI) || (api == cudaAPI)) {
    ErrorOr<int> count = CUDADevice::numberOfGPUs();
    if (!count.isError())
      return *count;
  }
  if ((api == gpuAPI) || (api == hipAPI)) {
    ErrorOr<int> count = HIPDevice::numberOfGPUs();
    if (!count.isError())
      return *count;
  }
  if ((api == gpuAPI) || (api == metalAPI)) {
#if MODULAR_MACOS
    ErrorOr<int> count = MetalDevice::numberOfGPUs();
    if (!count.isError())
      return *count;
#endif
  }
  // Check plugins for any API (generic, no hardcoded API names).
  {
    Plugin::ensurePluginsLoaded();
    int count = Plugin::deviceCount(api);
    if (count > 0)
      return count;
  }

  if (api == cpuAPI) {
    // For CPU we go back to the device to get the number of supported devices.
    ErrorOr<CpuDevice &> device = CpuDevice::get(0);
    if (!device.isError())
      return device->deviceCount();
  }

  return 0;
}

MemoryManager &Device::memoryManager() const {
  // Built lazily on first access rather than in the constructor so onDevice()
  // sees the fully-constructed derived Device when it queries the virtual
  // vmmSupported(). First access must therefore occur while the derived Device
  // is alive: an access during or after subclass destruction queries the base
  // vmmSupported() (always false) and silently builds a non-VMM manager.
  std::call_once(memoryManagerOnce_, [&] {
    memoryManager_.reset(MemoryManager::onDevice(const_cast<Device &>(*this),
                                                 memInfo_, allocAlignment_));
  });
  return *memoryManager_;
}

size_t Device::allocatedSize(void *ptr) const {
  return memoryManager().allocatedSize(ptr);
}

void Device::clearMemoryManager() {
  // Tear the device MemoryManager down here, while the derived Device is still
  // alive, so the virtual VMM unmap/release calls in clear() dispatch to the
  // derived overrides. The base ~Device runs after the derived subobject is
  // gone, so subclass destructors must call this. Resetting the pointer ensures
  // the manager is destroyed cleared (its ~MemoryManager asserts no live
  // chunks) and is not destroyed a second time by the base ~Device.
  if (memoryManager_) {
    memoryManager_->clear();
    memoryManager_.reset();
  }
}

MemoryManager &Device::hostMemoryManager() const {
  assert(hostMemoryManager_ != nullptr);
  return *hostMemoryManager_;
}

bool DeviceContext::syncMode() const {
  return syncMode_.load(std::memory_order_relaxed) > 0;
}
void DeviceContext::incSyncMode() {
  syncMode_.fetch_add(1, std::memory_order_relaxed);
}
void DeviceContext::decSyncMode() {
  syncMode_.fetch_add(-1, std::memory_order_relaxed);
}

bool DeviceContext::isGraphCapturing() const { return false; }

ErrorOr<DeviceGraphBuilderRef>
DeviceContext::createGraphBuilder(DeviceGraphMemoryPoolRef) {
  RETURN_ERROR("createGraphBuilder() not supported on this device context");
}

ErrorOrSuccess DeviceContext::streamPriorityRange(int *leastPriority,
                                                  int *greatestPriority) {
  if (leastPriority)
    *leastPriority = 0;
  if (greatestPriority)
    *greatestPriority = 0;
  return success();
}

ErrorOr<DeviceStreamRef> DeviceContext::createStream(int priority) {
  RETURN_ERROR("createStream is not supported on this device");
}

ErrorOr<DeviceStreamRef>
DeviceContext::createExternalStream(void *externalStream) {
  RETURN_ERROR("createExternalStream is not supported on this device");
}

ErrorOr<DeviceEventRef> DeviceContext::eventCreate(bool enableTiming) {
  RETURN_ERROR("eventCreate is not supported on this device");
}

ErrorOr<void *> DeviceContext::getHostAccessiblePointer(const void *devicePtr) {
  RETURN_ERROR("getHostAccessiblePointer is not supported on this device");
}

ErrorOrSuccess DeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) {
  RETURN_ERROR(
      "occupancyMaxActiveBlocksPerMultiprocessor is not supported on this "
      "device");
}

bool DeviceContext::configAsBool(StringRef key, bool defaultValue,
                                 bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  if (isDefault != nullptr) {
    StringRef s =
        config->getValue((llvm::Twine("device-context.") + key).str());
    *isDefault = s.empty();
  }
  return config->getValueAsBool((llvm::Twine("device-context.") + key).str(),
                                defaultValue);
}

bool DeviceContext::syncModeFromConfig() {
  // Sync mode can be enabled via the new `max-debug.device-sync-mode` debug
  // configuration (e.g. `MODULAR_DEBUG=device-sync-mode`) or via the legacy
  // `device-context.sync-mode` key (e.g. `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1`).
  if (configAsBool("sync-mode", false))
    return true;
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError())
    return false;
  return config->getValueAsBool("max-debug.device-sync-mode", false);
}

int64_t DeviceContext::configAsInt(StringRef key, int64_t defaultValue,
                                   bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  StringRef s = config->getValue((llvm::Twine("device-context.") + key).str());
  if (isDefault != nullptr) {
    *isDefault = s.empty();
  }
  long long result = defaultValue;
  if (s.getAsInteger(10, result)) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }
  return result;
}

StringRef DeviceContext::configAsString(StringRef key, StringRef defaultValue,
                                        bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  if (isDefault != nullptr) {
    StringRef s =
        config->getValue((llvm::Twine("device-context.") + key).str());
    *isDefault = s.empty();
  }
  return config->getValueOr((llvm::Twine("device-context.") + key).str(),
                            defaultValue);
}

bool DeviceContext::isDebugAllocatorEnabled(StringRef name) const {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  return config->isValueInList("debug.device_allocator", name);
}

ErrorOr<Config> &DeviceContext::getConfig() {
  static ErrorOr<Config> configuration = Config::open();
  return configuration;
}

MODULAR_CXX_EXPORT const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    // First check if we're in virtual device mode (cross-compilation).
    // Virtual device config is set at startup and takes precedence over
    // physical hardware detection.
    std::string virtualTargetArch = AsyncRT::getVirtualDeviceTargetArch();
    if (!virtualTargetArch.empty()) {
      std::string virtualAPI = AsyncRT::getVirtualDeviceAPI();
      return (llvm::Twine(apiToVendorPrefix(virtualAPI)) + ":" +
              virtualTargetArch)
          .str();
    }

    // Otherwise, detect from physical hardware
    std::string vendor("nvidia");
    ErrorOr<std::string> archName = CUDADevice::getFirstArchName();

    if (archName.isError()) {
      vendor = "amdgpu";
      archName = HIPDevice::getArchName(0);
    }

#if MODULAR_MACOS
    if (archName.isError()) {
      vendor = "metal";
      archName = MetalDevice::getArchName(nullptr);
    }
#endif

    // Check all loaded plugins (generic, no hardcoded vendor names).
    if (archName.isError()) {
      Plugin::ensurePluginsLoaded();
      auto pluginResult = Plugin::getArchFromPlugins();
      if (!pluginResult.isError()) {
        vendor = pluginResult.get().first;
        archName = pluginResult.get().second;
      }
    }

    if (archName.isError())
      return std::string();

    return (llvm::Twine(vendor) + ":" + archName.get()).str();
  }();
  return arch;
}

ErrorOrSuccess Device::enablePeerAccess(const Device &peer) {
  // Call device-specific implementation.
  ErrorOrSuccess result = enablePeerAccessImpl(peer);
  // If successful, set the bitvector to indicate that the peer is accessible.
  if (!result.isError()) {
    std::lock_guard<std::mutex> lock(mu_);
    auto setBit = [&](int deviceId) {
      auto id = static_cast<unsigned int>(deviceId);
      if (canAccessPeer_.size() <= id)
        canAccessPeer_.resize(id + 1);
      canAccessPeer_.set(id);
    };
    // Set peer bit.
    setBit(peer.id());
    // Set self bit so we can use all() to check if all peers are enabled.
    setBit(id());
  }
  return result;
}

ErrorOrSuccess Device::enablePeerAccessImpl(const Device &peer) {
  RETURN_ERROR("Peer access not implemented");
}

std::string KernelLaunchMetadata::format(int count) const {
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "    - " << kernelName << " (hash: " << semanticHash << ")";

  // Show operation-specific details based on kind
  switch (kind) {
  case OperationKind::KernelLaunch:
    os << "\n        Grid: (" << gridX << ", " << gridY << ", " << gridZ
       << "), Block: (" << blockX << ", " << blockY << ", " << blockZ << ")";
    if (sharedMemBytes > 0)
      os << ", Shared: " << sharedMemBytes << " bytes";
    if (numAttrs > 0)
      os << ", Attrs: " << numAttrs;
    break;

  case OperationKind::Memcpy:
    os << "\n        Kind: ";
    switch (memcpyKind) {
    case MemcpyKind::HtoD:
      os << "Host-to-Device";
      break;
    case MemcpyKind::DtoH:
      os << "Device-to-Host";
      break;
    case MemcpyKind::DtoD:
      os << "Device-to-Device";
      break;
    case MemcpyKind::None:
      os << "Unknown";
      break;
    }
    os << ", Size: " << memcpyByteSize << " bytes";
    break;

  case OperationKind::Memset:
    os << "\n        Size: " << memsetByteSize << " bytes, Value: "
       << llvm::format_hex(memsetValue, memsetValueSize * 2)
       << " (valSize: " << memsetValueSize << ")";
    break;
  }

  os << " [count: " << count << "]\n";
  return os.str();
}

llvm::json::Value KernelLaunchMetadata::formatJSON() const {
  llvm::json::Object json;

  // Add common fields
  json["kernelName"] = kernelName;
  json["semanticHash"] = static_cast<int64_t>(semanticHash);

  // Add operation kind as a string
  switch (kind) {
  case OperationKind::KernelLaunch:
    json["kind"] = "KernelLaunch";
    break;
  case OperationKind::Memcpy:
    json["kind"] = "Memcpy";
    break;
  case OperationKind::Memset:
    json["kind"] = "Memset";
    break;
  }

  // Add operation-specific fields
  switch (kind) {
  case OperationKind::KernelLaunch: {
    llvm::json::Object grid;
    grid["x"] = gridX;
    grid["y"] = gridY;
    grid["z"] = gridZ;
    json["grid"] = std::move(grid);

    llvm::json::Object block;
    block["x"] = blockX;
    block["y"] = blockY;
    block["z"] = blockZ;
    json["block"] = std::move(block);

    json["sharedMemBytes"] = sharedMemBytes;
    json["numAttrs"] = numAttrs;
    json["streamIndex"] = streamIndex;
    break;
  }

  case OperationKind::Memcpy: {
    // Convert MemcpyKind to string
    switch (memcpyKind) {
    case MemcpyKind::HtoD:
      json["memcpyKind"] = "HtoD";
      break;
    case MemcpyKind::DtoH:
      json["memcpyKind"] = "DtoH";
      break;
    case MemcpyKind::DtoD:
      json["memcpyKind"] = "DtoD";
      break;
    case MemcpyKind::None:
      json["memcpyKind"] = "None";
      break;
    }
    json["byteSize"] = static_cast<int64_t>(memcpyByteSize);
    break;
  }

  case OperationKind::Memset: {
    json["byteSize"] = static_cast<int64_t>(memsetByteSize);
    // Format value as hex string for clarity
    std::string hexValue;
    llvm::raw_string_ostream hexStream(hexValue);
    hexStream << llvm::format_hex(memsetValue, memsetValueSize * 2);
    json["value"] = hexStream.str();
    json["valueSize"] = static_cast<int64_t>(memsetValueSize);
    break;
  }
  }

  return llvm::json::Value(std::move(json));
}

ErrorOrSuccess Device::enableAllPeerAccess() {
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  if (numDevices <= 1)
    return success();

  SmallVector<DeviceRef> devices;
  devices.reserve(numDevices);
  for (int id = 0; id < numDevices; ++id) {
    DEFINE_OR_RETURN_ERROR(DeviceRef, device,
                           Device::create(Device::gpuAPI, id));
    devices.emplace_back(std::move(device));
  }

  for (int i = 0; i < numDevices; i++) {
    for (int j = i + 1; j < numDevices; j++) {
      // i -> j
      RETURN_IF_ERROR(devices[i]->enablePeerAccess(*devices[j]));
      // j -> i
      RETURN_IF_ERROR(devices[j]->enablePeerAccess(*devices[i]));
    }
  }

  g_allPeerAccessEnabled = true;
  return success();
}

ErrorOr<bool> Device::allPeerAccessEnabled() {
  if (g_allPeerAccessEnabled.load(std::memory_order_relaxed))
    return true;

  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  if (numDevices < 2)
    return false;
  for (int i : llvm::seq(0, numDevices)) {
    DEFINE_OR_RETURN_ERROR(DeviceRef, device,
                           Device::create(Device::gpuAPI, i));
    std::lock_guard<std::mutex> lock(device->mu_);
    if (device->canAccessPeer_.empty() || !device->canAccessPeer_.all())
      return false;
  }

  g_allPeerAccessEnabled.store(true, std::memory_order_relaxed);
  return true;
}

} // namespace M::Driver
