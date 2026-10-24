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

#include "MLRT/Driver/DeviceContext/CPU/CpuDeviceContext.h"

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "Support/AlignedAlloc.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/ML/Fill.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/ADT/Twine.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#define CHECK_IS_CPU_ERR(dev)                                                  \
  do {                                                                         \
    if ((dev).api() != Device::cpuAPI) {                                       \
      RETURN_ERROR("Not a CPU device.");                                       \
    }                                                                          \
  } while (0)

namespace M::Driver {

//
// ===== CpuDevice ===== //
//
// TODO(iposva): Consider using the actual device name.
CpuDevice::CpuDevice(Device::MemInfo memInfo)
    : Device(api, "cpu", 0, memInfo, M::kPreferredMemoryAlignment) {}

CpuDevice::~CpuDevice() { clearMemoryManager(); }

ErrorOr<int> CpuDevice::getApiVersion() const { return 0; }

ErrorOr<DriverVersion &> CpuDevice::getDriverVersion() const {
  static DriverVersion version("0.0.0");
  return version;
}

ErrorOrSuccess CpuDevice::isCompatible() const { return success(); }

ErrorOr<int> CpuDevice::computeCapability() const {
  // RUNP-494 Remove this, compute capability is a CUDA-specific concept.
  return 0;
}

ErrorOr<bool> CpuDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> CpuDevice::getAttribute(Attribute attr) const { return 0; }

ErrorOr<std::string> CpuDevice::getPciBusId() const {
  RETURN_ERROR("PCI bus address not available for CPU devices");
}

ErrorOr<DeviceContextRef> CpuDevice::createContext() {
  return CpuDeviceContextRef::create(*this);
}

int CpuDevice::deviceCount() const {
  // Only a single CpuDevice is supported.
  return 1;
}

ErrorOrSuccess CpuDevice::chunkAlloc(void **ptr, size_t byteSize, bool isHost) {
  assert(byteSize != 0);
  *ptr = M::AsyncRT::TCMallocGlobals::tc_new(1, byteSize);
  return success();
}

ErrorOrSuccess CpuDevice::chunkFree(void *ptr, bool isHost) {
  M::AsyncRT::TCMallocGlobals::tc_delete(ptr);
  return success();
}

#ifdef _WIN32

ErrorOr<Device::MemInfo> CpuDevice::getMemoryInfo() const {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    RETURN_ERROR("Failed to get memory status via GlobalMemoryStatusEx");
  Device::MemInfo info;
  info.total = status.ullTotalPhys;
  info.free = status.ullAvailPhys;
  return info;
}

#else // _WIN32

// Call sysconf() and check for errors.
static ErrorOr<size_t> getSystemValue(int name) {
  errno = 0;
  auto value = sysconf(name);
  if (errno != 0)
    RETURN_ERROR(strerror(errno));
  return static_cast<size_t>(value);
}

ErrorOr<Device::MemInfo> CpuDevice::getMemoryInfo() const {
  Device::MemInfo info;
  auto numPages = getSystemValue(_SC_PHYS_PAGES);
  if (numPages.isError())
    return numPages.takeError();
  auto pageSize = getSystemValue(_SC_PAGE_SIZE);
  if (pageSize.isError())
    return pageSize.takeError();

  rusage usage;
  if (-1 == getrusage(RUSAGE_SELF, &usage))
    RETURN_ERROR("Failed to get current memory usage via rusage");

// The unit for the _SC_PAGE_SIZE is bytes. The unit for the ru_maxrss is
// kilobytes on Linux and bytes on MacOS.
#if MODULAR_LINUX
  const size_t maxrssMultiple = 1024;
#else // macOS
  const size_t maxrssMultiple = 1;
#endif

  // This is the maximum resident set size used.
  // Current memory use may be lower, this is the "high water mark".
  size_t used = usage.ru_maxrss * maxrssMultiple;
  // Total memory = number of pages times the page size.
  info.total = numPages.takeValue() * pageSize.takeValue();
  // Free memory = total memory minus memory in use.
  info.free = info.total - used;
  return info;
}

#endif // _WIN32

ErrorOr<void *> CpuDevice::getCurrentDriverContext() const { return nullptr; }
ErrorOrSuccess CpuDevice::setCurrentDriverContext(void *ctx) {
  // Not tracking a context for CPU.
  return success();
}

const CpuDevice &CpuDevice::cast(const Device &device) {
  return static_cast<const CpuDevice &>(device);
}

ErrorOr<CpuDevice &> CpuDevice::get(int id) {
  if (id != 0) {
    RETURN_ERROR("CPU device only supports id == 0.");
  }

  static CpuDevice *device = new CpuDevice(Device::MemInfo());
  return *device;
}

// ===== CpuDeviceStream ===== //
CpuDeviceStream::CpuDeviceStream(CpuDeviceContext &ctx) : DeviceStream(ctx) {}

ErrorOrSuccess CpuDeviceStream::synchronize() { return success(); }

ErrorOr<DeviceEventRef> CpuDeviceStream::enqueueEvent() const {
  RETURN_ERROR("Unimplemented");
}

ErrorOrSuccess
CpuDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  RETURN_ERROR("Unimplemented");
}

ErrorOrSuccess CpuDeviceStream::eventRecord(const DeviceEvent &event) const {
  RETURN_ERROR("Unimplemented");
}

ErrorOrSuccess CpuDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t /*argCount*/,
    const size_t * /*argSizes*/) const {
  RETURN_ERROR("Unimplemented");
}

void *CpuDeviceStream::ptr() const { return nullptr; }

// ===== CpuDeviceTimer ===== //
CpuDeviceTimer::CpuDeviceTimer(
    CpuDeviceContext &context,
    std::chrono::time_point<std::chrono::steady_clock> start)
    : DeviceTimer(context), start_(start) {}

std::chrono::time_point<std::chrono::steady_clock>
CpuDeviceTimer::cpu_start() const {
  return start_;
}

const CpuDeviceTimer &CpuDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const CpuDeviceTimer &>(timer);
}

// ===== CpuDeviceEvent ===== //
CpuDeviceEvent::CpuDeviceEvent(CpuDevice &device,
                               AsyncRT::CPUDeviceRef cpuDevice,
                               AsyncRT::AnyAsyncValueRef value)
    : DeviceEvent(device), cpuDevice(std::move(cpuDevice)),
      value(std::move(value)) {}

ErrorOrSuccess CpuDeviceEvent::synchronize() {
  cpuDevice->getWorkQueue()->await({&value, 1});
  return success();
}

ErrorOr<bool> CpuDeviceEvent::isReady() const { return value.isReady(); }

//
// ===== CpuDeviceContext === //
//
CpuDeviceContext::CpuDeviceContext(CpuDevice &device)
    : DeviceContext(device, TypeID::get<CpuDeviceContext>()) {
  // Get or create the cpuDevice. Request mainWillDonate=false if this is the
  // only cpuDevice user, but allow using the existing cpuDevice options if they
  // are set. This is safe as the other cpuDevice users are guaranteeing they
  // will wait on the main thread to progress work. Otherwise CpuDeviceContext
  // is invariant to the cpuDevice options, so this is safe.
  cpuDevice = AsyncRT::getOrCreateCPUDevice(
      AsyncRT::CPUDeviceSource::CPUDeviceContext,
      AsyncRT::CPUDeviceOptions().withMainWillNotDonate(),
      /*allowUsingExistingOptions=*/true);

  streamChain = cpuDevice->getReadyChain().copy();
}

CpuDeviceContext::~CpuDeviceContext() {
  // Wait for pending work to finish
  if (!streamChain.isReady())
    cpuDevice->getWorkQueue()->await({&streamChain, 1});
}

ErrorOrSuccess CpuDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          Device::AllocMode mode) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::memAlloc", TRACY_COLOR_PURPLE);
  assert(byteSize != 0);
  *ptr = cpuDevice->getAllocator()->allocateBytes(byteSize, /*alignment=*/1);
  if (*ptr == nullptr) {
    RETURN_ERROR("Memory allocation failed");
  }
  return success();
}

ErrorOrSuccess CpuDeviceContext::memFree(void *ptr, Device::FreeKind kind) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::memFree", TRACY_COLOR_PURPLE);
  // A CPU device's allocations are already host memory; nothing to wrap.
  if (kind == Device::FreeKind::WRAPPED)
    RETURN_ERROR("CpuDeviceContext::memFree: WRAPPED memory is not supported");
  // The Runtime's allocator outlives any work enqueued on its work queue, so
  // capturing a raw pointer here (rather than a CPUDeviceRef) is safe.
  auto *allocator = cpuDevice->getAllocator();
  if (kind == Device::FreeKind::HOST) {
    allocator->deallocateBytes(ptr);
    return success();
  }
  return enqueueWork([=]() { allocator->deallocateBytes(ptr); });
}

void *CpuDeviceContext::driverContext() const { return nullptr; }

// Enqueue work with stream-like semantics. Keep a chain to the last enqueued
// function so the next task can await it.
// Returns any pending error from previously enqueued async work.
ErrorOrSuccess CpuDeviceContext::enqueueWork(AsyncRT::TaskFunction &&work) {
  RETURN_IF_PENDING_ERROR(*this);

  auto eventChain =
      AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(*cpuDevice);

  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  streamChain.andThenAsync(
      [work = std::move(work), eventChain = eventChain.copy()]() mutable {
        work();
        std::move(eventChain).emplace();
      });
  streamChain = std::move(eventChain);
  return success();
}

ErrorOrSuccess CpuDeviceContext::enqueueCopyToDevice(void *dst, const void *src,
                                                     size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);
  return enqueueWork([=]() { memcpy(dst, src, byteSize); });
}

ErrorOrSuccess CpuDeviceContext::enqueueCopyFromDevice(void *dst,
                                                       const void *src,
                                                       size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);
  return enqueueWork([=]() { memcpy(dst, src, byteSize); });
}

ErrorOrSuccess CpuDeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                             DeviceBuffer &src,
                                             size_t srcByteOffset,
                                             size_t srcByteSize,
                                             bool /*crossStreamSync*/) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);
  CHECK_IS_CPU_ERR(src.context().device());
  CHECK_IS_CPU_ERR(dst.context().device());
  void *srcPtr = (uint8_t *)src.ptr() + srcByteOffset;
  void *dstPtr = dst.ptr();
  size_t size = dst.byteSize();
  assert(srcByteOffset + srcByteSize <= src.byteSize());
  assert(srcByteSize == dst.byteSize());
  return enqueueWork([=]() { memcpy(dstPtr, srcPtr, size); });
}

ErrorOrSuccess CpuDeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                                  uint64_t val,
                                                  size_t valSize) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

  if (valSize > sizeof(uint64_t))
    return Error("unsupported val size");

  if (valSize != 1 && (byteSize & (valSize - 1)))
    return Error("byteSize of buffer is misaligned");

  return enqueueWork([this, dst, byteSize, val, valSize]() {
    size_t len = byteSize / valSize;
    switch (valSize) {
    case 1: {
      uint8_t u8Val = static_cast<uint8_t>(val);
      RECORD_PENDING_ERROR(*this,
                           M::fillHomogeneous(dst, len, DType::ui8, &u8Val));
      break;
    }
    case 2: {
      uint16_t ui16Val = static_cast<uint16_t>(val);
      RECORD_PENDING_ERROR(*this,
                           M::fillHomogeneous(dst, len, DType::ui16, &ui16Val));
      break;
    }
    case 4: {
      uint32_t ui32Val = static_cast<uint32_t>(val);
      RECORD_PENDING_ERROR(*this,
                           M::fillHomogeneous(dst, len, DType::ui32, &ui32Val));
      break;
    }
    case 8: {
      uint64_t ui64Val = static_cast<uint64_t>(val);
      RECORD_PENDING_ERROR(*this,
                           M::fillHomogeneous(dst, len, DType::ui64, &ui64Val));
      break;
    }
    default:
      llvm_unreachable("unsupported val size");
    }
  });
}

DeviceContext::MMapFileImplResult
CpuDeviceContext::memmapFileImpl(const std::filesystem::path &filename) const {
  auto result = openFile(filename, /*readOnly*/ true);
  if (result)
    return result.takeError();
  auto [fd, status] = *result;

  auto size = status.getSize();

  std::error_code ec;
  llvm::sys::fs::mapped_file_region mappedFile(
      fd, llvm::sys::fs::mapped_file_region::readonly, size, 0, ec);
  if (ec) {
    llvm::sys::fs::closeFile(fd);
    return Error(ec.message());
  }

  // Close the file, mmap will hold a ref to the descriptor.
  llvm::sys::fs::closeFile(fd);

  return std::tuple{
      mappedFile.data(),
      size,
      std::optional{std::move(mappedFile)},
  };
}

ErrorOr<DeviceFunctionRef>
CpuDeviceContext::loadFunction(StringRef moduleName, StringRef functionName,
                               StringRef code, int32_t maxDynamicSharedBytes,
                               StringRef debugLevel,
                               int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::loadFunction", TRACY_COLOR_PURPLE);
  // Currently not implemented. Maybe never to be implemented.
  RETURN_ERROR("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);
  // Currently not implemented. Maybe never to be implemented.
  RETURN_ERROR("Unimplemented");
}

ErrorOrSuccess
CpuDeviceContext::enqueueHostFunction(void (*resumeFunc)(int8_t *),
                                      void (*destroyFunc)(int8_t *),
                                      int8_t *coroHandle) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueHostFunction",
                       TRACY_COLOR_PURPLE);
  // Resume the coroutine to execute the enqueued function, then immediately
  // destroy the coroutine as ownership of it has been transferred from Mojo by
  // this point.
  return enqueueWork([=]() {
    resumeFunc(coroHandle);
    destroyFunc(coroHandle);
  });
}

ErrorOrSuccess CpuDeviceContext::enqueueHostFunctionRange(
    void (*resumeFunc)(int8_t *), void (*destroyFunc)(int8_t *),
    int8_t **coroHandles, size_t numHandles) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueHostFunctionRange",
                       TRACY_COLOR_PURPLE);
  RETURN_IF_PENDING_ERROR(*this);

  if (numHandles == 0)
    return success();

  size_t numWorkers = cpuDevice->getWorkQueue()->getParallelismLevel();
  auto eventChain =
      AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(*cpuDevice);
  std::lock_guard<std::mutex> streamLock{streamChainMutex};

  // When the current streamChain is ready, fan out all coroutine handles
  // across worker threads via parallelForEachNChainWithTaskIds, then
  // complete eventChain when all handles have been processed.
  // Copy the handles array since it may not outlive this call. Use a plain
  // std::vector as we will typically have one handle per CPU thread which is
  // too big for SmallVector.
  streamChain.andThenSync([cpuDevice = this->cpuDevice, numWorkers, resumeFunc,
                           destroyFunc, eventChain = eventChain.copy(),
                           handles = std::vector<int8_t *>(
                               coroHandles,
                               coroHandles + numHandles)]() mutable {
    size_t count = handles.size();
    auto completionChain = AsyncRT::parallelForEachNChainWithTaskIds(
        *cpuDevice, count,
        [numWorkers](size_t i) -> int {
          return static_cast<int>(i % numWorkers);
        },
        [resumeFunc, destroyFunc](size_t i, std::vector<int8_t *> &handles) {
          resumeFunc(handles[i]);
          destroyFunc(handles[i]);
        },
        std::move(handles));
    completionChain.andThenSync([eventChain = std::move(eventChain)]() mutable {
      std::move(eventChain).emplace();
    });
  });

  streamChain = std::move(eventChain);
  return success();
}

ErrorOr<DeviceStreamRef> CpuDeviceContext::stream() const {
  // This function is const only for the duration of its execution;
  // `this` is non-const downstream so cast in order to provide
  // mutable reference to the new stream object.
  return CpuDeviceStreamRef::create(const_cast<CpuDeviceContext &>(*this));
}

ErrorOr<DeviceStreamRef>
CpuDeviceContext::createExternalStream(void *externalStream) {
  return CpuDeviceStreamRef::create(*this);
}

ErrorOrSuccess CpuDeviceContext::synchronize() {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::synchronize", TRACY_COLOR_PURPLE);
  RETURN_IF_PENDING_ERROR(*this);
  AsyncRT::AnyAsyncValueRef head;
  {
    std::lock_guard<std::mutex> streamLock{streamChainMutex};
    head = streamChain.copy();
  }
  cpuDevice->getWorkQueue()->await({&head, 1});
  RETURN_IF_ERROR(pollPendingEvents());
  return checkPendingError();
}

ErrorOr<DeviceEventRef> CpuDeviceContext::enqueueEvent() {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);
  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  return CpuDeviceEventRef::create(static_cast<CpuDevice &>(device()),
                                   cpuDevice, streamChain.copy());
}

ErrorOrSuccess CpuDeviceContext::enqueueEvent(const DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);
  auto &cpuEvent =
      const_cast<CpuDeviceEvent &>(static_cast<const CpuDeviceEvent &>(event));
  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  cpuEvent.asyncValue() = streamChain.copy();
  return success();
}

ErrorOr<DeviceEventRef> CpuDeviceContext::eventCreate(bool enableTiming) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::eventCreate", TRACY_COLOR_PURPLE);
  return CpuDeviceEventRef::create(static_cast<CpuDevice &>(device()),
                                   cpuDevice,
                                   cpuDevice->getReadyChain().copy());
}

ErrorOrSuccess CpuDeviceContext::enqueueWaitFor(DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);
  RETURN_IF_PENDING_ERROR(*this);

  auto &cpuEvent = static_cast<CpuDeviceEvent &>(event);
  auto eventChain =
      AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(*cpuDevice);

  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  AsyncRT::andThenAsync(
      std::make_tuple(streamChain.copy(), cpuEvent.asyncValue().copy()),
      [eventChain = eventChain.copy()](auto, auto) mutable {
        std::move(eventChain).emplace();
      });
  streamChain = std::move(eventChain);
  return success();
}

ErrorOrSuccess CpuDeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  TRACY_ZONE_SCOPED_NC("CpuDeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);
  RETURN_IF_PENDING_ERROR(*this);

  auto &other = static_cast<CpuDeviceContext &>(ctx);
  auto eventChain =
      AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(*cpuDevice);

  std::scoped_lock<std::mutex, std::mutex> lock{streamChainMutex,
                                                other.streamChainMutex};
  AsyncRT::andThenAsync(
      std::make_tuple(streamChain.copy(), other.streamChain.copy()),
      [eventChain = eventChain.copy()](auto, auto) mutable {
        std::move(eventChain).emplace();
      });
  streamChain = std::move(eventChain);
  return success();
}

ErrorOr<DeviceTimerRef> CpuDeviceContext::startTimer() {
  return CpuDeviceTimerRef::create(*this, std::chrono::steady_clock::now());
}

ErrorOr<std::chrono::nanoseconds>
CpuDeviceContext::stopTimer(const DeviceTimer &timer) {
  return std::chrono::steady_clock::now() -
         CpuDeviceTimer::cast(timer).cpu_start();
}

ErrorOr<int> CpuDeviceContext::getAttribute(Attribute attr) const {
  switch (attr) {
  case Attribute::PARALLELISM_LEVEL:
    return static_cast<int>(cpuDevice->getWorkQueue()->getParallelismLevel());
  }
  RETURN_ERROR("Unsupported CpuDeviceContext attribute");
}

ErrorOr<Device::MemInfo> CpuDeviceContext::getDeviceMemoryInfo() const {
  return device().getMemoryInfo();
}

ErrorOr<void *>
CpuDeviceContext::getHostAccessiblePointer(const void *devicePtr) {
  if (devicePtr == nullptr)
    RETURN_ERROR("getHostAccessiblePointer: pointer is null");
  return const_cast<void *>(devicePtr);
}

MODULAR_CXX_EXPORT const CpuDeviceContext &
CpuDeviceContext::cast(const DeviceContext &ctx) {
  return DeviceContext::cast<CpuDeviceContext>(ctx);
}

} // namespace M::Driver
