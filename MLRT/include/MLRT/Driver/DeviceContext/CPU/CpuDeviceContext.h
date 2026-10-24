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
// This file declares the CPU device context.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
#define MLRT_DRIVER_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stddef.h>

namespace M::Driver {

class CpuDevice;
class CpuDeviceContext;
class CpuDeviceEvent;
class CpuDeviceStream;
class CpuDeviceTimer;

using CpuDeviceRef = RCRef<CpuDevice>;
using CpuDeviceContextRef = RCRef<CpuDeviceContext>;
using CpuDeviceEventRef = RCRef<CpuDeviceEvent>;
using CpuDeviceStreamRef = RCRef<CpuDeviceStream>;
using CpuDeviceTimerRef = RCRef<CpuDeviceTimer>;

class CpuDevice : public Device {
public:
  static constexpr API api = Device::cpuAPI;

  CpuDevice(Device::MemInfo memInfo = Device::MemInfo());
  ~CpuDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOrSuccess runHealthcheck() const override { return success(); }
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<std::string> getPciBusId() const override;

  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override;

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) override;

  ErrorOr<DeviceContextRef> createContext() override;

  /// The CPU device is the host, so there is no second pool to unify with.
  ErrorOr<bool> isHostUnified() const override { return true; }

  DLDevice getDLDevice(bool /*pinned*/) const override { return {kDLCPU, 0}; }

  static const CpuDevice &cast(const Device &device);

  static ErrorOr<CpuDevice &> get(int id);
};

class CpuDeviceStream : public DeviceStream {
public:
  CpuDeviceStream(CpuDeviceContext &ctx);
  ~CpuDeviceStream() override = default;

  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) const override;

  void *ptr() const override;
};

class CpuDeviceTimer : public DeviceTimer {
public:
  CpuDeviceTimer(CpuDeviceContext &context,
                 std::chrono::time_point<std::chrono::steady_clock> start);
  ~CpuDeviceTimer() override = default;

  std::chrono::time_point<std::chrono::steady_clock> cpu_start() const;

  static const CpuDeviceTimer &cast(const DeviceTimer &timer);

private:
  const std::chrono::time_point<std::chrono::steady_clock> start_;
};

class CpuDeviceEvent : public DeviceEvent {
public:
  CpuDeviceEvent(CpuDevice &device, AsyncRT::CPUDeviceRef cpuDevice,
                 AsyncRT::AnyAsyncValueRef value);
  ~CpuDeviceEvent() override = default;

  ErrorOrSuccess synchronize() override;
  ErrorOr<bool> isReady() const override;

  AsyncRT::AnyAsyncValueRef &asyncValue() { return value; }

private:
  AsyncRT::CPUDeviceRef cpuDevice;
  AsyncRT::AnyAsyncValueRef value;
};

class CpuDeviceContext : public DeviceContext {
public:
  explicit CpuDeviceContext(CpuDevice &device);
  ~CpuDeviceContext() override;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override;

  // DeviceFunction loading and execution.
  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes, StringRef debugLevel,
               int32_t optimizationLevel) override;
  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) override;

  ErrorOrSuccess enqueueHostFunction(void (*resumeFunc)(int8_t *),
                                     void (*destroyFunc)(int8_t *),
                                     int8_t *coroHandle) override;

  ErrorOrSuccess enqueueHostFunctionRange(void (*resumeFunc)(int8_t *),
                                          void (*destroyFunc)(int8_t *),
                                          int8_t **coroHandles,
                                          size_t numHandles) override;

  // Synchronize asynchronous operations.
  ErrorOr<DeviceStreamRef> stream() const override;

  ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream) override;

  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) override;
  ErrorOr<DeviceEventRef> eventCreate(bool enableTiming) override;
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) override;

  ErrorOr<DeviceTimerRef> startTimer() override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) override;

  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  /// CPU "device" memory is ordinary host memory, so a device pointer is
  /// already the host-visible address and is returned unchanged.
  ErrorOr<void *> getHostAccessiblePointer(const void *devicePtr) override;

  static const CpuDeviceContext &cast(const DeviceContext &ctx);

  /// Returns the AsyncRT CPUDevice that backs this context's worker
  /// pool. Allows callers (e.g. nanobind code that needs to dispatch
  /// onto an explicit, caller-supplied AsyncRT runtime) to reach the
  /// CPUDevice without going through a global/TLS lookup.
  const AsyncRT::CPUDeviceRef &getCPUDevice() const { return cpuDevice; }

private:
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) override;
  ErrorOrSuccess memFree(void *ptr, Device::FreeKind kind) override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;

  ErrorOrSuccess enqueueWork(AsyncRT::TaskFunction &&work);

  AsyncRT::CPUDeviceRef cpuDevice;

  // Contains the chain of the last enqueued operation, which may already have
  // completed
  AsyncRT::AnyAsyncValueRef streamChain;
  std::mutex streamChainMutex{};
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
