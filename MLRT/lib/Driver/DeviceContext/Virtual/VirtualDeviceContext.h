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
// This file declares the Virtual device context for compile-only mode.
//
// VirtualDeviceContext is a minimal DeviceContext implementation that can be
// used for model compilation without actual hardware. It implements the
// DeviceContext API but most operations are no-ops or return errors if called.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_VIRTUAL_VIRTUALDEVICECONTEXT_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_VIRTUAL_VIRTUALDEVICECONTEXT_H

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace M::Driver {

class VirtualDevice;
class VirtualDeviceContext;
class VirtualDeviceStream;

using VirtualDeviceRef = RCRef<VirtualDevice>;
using VirtualDeviceContextRef = RCRef<VirtualDeviceContext>;
using VirtualDeviceStreamRef = RCRef<VirtualDeviceStream>;

/// Virtual Device for compile-only mode.
///
/// This device mimics GPU devices but doesn't require actual hardware.
/// It's used during compilation to validate graph structure and generate
/// code without needing to execute on real GPUs.
/// Virtual devices use a CompilationDevice for compilation operations.
class VirtualDevice : public Device {
public:
  VirtualDevice(int id, std::string name,
                std::unique_ptr<CompilationDevice> compilationDevice);
  ~VirtualDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOrSuccess runHealthcheck() const override { return success(); }
  ErrorOr<int> computeCapability() const override;
  ErrorOr<const char *> getArchName() const override;

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

  DLDevice getDLDevice(bool /*pinned*/) const override {
    // VirtualDevice is compile-only and does not support DLPack operations.
    assert(false && "DLPack not supported on virtual (compile-only) devices");
    return {};
  }

  static const VirtualDevice &cast(const Device &device);

private:
  // Compilation device used for code generation (e.g., PTX→CUBIN)
  std::unique_ptr<CompilationDevice> compilationDevice_;
};

/// Virtual Device Stream for compile-only mode.
class VirtualDeviceStream : public DeviceStream {
public:
  explicit VirtualDeviceStream(VirtualDeviceContext &ctx);
  ~VirtualDeviceStream() override = default;

  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) const override;

  // The base class refuses these with a generic message. Overriding them names
  // the virtual device, which is how a caller tells "needs hardware" apart
  // from a failure.
  ErrorOrSuccess enqueueHostFunc(void (*fn)(void *),
                                 void *userData) const override;
  ErrorOrSuccess enqueueWaitOnHostValue(const CompletionFlag &flag,
                                        uint64_t value) const override;
  ErrorOrSuccess beginCapture() const override;
  ErrorOr<DeviceGraphRef> endCapture() override;

  void *ptr() const override;

  static const VirtualDeviceStream *verified_cast(const DeviceStream *stream);
};

/// Virtual Device Context for compile-only mode.
///
/// This context provides minimal functionality needed for compilation:
/// - Memory allocation tracking (doesn't actually allocate)
/// - Stream/event creation (returns stub objects)
/// - Function compilation (uses CompilationDevice)
/// - Synchronization (no-ops)
///
/// Compilation operations use the VirtualDevice's CompilationDevice,
/// which can compile code without requiring GPU hardware.
class VirtualDeviceContext : public DeviceContext {
public:
  explicit VirtualDeviceContext(VirtualDevice &device);
  ~VirtualDeviceContext() override;

  // DeviceBuffer copies - not implemented for virtual context
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override;

  // DeviceFunction compiling - minimal implementation
  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes, StringRef debugLevel,
               int32_t optimizationLevel) override;

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) override;

  // Synchronize asynchronous operations - no-ops
  ErrorOr<DeviceStreamRef> stream() const override;
  ErrorOr<DeviceStreamRef> createStream(int priority) override;

  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) override;

  ErrorOr<DeviceTimerRef> startTimer() override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) override;

  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  // Refused by name rather than by the base class's generic message; see
  // `VirtualDeviceStream`.
  ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream) override;
  ErrorOr<DeviceEventRef> eventCreate(bool enableTiming) override;
  ErrorOr<DeviceGraphBuilderRef>
  createGraphBuilder(DeviceGraphMemoryPoolRef pool) override;
  ErrorOr<CompletionFlagRef> allocateCompletionFlag() override;
  ErrorOrSuccess freeCompletionFlag(void *hostPtr) override;
  ErrorOr<void *> getHostAccessiblePointer(const void *devicePtr) override;

  static const VirtualDeviceContext &cast(const DeviceContext &ctx);
  static const VirtualDeviceContext *verified_cast(const DeviceContext *ctx);

private:
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) override;
  ErrorOrSuccess memFree(void *ptr, Device::FreeKind kind) override;
  ErrorOrSuccess memWrap(void **devPtr, void *hostPtr,
                         size_t byteSize) override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_VIRTUAL_VIRTUALDEVICECONTEXT_H
