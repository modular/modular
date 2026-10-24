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
// This file declares the HIP device context.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H

#include "HIPDriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::Driver {

class HIPDevice;
class HIPDeviceContext;
class HIPStreamPool;
class HIPDeviceEvent;
class HIPDeviceFunction;
class HIPDeviceFunctionArguments;
class HIPDeviceGraph;
class HIPDeviceStream;
class HIPDeviceTimer;
class HIPModuleFunction;

using HIPDeviceRef = RCRef<HIPDevice>;
using HIPDeviceContextRef = RCRef<HIPDeviceContext>;
using HIPStreamPoolRef = RCRef<HIPStreamPool>;
using HIPDeviceEventRef = RCRef<HIPDeviceEvent>;
using HIPDeviceFunctionRef = RCRef<HIPDeviceFunction>;
using HIPDeviceFunctionArgumentsRef = RCRef<const HIPDeviceFunctionArguments>;
using HIPDeviceGraphRef = RCRef<HIPDeviceGraph>;
using HIPDeviceStreamRef = RCRef<HIPDeviceStream>;
using HIPDeviceTimerRef = RCRef<HIPDeviceTimer>;
using HIPModuleFunctionRef = RCRef<const HIPModuleFunction>;

class HIPDevice : public Device {
public:
  static constexpr API api = Device::hipAPI;

  using DeviceList = std::vector<HIPDevice *>;
  HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
            std::string name, std::string arch, MemInfo memInfo,
            size_t allocAlignment);
  ~HIPDevice() override;

  ErrorOr<const char *> getArchName() const override;
  static ErrorOr<std::string> getArchName(hipDevice_t device = 0);

  const HIPDriver &hipDriver() const;
  hipDevice_t hipDevice() const;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;

  // Enable peer access between this device and the peer device.
  ErrorOrSuccess enablePeerAccessImpl(const Device &peer) override;

  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<std::string> getPciBusId() const override;

  ErrorOr<bool> isHostUnified() const override;

  static ErrorOr<int> numberOfGPUs();
  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override;

  // HIP Virtual Memory Management (VMM) for the defragmenting allocator.
  // Probes the VMM allocation-granularity query once and caches whether it
  // succeeds, so this reflects whether VMM is actually usable (not just that
  // the backend is HIP).
  bool vmmSupported() const override;
  ErrorOr<size_t> vmmGranularity() const override;
  ErrorOr<void *> vmmAddressReserve(size_t byteSize) override;
  ErrorOrSuccess vmmAddressFree(void *va, size_t byteSize) override;
  ErrorOr<uint64_t> vmmCreate(size_t byteSize) override;
  ErrorOrSuccess vmmRelease(uint64_t handle) override;
  ErrorOrSuccess vmmMapAndSetAccess(void *va, size_t byteSize,
                                    uint64_t handle) override;
  ErrorOrSuccess vmmUnmap(void *va, size_t byteSize) override;

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) override;

  ErrorOr<DeviceContextRef> createContext() override;

  static const HIPDevice &cast(const Device &device);
  static HIPDevice &cast(Device &device);
  static const HIPDevice *verified_cast(const Device *device);

  static ErrorOr<HIPDevice &> get(int id);

  ErrorOrSuccess runHealthcheck() const override;

  DLDevice getDLDevice(bool pinned) const override {
    return pinned ? DLDevice{kDLROCMHost, 0} : DLDevice{kDLROCM, id()};
  }

private:
  // hipMemAccessDesc list (owning device + every P2P-capable peer) used to
  // grant access to VMM mappings. VMM access does NOT inherit from
  // hipDeviceEnablePeerAccess, so each mapping must be made accessible to every
  // device that may read it (e.g. multi-GPU expert-parallel collectives).
  // Computed once and cached.
  const std::vector<hipMemAccessDesc> &vmmAccessDescs() const;

  // We only support HIP architectures sm_80 and above.
  static constexpr int MIN_COMPUTE_CAPABILITY = 80;

  // We only support HIP versions 6.3.30000 and above.
  static constexpr int MIN_HIP_DRIVER_VERSION_MAJOR = 6;
  static constexpr int MIN_HIP_DRIVER_VERSION_MINOR = 3;
  static constexpr int MIN_HIP_DRIVER_VERSION_PATCH = 30000;

  const std::string arch_;
  const HIPDriver &driver_;
  hipDevice_t device_;

  // Lazily-computed, cached VMM peer-access descriptor list (see
  // vmmAccessDescs()).
  mutable std::once_flag vmmAccessOnce_;
  mutable std::vector<hipMemAccessDesc> vmmAccessDescs_;

  // Lazily-computed, cached result of whether VMM is usable on this device (see
  // vmmSupported()).
  mutable std::once_flag vmmSupportedOnce_;
  mutable bool vmmSupportedCached_ = false;
};

class HIPDeviceEvent : public DeviceEvent {
public:
  HIPDeviceEvent(HIPDevice &device, hipEvent_t event,
                 bool recordedDuringCapture);
  ~HIPDeviceEvent() override;

  ErrorOrSuccess synchronize() override;

  ErrorOr<bool> isReady() const override;

  ErrorOr<float> elapsedTime(const DeviceEvent &end) const override;

  hipEvent_t hipEvent() const;

  static const HIPDeviceEvent &cast(const DeviceEvent &event);

private:
  hipEvent_t event_;
  bool recordedDuringCapture_;
};

class HIPDeviceGraph : public DeviceGraph {
public:
  /// Construct an uninstantiated graph from a raw hipGraph_t returned by
  /// hipStreamEndCapture. Call instantiate() before replay().
  HIPDeviceGraph(DeviceContextRef context, hipGraph_t rawGraph,
                 DeviceStreamRef stream)
      : DeviceGraph(std::move(context), std::move(stream)),
        rawGraph_(rawGraph) {}

  ~HIPDeviceGraph() override;

  /// Instantiate the captured hipGraph_t into a hipGraphExec_t. Expensive;
  /// safe to call in parallel with instantiation of other graphs.
  ErrorOrSuccess instantiate() override;

  /// Replay using the stored stream (set during capture).
  ErrorOrSuccess replay() const override;

  /// Check if this graph has a stored stream for replay.
  bool hasStream() const { return hasCaptureStream(); }

private:
  hipGraph_t rawGraph_ = nullptr; // non-null until instantiate() is called
  hipGraphExec_t exec_ = nullptr; // non-null after instantiate() succeeds
};

class HIPDeviceStream : public DeviceStream {
public:
  HIPDeviceStream(HIPDeviceContext &ctx, hipStream_t stream);

  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueHostFunc(void (*fn)(void *),
                                 void *userData) const override;

  ErrorOrSuccess enqueueWaitOnHostValue(const CompletionFlag &flag,
                                        uint64_t value) const override;

  virtual ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) const override;

  void *ptr() const override;

  ErrorOrSuccess beginCapture() const override;
  ErrorOr<DeviceGraphRef> endCapture() override;

  // Launch trace capture methods
  void beginLaunchTrace() const override;
  std::vector<KernelLaunchMetadata> takeLaunchTrace() const override;

  hipStream_t hipStream() const;

  static const HIPDeviceStream *verified_cast(const DeviceStream *stream);

private:
  /// Query the current HIP stream capture status.
  hipStreamCaptureStatus getCaptureStatus() const;

  hipStream_t stream_;

  StringRef logGraphCapture_ = {};
};

class HIPModuleFunction : public DeviceFunctionCacheEntry {
public:
  HIPModuleFunction(const std::string name, const HIPDriver &driver,
                    hipModule_t module, hipFunction_t function);
  ~HIPModuleFunction();

  const HIPDriver &hipDriver() const;
  hipModule_t hipModule() const;
  hipFunction_t hipFunction() const;

  static const HIPModuleFunction *cast(const DeviceFunctionCacheEntry *entry);

private:
  const HIPDriver &driver_;
  hipModule_t module_;
  hipFunction_t function_;
};

class HIPDeviceFunction : public DeviceFunction {
public:
  HIPDeviceFunction(HIPDeviceContext &context, HIPModuleFunctionRef modFunc);
  ~HIPDeviceFunction() override;

  const std::string &name() const override;

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byteSize) override;

  ErrorOr<int32_t> getAttribute(int32_t code) override;

  ErrorOrSuccess hipCallKernelDirect(uint32_t gridX, uint32_t gridY,
                                     uint32_t gridZ, uint32_t blockX,
                                     uint32_t blockY, uint32_t blockZ,
                                     uint32_t sharedMemBytes, void **args,
                                     hipStream_t stream) const;

  static HIPDeviceFunction &cast(DeviceFunction &func);
  static const HIPDeviceFunction &cast(const DeviceFunction &func);

  ErrorOrSuccess
  occupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, int blockSize,
                                            size_t dynamicSharedMemSize);

  hipFunction_t hipFunction() const;
  hipModule_t hipModule() const;

private:
  // The moduleFunction_ holds the HIP specific implementation. Methods on
  // HIPDeviceFunction forward to moduleFunction_ as needed.
  // This layout is needed to break the reference cycle between the function
  // cache in the Device and the cached functions.
  HIPModuleFunctionRef moduleFunction_;
};

class HIPDeviceTimer : public DeviceTimer {
public:
  HIPDeviceTimer(HIPDeviceContext &context, hipEvent_t start, hipEvent_t stop);
  ~HIPDeviceTimer() override;

  hipEvent_t hipStartEvent() const;
  hipEvent_t hipStopEvent() const;

  static const HIPDeviceTimer &cast(const DeviceTimer &timer);

private:
  hipEvent_t start_;
  hipEvent_t stop_;
};

/// A reference-counted set of HIP streams owned by a `HIPDeviceContext`.
/// Stream 0 is the base/default stream the context was constructed with;
/// additional streams are appended on demand via `createStream`. The pool owns
/// every stream it holds and synchronizes + destroys them when the last
/// co-owner is released.
class HIPStreamPool : public ReferenceCounted<HIPStreamPool> {
public:
  HIPStreamPool(const HIPDriver &driver, hipDevice_t device, hipStream_t base);
  ~HIPStreamPool();

  /// Returns the stream at index `i`. REQUIRES: i < size().
  hipStream_t get(unsigned i) const;

  /// Number of streams currently in the pool.
  unsigned size() const;

  /// Appends a stream to the pool, returning the index of the new stream. The
  /// pool takes ownership of the stream. Thread-safe.
  unsigned emplace_back(hipStream_t stream);

private:
  const HIPDriver &driver_;
  // The device the streams belong to; needed to bind the device while
  // synchronizing/destroying the streams.
  hipDevice_t device_;
  mutable std::mutex mu_;
  std::vector<hipStream_t> streams_;
};

class HIPDeviceContext : public DeviceContext {
public:
  HIPDeviceContext(HIPDevice &device, hipStream_t stream);
  ~HIPDeviceContext() override;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;
  ErrorOrSuccess memBatchCopy(llvm::ArrayRef<DeviceBuffer *> dsts,
                              llvm::ArrayRef<DeviceBuffer *> srcs) override;

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

  // Synchronize asynchronous operations.
  std::optional<intptr_t> getDLPackStream() const override;
  ErrorOr<DeviceStreamRef> stream() const override;
  unsigned numStreams() const override;
  ErrorOrSuccess streamPriorityRange(int *leastPriority,
                                     int *greatestPriority) override;
  ErrorOr<DeviceStreamRef> createStream(int priority = 0) override;
  ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream) override;
  ErrorOr<DeviceContextRef> selectStream(unsigned streamId) override;
  ErrorOr<DeviceGraphBuilderRef>
  createGraphBuilder(DeviceGraphMemoryPoolRef pool) override;

  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) override;
  ErrorOr<DeviceEventRef> eventCreate(bool enableTiming) override;
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) override;

  ErrorOr<DeviceTimerRef> startTimer() override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) override;

  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) override;

  bool isGraphCapturing() const override;

  ErrorOr<CompletionFlagRef> allocateCompletionFlag() override;
  ErrorOrSuccess freeCompletionFlag(void *hostPtr) override;

  const HIPDriver &hipDriver() const;
  hipDevice_t hipDevice() const;
  hipStream_t hipStream() const;

  static HIPDeviceContext &cast(DeviceContext &ctx);
  static const HIPDeviceContext &cast(const DeviceContext &ctx);
  static const HIPDeviceContext *verified_cast(const DeviceContext *ctx);

private:
  friend class HIPDeviceBuffer;

  // Private constructor for `selectStream` views: shares an existing stream
  // pool (and therefore the base context's streams + device) and only differs
  // in which pooled stream is active.
  HIPDeviceContext(HIPDevice &device, HIPStreamPoolRef streamPool,
                   unsigned activeStreamId);

  const HIPDriver &driver_;
  hipDevice_t device_;

  // The set of streams owned by this context. Stream 0 is the base stream this
  // context submits to; `createStream` appends additional streams. The pool
  // owns the streams and destroys them when the context is released.
  HIPStreamPoolRef streamPool_;

  // Cached handle of the base stream (streamPool_->get(0)). All enqueue/sync/
  // alloc operations submit to this stream.
  hipStream_t stream_;

  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) override;
  ErrorOrSuccess memWrap(void **devPtr, void *hostPtr,
                         size_t byteSize) override;
  ErrorOrSuccess memFree(void *ptr, Device::FreeKind kind) override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
