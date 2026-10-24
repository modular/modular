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
// This file declares the CUDA device context.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H

#include "CUDADriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "NVML.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::Driver {

class CUDADevice;
class CUDADeviceContext;
class CUDAStreamPool;
class CUDADeviceEvent;
class CUDADeviceGraph;
class CUDAModuleFunction;
class CUDADeviceFunction;
class CUDADeviceFunctionArguments;
class CUDADeviceMulticastBuffer;
class CUDADeviceStream;
class CUDADeviceTimer;

using CUDADeviceRef = RCRef<CUDADevice>;
using CUDADeviceContextRef = RCRef<CUDADeviceContext>;
using CUDAStreamPoolRef = RCRef<CUDAStreamPool>;
using CUDADeviceEventRef = RCRef<CUDADeviceEvent>;
using CUDADeviceGraphRef = RCRef<CUDADeviceGraph>;
using CUDAModuleFunctionRef = RCRef<const CUDAModuleFunction>;
using CUDADeviceFunctionRef = RCRef<CUDADeviceFunction>;
using CUDADeviceFunctionArgumentsRef = RCRef<const CUDADeviceFunctionArguments>;
using CUDADeviceMulticastBufferRef = RCRef<CUDADeviceMulticastBuffer>;
using CUDADeviceStreamRef = RCRef<CUDADeviceStream>;
using CUDADeviceTimerRef = RCRef<CUDADeviceTimer>;

class CUDADevice : public Device {
public:
  static constexpr API api = Device::cudaAPI;

  using DeviceList = std::vector<CUDADevice *>;
  CUDADevice(const CUDADriver &driver, int id, nvmlDevice_t nvmlDevice,
             CUdevice cudaDevice, CUcontext context, std::string name,
             int computeCapabilityMajor, int computeCapabilityMinor,
             Device::MemInfo memInfo, size_t allocAlignment);
  ~CUDADevice() override;

  const CUDADriver &cudaDriver() const;
  CUdevice cudaDevice() const;
  nvmlDevice_t *getNvmlDevice();
  ErrorOr<CUcontext> cudaContext() const;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;

  // Enable peer access between this device and the peer device.
  ErrorOrSuccess enablePeerAccessImpl(const Device &peer) override;

  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<std::string> getPciBusId() const override;

  ErrorOr<bool> supportsMulticast() const override;

  ErrorOr<bool> isHostUnified() const override;

  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override;

  // CUDA Virtual Memory Management (VMM) for the defragmenting allocator.
  // Probes the VMM allocation-granularity query once and caches whether it
  // succeeds, so this reflects whether VMM is actually usable (not just that
  // the backend is CUDA).
  bool vmmSupported() const override;
  // VMM is enabled by default on NVIDIA GPUs (see base class).
  bool vmmEnabledByDefault() const override { return true; }
  ErrorOr<size_t> vmmGranularity() const override;
  ErrorOr<void *> vmmAddressReserve(size_t byteSize) override;
  ErrorOrSuccess vmmAddressFree(void *va, size_t byteSize) override;
  ErrorOr<uint64_t> vmmCreate(size_t byteSize) override;
  ErrorOrSuccess vmmRelease(uint64_t handle) override;
  ErrorOrSuccess vmmMapAndSetAccess(void *va, size_t byteSize,
                                    uint64_t handle) override;
  ErrorOrSuccess vmmUnmap(void *va, size_t byteSize) override;

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;
  ErrorOr<AllocationRange> allocationRange(void *ptr) const override;

  ErrorOr<Device::GraphMemInfo> getGraphMemoryInfo() const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) override;

  ErrorOr<DeviceContextRef> createContext() override;

  ErrorOr<const char *> getArchName() const override;

  static const CUDADevice &cast(const Device &device);
  static CUDADevice &cast(Device &device);

  static ErrorOr<std::string> getFirstArchName();

  static ErrorOr<int> numberOfGPUs();
  static ErrorOr<CUDADevice &> get(int id);

  ErrorOrSuccess runHealthcheck() const override;

  DLDevice getDLDevice(bool pinned) const override {
    return pinned ? DLDevice{kDLCUDAHost, 0} : DLDevice{kDLCUDA, id()};
  }

  ErrorOr<CUevent> eventCachePop(bool useCache = true) const;
  void eventCachePush(CUevent event) const;

private:
  void eventCacheClear() const;

  // CUmemAccessDesc list (owning device + every P2P-capable peer) used to grant
  // access to VMM mappings. VMM access does NOT inherit from
  // cuCtxEnablePeerAccess, so each mapping must be made accessible to every
  // device that may read it (e.g. multi-GPU expert-parallel collectives).
  // Computed once and cached.
  const std::vector<CUmemAccessDesc> &vmmAccessDescs() const;

  // Allocation properties for VMM physical chunks on this device, including
  // the GPUDirect-RDMA-capable flag when the platform supports it. Computed
  // once and cached.
  const CUmemAllocationProp &vmmAllocProp() const;

  // CUDA 13 only supports Turing (sm_75) architectures and later.
  static constexpr int MIN_COMPUTE_CAPABILITY = 75;

  // We require CUDA 13 and NVIDIA driver 580 or above.
  // CUDA 13 raised the minimum driver requirement to 580 and dropped
  // support for pre-Turing hardware.
  //
  // NVIDIA driver to CUDA version mapping:
  // https://docs.nvidia.com/deploy/cuda-compatibility/index.html#
  // use-the-right-compat-package
  //
  // Users on older drivers or hardware can set MODULAR_NVPTX_COMPILER_PATH
  // to use an external ptxas binary instead of the bundled libnvptxcompiler.
  static constexpr int MIN_CUDA_DRIVER_VERSION = 13000;
  static constexpr int MIN_NVIDIA_DRIVER_VERSION = 580;

  const CUDADriver &driver_;
  const CUdevice cudaDevice_;
  nvmlDevice_t nvmlDevice_;

  // Lazily-computed, cached VMM peer-access descriptor list (see
  // vmmAccessDescs()).
  mutable std::once_flag vmmAccessOnce_;
  mutable std::vector<CUmemAccessDesc> vmmAccessDescs_;

  // Lazily-computed, cached VMM allocation properties (see vmmAllocProp()).
  mutable std::once_flag vmmAllocPropOnce_;
  mutable CUmemAllocationProp vmmAllocProp_ = {};

  // Lazily-computed, cached result of whether VMM is usable on this device (see
  // vmmSupported()).
  mutable std::once_flag vmmSupportedOnce_;
  mutable bool vmmSupportedCached_ = false;
  // Device context to be shared across all `CUDADeviceContext` objects. We
  // initialize this with the primary device context for the device to be able
  // to share resources with libraries using the CUDA Runtime API.
  CUcontext context_;

  const int computeCapability_;
  std::string targetArch_;

  // Mutex to protect mutable data.
  mutable std::mutex mu_;
  // Non-timing enabled events which can be reused for record/wait pairs.
  mutable std::vector<CUevent> eventCache_;
};

class CUDADeviceMulticastBuffer : public DeviceMulticastBuffer {
public:
  // Per-device allocation backing a multicast buffer.
  struct DeviceToBuffer {
    const CUdevice device_;
    const CUmemGenericAllocationHandle handle_;
    const CUdeviceptr ptr_;
  };

  CUDADeviceMulticastBuffer(
      size_t byteSize, size_t paddedByteSize, const CUDADriver &driver,
      CUmemGenericAllocationHandle multicastHandle, CUdeviceptr multicastPtr,
      std::vector<CUDADeviceMulticastBuffer::DeviceToBuffer> unicastBuffers);

  ~CUDADeviceMulticastBuffer() override;

  ErrorOr<DeviceBufferRef> multicastBuffer(DeviceContext &ctx) const override;
  ErrorOr<DeviceBufferRef> unicastBufferFor(DeviceContext &ctx) const override;

  static ErrorOr<DeviceMulticastBufferRef>
  allocate(const std::vector<const Device *> &devices, size_t byteSize);

private:
  const CUDADriver &driver_;
  CUmemGenericAllocationHandle multicastHandle_;
  CUdeviceptr multicastPtr_;
  size_t paddedByteSize_;
  std::vector<DeviceToBuffer> unicastBuffers_;
};

class CUDADeviceEvent : public DeviceEvent {
public:
  CUDADeviceEvent(CUDADevice &device, CUevent event, bool recordedDuringCapture,
                  bool enableTiming = false);
  ~CUDADeviceEvent() override;

  ErrorOrSuccess synchronize() override;

  ErrorOr<bool> isReady() const override;

  ErrorOr<float> elapsedTime(const DeviceEvent &end) const override;

  CUevent cudaEvent() const;

  static const CUDADeviceEvent &cast(const DeviceEvent &event);

private:
  CUevent event_;
  bool recordedDuringCapture_;
  bool enableTiming_;
};

class CUDADeviceGraph : public DeviceGraph {
public:
  /// Construct an uninstantiated graph from a raw CUgraph returned by
  /// cuStreamEndCapture. Call instantiate() before replay().
  CUDADeviceGraph(DeviceContextRef context, CUgraph rawGraph,
                  DeviceStreamRef stream)
      : DeviceGraph(std::move(context), std::move(stream)),
        rawGraph_(rawGraph) {}

  ~CUDADeviceGraph() override;

  /// Instantiate the captured CUgraph into a CUgraphExec. Expensive; safe to
  /// call in parallel with instantiation of other graphs.
  ErrorOrSuccess instantiate() override;

  /// Replay using the stored stream (set during capture).
  ErrorOrSuccess replay() const override;

  /// Check if this graph has a stored stream for replay.
  bool hasStream() const { return hasCaptureStream(); }

private:
  CUgraph rawGraph_ = nullptr; // non-null until instantiate() is called
  CUgraphExec exec_ = nullptr; // non-null after instantiate() succeeds
};

class CUDADeviceStream : public DeviceStream {
public:
  CUDADeviceStream(CUDADeviceContext &ctx, CUstream stream);

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

  void beginLaunchTrace() const override;
  std::vector<KernelLaunchMetadata> takeLaunchTrace() const override;

  CUstream cudaStream() const;

  static const CUDADeviceStream *verified_cast(const DeviceStream *stream);

private:
  /// Query the current stream capture status.
  CUstreamCaptureStatus getCaptureStatus() const;

  CUstream stream_;

  StringRef logGraphCapture_ = {};
};

class CUDAModuleFunction : public DeviceFunctionCacheEntry {
public:
  CUDAModuleFunction(const std::string name, const CUDADriver &driver,
                     CUmodule module, CUfunction function);
  ~CUDAModuleFunction();

  const CUDADriver &cudaDriver() const;
  CUmodule cudaModule() const;
  CUfunction cudaFunction() const;
  ArrayRef<size_t> paramSizes() const;

  static const CUDAModuleFunction *cast(const DeviceFunctionCacheEntry *entry);

private:
  const CUDADriver &driver_;
  const CUmodule module_;
  const CUfunction function_;
  mutable std::once_flag paramSizesOnce_;
  mutable std::vector<size_t> paramSizes_;
};

class CUDADeviceFunction : public DeviceFunction {
public:
  CUDADeviceFunction(CUDADeviceContext &context, CUDAModuleFunctionRef modFunc);
  ~CUDADeviceFunction() override;

  const std::string &name() const override;

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byteSize) override;

  ErrorOr<int32_t> getAttribute(int32_t code) override;

  ErrorOrSuccess
  cudaCallKernelDirect(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                       uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                       uint32_t sharedMemBytes, CUlaunchAttribute *attrs,
                       uint32_t numAttrs, void **args, CUstream stream) const;

  ErrorOrSuccess
  occupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, int blockSize,
                                            size_t dynamicSharedMemSize);

  CUfunction cudaFunction() const;
  CUmodule cudaModule() const;

  static const CUDADeviceFunction &cast(const DeviceFunction &func);
  static CUDADeviceFunction &cast(DeviceFunction &func);

private:
  // The moduleFunction_ holds the CUDA specific implementation. Methods on
  // CUDADeviceFunction forward to moduleFunction_ as needed.
  // This layout is needed to break the reference cycle between the function
  // cache in the Device and the cached functions.
  CUDAModuleFunctionRef moduleFunction_;
};

class CUDADeviceTimer : public DeviceTimer {
public:
  CUDADeviceTimer(CUDADeviceContext &context, CUevent start, CUevent stop);
  ~CUDADeviceTimer() override;

  CUevent cudaStartEvent() const;
  CUevent cudaStopEvent() const;

  static const CUDADeviceTimer &cast(const DeviceTimer &timer);

private:
  CUevent start_;
  CUevent stop_;
};

/// A reference-counted set of CUDA streams owned by a `CUDADeviceContext`.
/// Stream 0 is the base/default stream the context was constructed with;
/// additional streams are appended on demand via `createStream`. The pool owns
/// every stream it holds and synchronizes + destroys them when the last
/// co-owner is released.
class CUDAStreamPool : public ReferenceCounted<CUDAStreamPool> {
public:
  CUDAStreamPool(const CUDADriver &driver, CUcontext context, CUstream base);
  ~CUDAStreamPool();

  /// Returns the stream at index `i`. REQUIRES: i < size().
  CUstream get(unsigned i) const;

  /// Number of streams currently in the pool.
  unsigned size() const;

  /// Appends a stream to the pool, returning the index of the new stream. The
  /// pool takes ownership of the stream. Thread-safe.
  unsigned emplace_back(CUstream stream);

private:
  const CUDADriver &driver_;
  // The CUcontext the streams belong to; needed to bind the context while
  // synchronizing/destroying the streams.
  CUcontext context_;
  mutable std::mutex mu_;
  std::vector<CUstream> streams_;
};

class CUDADeviceContext : public DeviceContext {
public:
  CUDADeviceContext(CUDADevice &device, CUcontext context, CUstream stream);
  ~CUDADeviceContext() override;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;
  /// Issues the whole batch as one `cuMemcpyBatchAsync` (CUDA 12.8+) on this
  /// stream, letting UVA resolve each pointer. Falls back to the scalar loop
  /// without that entry point, or when a peer source cannot be reached
  /// directly.
  ErrorOrSuccess memBatchCopy(llvm::ArrayRef<DeviceBuffer *> dsts,
                              llvm::ArrayRef<DeviceBuffer *> srcs) override;

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override;

  KernelLoadGate *kernelLoadGate() override {
    return &driver_.kernelLoadGate();
  }

  // DeviceFunction loading.
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
  bool flushIdleStreamViews() override;
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

  bool isGraphCapturing() const override;

  ErrorOr<CompletionFlagRef> allocateCompletionFlag() override;
  ErrorOrSuccess freeCompletionFlag(void *hostPtr) override;

  bool ownsDeviceAddress(const void *ptr) const override;

  const CUDADriver &cudaDriver() const;
  CUcontext cudaContext() const;
  CUstream cudaStream() const;

  ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) override;

  ErrorOrSuccess cudaTensorMapEncodeTiled(
      void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
      const DeviceBuffer &globalAddress, const uint64_t *globalDim,
      const uint64_t *globalStrides, const uint32_t *boxDim,
      const uint32_t *elementStrides, CUtensorMapInterleave interleave,
      CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
      CUtensorMapFloatOOBfill oobFill);

  /// Create a TMA descriptor for im2col transformation.
  /// Used for implicit GEMM convolution where the TMA hardware performs
  /// the im2col coordinate transformation on-the-fly.
  ErrorOrSuccess cudaTensorMapEncodeIm2col(
      void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
      const DeviceBuffer &globalAddress, const uint64_t *globalDim,
      const uint64_t *globalStrides, const int *pixelBoxLowerCorner,
      const int *pixelBoxUpperCorner, uint32_t channelsPerPixel,
      uint32_t pixelsPerColumn, const uint32_t *elementStrides,
      CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
      CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);

  static const CUDADeviceContext &cast(const DeviceContext &ctx);
  static CUDADeviceContext &cast(DeviceContext &ctx);
  static const CUDADeviceContext *verified_cast(const DeviceContext *ctx);

private:
  // Private constructor for `selectStream` views: shares an existing stream
  // pool (and therefore the base context's streams + CUcontext) and only
  // differs in which pooled stream is active.
  CUDADeviceContext(CUDADevice &device, CUcontext context,
                    CUDAStreamPoolRef streamPool, unsigned activeStreamId);

  const CUDADriver &driver_;
  const CUcontext context_;

  // The set of streams owned by this context. Stream 0 is the base stream; a
  // `selectStream` view co-owns the same pool and selects another index.
  // `createStream` appends additional streams. The pool owns the streams and
  // destroys them when the last co-owner is released.
  CUDAStreamPoolRef streamPool_;

  // Cached copy of `streamPool_->get(activeStreamId)`: the currently selected
  // stream this context submits to. The base context selects stream 0; a
  // `selectStream` view selects another index into the shared pool. All
  // enqueue/sync/alloc operations submit to this stream.
  CUstream stream_;

  bool logLoadFunction_ = false;
  bool logVerbose_ = false;
  // Stress knob for tests: skips function-cache hits, so every launch is a
  // cold kernel load.
  bool bypassFunctionCache_ = false;

  // `selectStream` views by stream id, kept for this context's lifetime. A
  // captured device graph keeps using the memory its side-stream ops allocated
  // during capture, which lives in the view's `graphFreeList`; a view dropped
  // after one execution would flush that memory to the root free list while
  // the graph still references it. Ordinary frees stay reclaimable through
  // `flushIdleStreamViews`.
  std::mutex streamViewsMu_;
  std::vector<DeviceContextRef> streamViews_;

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

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
