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
// HAL replacement for the DeviceContext API. Under --config=hal-device-context
// DeviceContext.h dispatches to this header instead of the legacy
// declarations: same
// `M::Driver` class names and drop-in signatures, implemented as a pure shim
// over the Mojo HAL `DeviceContext` via the MojoHAL_* C-ABI exports
// (MLRT/lib/Driver/DeviceContext/HAL/hal_device_context_exports.mojo).
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_HALDEVICECONTEXT_H
#define MLRT_DRIVER_DEVICECONTEXT_HALDEVICECONTEXT_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "CompilationDevice.h"
#include "ErrorMacros.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <dlpack/dlpack.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Opaque handles to the heap allocated wrapped Mojo values the MojoHAL_*
// exports create and release. The Mojo side types the same handles as
// `_DeviceContextBoxPtr` etc. in hal_device_context_exports.mojo.
struct MojoHALContextBox;
struct MojoHALBufferBox;
struct MojoHALScopeBox;
struct MojoHALCompletionFlagBox;
struct MojoHALStreamBox;
struct MojoHALEventBox;

namespace M {
namespace Driver {

using llvm::StringLiteral;
using llvm::StringRef;

class CompletionFlag;
class Device;
class KernelLoadGate;
class DeviceBuffer;
class DeviceContext;
class DeviceEvent;
class DeviceGraph;
class DeviceStream;

using CompletionFlagRef = RCRef<CompletionFlag>;
using DeviceRef = RCRef<Device>;
using DeviceBufferRef = RCRef<DeviceBuffer>;
using DeviceContextRef = RCRef<DeviceContext>;
using DeviceEventRef = RCRef<DeviceEvent>;
using DeviceGraphRef = RCRef<DeviceGraph>;
using DeviceStreamRef = RCRef<DeviceStream>;

using AsyncDeviceGraphRef = AsyncValueRef<DeviceGraphRef>;

/// Device enumeration and identity.
class MODULAR_CXX_EXPORT Device final : public ReferenceCounted<Device> {
public:
  // API is a constexpr StringLiteral for fast comparison, while being
  // printable as a string.
  using API = StringLiteral;
  static constexpr API cpuAPI = "cpu";     // Generic CPU device
  static constexpr API gpuAPI = "gpu";     // Generic GPU device
  static constexpr API cudaAPI = "cuda";   // NVIDIA CUDA device
  static constexpr API hipAPI = "hip";     // AMDGPU HIP device
  static constexpr API metalAPI = "metal"; // Apple Metal device

  struct MemInfo {
    MemInfo() : total(0), free(0) {}
    size_t total;
    size_t free;
  };

  struct GraphMemInfo {
    size_t reserved = 0; // Physical memory reserved by graph pool
    size_t used = 0;     // Physical memory actively mapped by graphs
  };

  // Numeric device attribute, by CUDA-style `CUdevice_attribute` code (see the
  // out-of-line definition below). Queried via `getAttribute`.
  enum class Attribute;

  /// Returns the number of devices available that support `api`.
  static int numberOfDevices(StringRef api);

  /// Creates a device handle for device `id` of `api`.
  static ErrorOr<DeviceRef> create(StringRef api, int id = 0);

  /// Returns the vendor and architecture of the first accelerator in the form
  /// `vendor:arch` (for example `nvidia:sm_90a`), or an empty string when no
  /// accelerator is detected.
  static const std::string &getAcceleratorArchOrEmpty();

  /// Preferred allocation alignment for a device label. GPU-backed labels use
  /// the same hard-coded 256 as the legacy implementation; anything else
  /// falls back to the host's preferred memory alignment.
  static size_t preferredAlignment(StringRef api) {
    if (api == gpuAPI || api == cudaAPI || api == hipAPI || api == metalAPI)
      return 256;
    return kPreferredMemoryAlignment;
  }

  ~Device();

  /// Returns the device API as one of the `API` constants. Plugin APIs
  /// without a dedicated constant (for example the mock test plugin) report
  /// the generic `gpuAPI`.
  API api() const { return api_; }
  int id() const { return id_; }
  /// Whether this is a virtual (compile-only) device.
  bool isVirtual() const { return compilationDevice_ != nullptr; }
  ErrorOr<const char *> getArchName() const;
  const std::string &name() const { return name_; }
  ErrorOrSuccess isCompatible() const;
  ErrorOr<MemInfo> getMemoryInfo() const;
  /// Queries a numeric device attribute. `attr` is forwarded to the plugin as
  /// a CUDA-style `CUdevice_attribute` code; non-CUDA plugins translate it.
  ErrorOr<int> getAttribute(Attribute attr) const;
  /// Memory held by device-graph pools. The HAL backend does not support
  /// device graphs, so no graph pool ever exists and both counters are zero.
  ErrorOr<GraphMemInfo> getGraphMemoryInfo() const { return GraphMemInfo(); }
  /// The plugin's `max_allocation_size` property, else total device memory.
  ErrorOr<size_t> maxSingleAllocationSize() const;
  /// Mirrors `M::Driver::Device::AllocationRange`. The HAL plugin ABI has no
  /// allocation-range query, so `allocationRange` always errors.
  struct AllocationRange {
    void *base;
    size_t size;
  };
  ErrorOr<AllocationRange> allocationRange(void *ptr) const;
  /// Whether this device can directly access `other`'s memory.
  ErrorOr<bool> canAccess(const Device &other) const;
  /// Answered by the plugin's `host_unified` property; virtual devices
  /// report false.
  ErrorOr<bool> isHostUnified() const;

  /// Creates a new context (with its own stream) on this device.
  ErrorOr<DeviceContextRef> createContext();

  /// DLPack device descriptor for this device. Mirrors the legacy
  /// per-backend mappings; APIs without a DLPack code (for example the mock
  /// test plugin's generic gpu) report `kDLExtDev`.
  DLDevice getDLDevice(bool pinned) const {
    if (api_ == cudaAPI)
      return pinned ? DLDevice{kDLCUDAHost, 0} : DLDevice{kDLCUDA, id_};
    if (api_ == hipAPI)
      return pinned ? DLDevice{kDLROCMHost, 0} : DLDevice{kDLROCM, id_};
    if (api_ == metalAPI)
      return pinned ? DLDevice{kDLCPU, 0} : DLDevice{kDLMetal, id_};
    return DLDevice{kDLExtDev, id_};
  }

private:
  friend class DeviceContext;
  Device(MojoHALContextBox *mojoCtx, API api, std::string apiStr, int id,
         std::string name)
      : mojoCtx_(mojoCtx), api_(api), apiStr_(std::move(apiStr)), id_(id),
        name_(std::move(name)) {}

  /// Virtual (compile-only) device: no Mojo HAL box, every device query
  /// answered from `compilationDevice` or a constant.
  Device(std::unique_ptr<CompilationDevice> compilationDevice, API api,
         std::string apiStr, int id, std::string name)
      : mojoCtx_(nullptr), api_(api), apiStr_(std::move(apiStr)), id_(id),
        name_(std::move(name)),
        compilationDevice_(std::move(compilationDevice)) {}

  MojoHALContextBox *mojoCtx_; // DeviceContext owned by this Device.
  API api_;
  std::string apiStr_; // Plugin api string as passed to create().
  int id_;
  std::string name_;
  mutable std::string archName_;
  std::unique_ptr<CompilationDevice> compilationDevice_;
};

// Numeric device attributes, by CUDA-style `CUdevice_attribute` code. Mirrors
// the legacy `Device::Attribute` enum verbatim; the codes are forwarded to
// the driver plugin's `M_driver_device_attribute`.
enum class Device::Attribute : int {
  // Maximum number of threads per block
  MAX_THREADS_PER_BLOCK = 1,
  // Maximum block dimension X
  MAX_BLOCK_DIM_X = 2,
  // Maximum block dimension Y
  MAX_BLOCK_DIM_Y = 3,
  // Maximum block dimension Z
  MAX_BLOCK_DIM_Z = 4,
  // Maximum grid dimension X
  MAX_GRID_DIM_X = 5,
  // Maximum grid dimension Y
  MAX_GRID_DIM_Y = 6,
  // Maximum grid dimension Z
  MAX_GRID_DIM_Z = 7,
  // Maximum shared memory available per block in bytes
  MAX_SHARED_MEMORY_PER_BLOCK = 8,
  // Warp size in threads
  WARP_SIZE = 10,
  // Maximum number of 32-bit registers available per block
  MAX_REGISTERS_PER_BLOCK = 12,
  // Typical clock frequency in kilohertz
  CLOCK_RATE = 13,
  // Number of multiprocessors on device
  MULTIPROCESSOR_COUNT = 16,
  // Maximum resident threads per multiprocessor
  MAX_THREADS_PER_MULTIPROCESSOR = 39,
  // Major compute capability version number
  COMPUTE_CAPABILITY_MAJOR = 75,
  // Minor compute capability version number
  COMPUTE_CAPABILITY_MINOR = 76,
  // Maximum shared memory available per multiprocessor in bytes
  MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  // Maximum number of 32-bit registers available per multiprocessor
  MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  // Device supports launching cooperative kernels
  COOPERATIVE_LAUNCH = 95,
  // Maximum opt-in shared memory per block (CUDA:
  // cudaDevAttrMaxSharedMemoryPerBlockOptin)
  MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  // Maximum resident blocks per multiprocessor
  MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
  // CUDA-only: Maximum value of CUaccessPolicyWindow::num_bytes.
  MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
  // Metal-only: Disable `print` to improve kernel's execution time.
  METAL_DISABLE_LOGGING = 110,
};

/// Deprecated free-function form of `Device::preferredAlignment`.
inline size_t getPreferredAlignment(std::string label) {
  return Device::preferredAlignment(label);
}

// Virtual CPU target for CPU kernel compilation (empty = host CPU). Accepts an
// LLVM CPU name (or "generic"); must be set before compiling CPU kernels.
// Thread-safe. Backend-independent host configuration, identical to the
// legacy implementation.
void setVirtualCpuTarget(StringRef cpu);
std::string getVirtualCpuTarget();

/// A device- or host-resident allocation.
class MODULAR_CXX_EXPORT DeviceBuffer final
    : public ReferenceCounted<DeviceBuffer> {
public:
  ~DeviceBuffer();

  void *ptr() const { return ptr_; }
  /// Relinquishes ownership of the underlying allocation to the caller and
  /// returns its pointer; destroying the buffer then frees nothing.
  void *releasePtr();

  /// Wraps a heap-boxed Mojo `DeviceBuffer` surrendered by the Mojo side (the
  /// `take_handle` exchange) in an owning shim `DeviceBuffer`; the box is
  /// released when the last reference drops. The result has no shim context:
  /// it only keeps the storage alive, so `context()` and the operations that
  /// use it must not be called on it.
  static DeviceBufferRef adoptMojoBox(MojoHALBufferBox *mojoBox, void *ptr,
                                      size_t byteSize) {
    return DeviceBufferRef::take(new DeviceBuffer(
        mojoBox, ptr, byteSize,
        /*isHost=*/false, BoxKind::Device, DeviceContextRef()));
  }
  size_t byteSize() const { return byteSize_; }
  bool isHost() const { return isHost_; }

  DeviceContext &context() {
    assert(context_ && "DeviceBuffer from adoptMojoBox has no shim context");
    return *context_;
  }
  const DeviceContext &context() const {
    assert(context_ && "DeviceBuffer from adoptMojoBox has no shim context");
    return *context_;
  }

  /// Returns a view of [byteOffset, byteOffset + byteSize) of this buffer.
  /// The view keeps this buffer alive.
  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset, size_t byteSize);

  /// The caller orders `ctx` after this buffer's pending work.
  void setContext(DeviceContext &ctx);

  /// Copies this buffer's full contents into `dst`, enqueued on `dst`'s
  /// context.
  inline ErrorOrSuccess copyTo(DeviceBuffer &dst) {
    return copyTo(dst, 0, dst.byteSize());
  }
  /// Copies `srcByteSize` bytes starting at `srcByteOffset` of this buffer
  /// into `dst`, enqueued on `dst`'s context.
  ErrorOrSuccess copyTo(DeviceBuffer &dst, size_t srcByteOffset,
                        size_t srcByteSize);

  /// Either do a blocking synchronize or set up events with an external
  /// stream. The HAL backend does not support external streams, so passing
  /// one fails.
  ErrorOrSuccess synchronize(void *external_stream = nullptr);

  /// Replaces the release of the underlying memory with `destructor`: on
  /// destruction the callback runs instead of freeing through the driver.
  void setDestructorFn(llvm::unique_function<void()> destructor) {
    destructorFn_ = std::move(destructor);
  }

private:
  friend class DeviceContext;

  /// Which MojoHAL_*_release entry frees `mojoBox_`.
  enum class BoxKind : uint8_t { Device, Host, Mmap };

  DeviceBuffer(MojoHALBufferBox *mojoBox, void *ptr, size_t byteSize,
               bool isHost, BoxKind kind, DeviceContextRef context,
               DeviceBufferRef parent = DeviceBufferRef())
      : mojoBox_(mojoBox), ptr_(ptr), byteSize_(byteSize), isHost_(isHost),
        kind_(kind), context_(std::move(context)), parent_(std::move(parent)) {}

  MojoHALBufferBox
      *mojoBox_; // Heap-boxed Mojo buffer (DeviceBuffer/HostBuffer/mmap).
  void *ptr_;
  size_t byteSize_;
  bool isHost_;
  BoxKind kind_;
  DeviceContextRef context_;
  DeviceBufferRef parent_; // Keeps a sub-buffer's parent allocation alive.
  /// When set, runs in place of the default free (legacy semantics); the
  /// Mojo box is then released without freeing the wrapped memory.
  llvm::unique_function<void()> destructorFn_;
};

/// RAII guard that makes a context's driver context current on the calling
/// thread and restores the previous one on destruction.
class MODULAR_CXX_EXPORT DeviceContextScope {
public:
  explicit DeviceContextScope(const DeviceContext &ctx);
  ~DeviceContextScope();

  /// Reports whether activating the context succeeded.
  ErrorOrSuccess err() const {
    if (err_.isError())
      return Error(llvm::Twine(err_.getError()));
    return {};
  }

private:
  MojoHALScopeBox
      *mojoScope_; // Heap-boxed Mojo scope; null if activation failed.
  ErrorOrSuccess err_;
};

/// A host-signalable, device-waitable 64-bit slot in device-mapped pinned
/// host memory (allocated by `DeviceContext::allocateCompletionFlag`).
class MODULAR_CXX_EXPORT CompletionFlag final
    : public ReferenceCounted<CompletionFlag> {
public:
  ~CompletionFlag();

  CompletionFlag(const CompletionFlag &) = delete;
  CompletionFlag &operator=(const CompletionFlag &) = delete;

  uint64_t devicePtr() const { return devicePtr_; }

  void reset();
  void signal(uint64_t value);
  uint64_t load() const;

private:
  friend class DeviceContext;
  friend class DeviceStream;
  CompletionFlag(MojoHALCompletionFlagBox *mojoFlag, void *hostPtr,
                 uint64_t devicePtr)
      : mojoFlag_(mojoFlag), hostPtr_(hostPtr), devicePtr_(devicePtr) {}

  MojoHALCompletionFlagBox
      *mojoFlag_; // Heap-boxed Mojo-side flag (owns the pinned slot).
  void *hostPtr_;
  uint64_t devicePtr_;
};

/// An in-order command stream on one device.
class MODULAR_CXX_EXPORT DeviceStream final
    : public ReferenceCounted<DeviceStream> {
public:
  ~DeviceStream();

  /// Returns the device this stream executes on.
  Device &device() const { return *device_.getPointer(); }

  ErrorOrSuccess synchronize();

  /// Creates an event recorded on this stream. Only reachable through
  /// external streams, which the HAL backend does not support; fails.
  ErrorOr<DeviceEventRef> enqueueEvent();

  /// Stalls this stream until `event` has been reached.
  ErrorOrSuccess enqueueWaitForEvent(DeviceEvent &event);

  /// Stalls the stream until the flag's slot holds `value`.
  ErrorOrSuccess enqueueWaitOnHostValue(const CompletionFlag &flag,
                                        uint64_t value) const;

  /// Enqueues a host callback on this stream. Like the non-CUDA legacy
  /// implementation's backends, the HAL backend does not support it.
  ErrorOrSuccess enqueueHostFunc(void (*fn)(void *), void *userData) const {
    return Error("enqueueHostFunc is not supported on this device");
  }

  /// Raw device driver stream handle, to enable passing to PyTorch and
  /// similar APIs. Null if the handle cannot be queried.
  void *ptr() const;

private:
  friend class DeviceContext;
  DeviceStream(MojoHALStreamBox *mojoStream, DeviceRef device)
      : mojoStream_(mojoStream), device_(std::move(device)) {}

  MojoHALStreamBox *mojoStream_; // Heap-boxed Mojo HAL DeviceStream.
  DeviceRef device_;
};

/// A synchronization event.
class MODULAR_CXX_EXPORT DeviceEvent final
    : public ReferenceCounted<DeviceEvent> {
public:
  ~DeviceEvent();

  /// Returns the device this event was created on.
  Device &device() const { return *device_.getPointer(); }

  ErrorOrSuccess synchronize();

  /// Whether the event has completed. Unrecorded events report ready,
  /// matching the legacy backends.
  ErrorOr<bool> isReady() const;

  /// Returns the elapsed time in milliseconds between this event and `end`.
  /// The HAL backend has no event-timing support; fails.
  ErrorOr<float> elapsedTime(const DeviceEvent &end) const;

private:
  friend class DeviceContext;
  friend class DeviceStream;
  DeviceEvent(MojoHALEventBox *mojoEvent, DeviceRef device,
              DeviceContextRef context)
      : mojoEvent_(mojoEvent), device_(std::move(device)),
        context_(std::move(context)) {}

  MojoHALEventBox *mojoEvent_;
  DeviceRef device_;
  DeviceContextRef context_;
};

/// A collection of device graphs generated during a recording session.
///
/// Device graphs are not supported by the HAL backend: `DeviceGraph::capture`
/// fails with an error and no `DeviceGraph` instance is ever created, so the
/// replay entry points below are never reached at runtime. The class exists so
/// graph-aware consumers (MEFModel) compile unchanged; their capture attempts
/// surface the unsupported-feature error.
class MODULAR_CXX_EXPORT DeviceGraphCollection {
public:
  explicit DeviceGraphCollection(
      llvm::SmallVector<AsyncDeviceGraphRef> &&graphs)
      : graphs(std::move(graphs)) {}

  /// Set the outputs for the device graph collection by taking ownership of
  /// the set of provided async references.
  void setOutputs(llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> values) {
    outputs = llvm::map_to_vector(values, [](auto &ref) { return ref.copy(); });
  }

  /// Get the output async values for the graph collection.
  llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> getOutputs() const {
    return outputs;
  }

  /// Replay the graph collection by launching all graphs.
  ErrorOrSuccess replay() const;

  /// Execute eagerly and compare launch trace against captured graphs.
  ErrorOrSuccess
  debugVerifyReplay(llvm::function_ref<ErrorOrSuccess()> execute) const;

  /// The number of graphs in the collection.
  int64_t numGraphs() const { return graphs.size(); }

  /// Get captured device graphs in stream-capture order.
  llvm::ArrayRef<AsyncDeviceGraphRef> getGraphs() const { return graphs; }

private:
  llvm::SmallVector<AsyncDeviceGraphRef> graphs;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> outputs;
};

/// A captured sequence of GPU operations that can be replayed. See the
/// DeviceGraphCollection note: the HAL backend never creates instances.
class MODULAR_CXX_EXPORT DeviceGraph final
    : public ReferenceCounted<DeviceGraph> {
public:
  /// Capture a graph by executing a callback. Always fails on the HAL
  /// backend.
  static ErrorOr<DeviceGraphCollection>
  capture(M::AsyncRT::CPUDevice &cpuDevice,
          llvm::ArrayRef<DeviceStreamRef> streams,
          llvm::function_ref<ErrorOrSuccess()> execute);

  /// Replay the captured operations on the stored capture stream.
  ErrorOrSuccess replay() const;

  /// Take ownership of the graph's input async values so their stable backing
  /// memory lives as long as the graph that references it.
  void setInputs(llvm::SmallVector<DeviceBufferRef> inputs) {
    inputs_ = std::move(inputs);
  }

  /// Take ownership of the graph's output async values so their stable backing
  /// memory lives as long as the graph that references it.
  void setOutputs(llvm::SmallVector<AsyncRT::AnyAsyncValueRef> outputs) {
    outputs_ = std::move(outputs);
  }

  /// Get the graph's input async values, in graph-signature result order.
  llvm::ArrayRef<DeviceBufferRef> getInputs() const { return inputs_; }

  /// Get the graph's output async values, in graph-signature result order.
  llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> getOutputs() const {
    return outputs_;
  }

  /// Mirrors the legacy `DeviceGraph::StableAssignment`; see that declaration.
  struct StableAssignment {
    void **stableAddress;
    void *ptr;
  };

  void setStableAssignments(llvm::SmallVector<StableAssignment> assignments) {
    stableAssignments_ = std::move(assignments);
  }

  llvm::ArrayRef<StableAssignment> getStableAssignments() const {
    return stableAssignments_;
  }

  /// Always fails on the HAL backend, like `replay`.
  ErrorOrSuccess patchStableAddresses() const;

  /// The HAL backend never creates instances, so this is unreachable.
  DeviceContext &context() const;

private:
  llvm::SmallVector<DeviceBufferRef> inputs_;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> outputs_;
  llvm::SmallVector<StableAssignment> stableAssignments_;
};

/// A single stream of execution on one device, backed by a heap-boxed Mojo
/// HAL `DeviceContext`.
class MODULAR_CXX_EXPORT DeviceContext final
    : public ReferenceCounted<DeviceContext> {
public:
  static ErrorOr<DeviceContextRef> create(StringRef api, int id = 0);
  static ErrorOr<DeviceContextRef> getOrCreate(StringRef api, int id = 0);

  /// Access configuration flags from the "device-context" section. Identical
  /// semantics to the legacy implementation.
  static int64_t configAsInt(StringRef key, int64_t defaultValue,
                             bool *isDefault = nullptr);

  /// Always null: the HAL shim has no kernel-load gate.
  KernelLoadGate *kernelLoadGate() { return nullptr; }

  /// Returns the device this context executes on.
  Device &device() { return *device_; }
  const Device &device() const { return *device_; }

  // ===-- Streams / events ---------------------------------------------=== //

  ErrorOr<DeviceStreamRef> stream() const;
  ErrorOr<DeviceStreamRef> createStream(int priority = 0);
  /// Wraps a foreign driver stream. Not supported by the HAL backend; fails.
  ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream);
  /// Reports the pool size: this context's own stream plus the side streams
  /// created via createStream().
  unsigned numStreams() const;
  /// Returns a view of this context whose current stream is `streamId`
  /// (0 = this context's own stream, 1.. = side streams in creation order).
  /// The view shares this context's stream pool; an id outside the pool is
  /// an error.
  ErrorOr<DeviceContextRef> selectStream(unsigned streamId);

  /// Creates an event recorded on this context's stream.
  ErrorOr<DeviceEventRef> enqueueEvent();
  /// Records an existing event on this context's stream.
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event);
  /// Creates an unrecorded event. HAL events are always CPU-visible, which
  /// subsumes `enableTiming`.
  ErrorOr<DeviceEventRef> eventCreate(bool enableTiming);
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event);
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx);
  ErrorOrSuccess enqueueWaitForEvent(DeviceEvent &event) {
    return enqueueWaitFor(event);
  }

  // ===-- Thread binding / interop -------------------------------------=== //

  ErrorOrSuccess bindToCurrentThread() const;
  /// The legacy implementation only clears its internal thread-local
  /// bookkeeping (no driver call); the HAL shim keeps none, so this is a
  /// no-op.
  static void unbindFromCurrentThread() {}
  std::optional<intptr_t> getDLPackStream() const;

  // ===-- Completion flags ---------------------------------------------=== //

  /// Allocates a zero-initialized 64-bit completion flag in device-mapped
  /// pinned host memory.
  ErrorOr<CompletionFlagRef> allocateCompletionFlag();

  // ===-- Error helpers -------------------------------------------------=== //

  /// Formats an error with source context (used by ErrorMacros.h).
  static Error formatError(const char *file, int line, const char *func,
                           const llvm::Twine &msg);

  ~DeviceContext();

  // ===-- Buffers ------------------------------------------------------=== //

  ErrorOr<DeviceBufferRef> enqueueCreateBuffer(size_t byteSize);
  ErrorOr<DeviceBufferRef> createHostBuffer(size_t byteSize);
  DeviceBufferRef createBufferOwning(void *ptr, size_t byteSize, bool owning);
  DeviceBufferRef createHostBufferOwning(void *ptr, size_t byteSize,
                                         bool owning);

  /// Maps `filename` read-only into memory as a host-visible buffer. The
  /// mapping is released when the buffer is destroyed.
  ErrorOr<DeviceBufferRef> memmapFile(const std::filesystem::path &filename);

  // ===-- Copies / memset ----------------------------------------------=== //

  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize);
  inline ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                            const void *src) {
    return enqueueCopyToDevice(dst.ptr(), src, dst.byteSize());
  }

  /// Batched `dsts[i] <- srcs[i]`. The HAL has no batched-submit primitive, so
  /// this loops the scalar path, which resolves direction per pair.
  inline ErrorOrSuccess enqueueBatchCopy(llvm::ArrayRef<DeviceBuffer *> dsts,
                                         llvm::ArrayRef<DeviceBuffer *> srcs) {
    if (dsts.size() != srcs.size())
      RETURN_ERROR("batch copy requires dsts and srcs of equal length");
    for (size_t i = 0, n = dsts.size(); i < n; ++i) {
      if (dsts[i]->byteSize() != srcs[i]->byteSize())
        RETURN_ERROR("batch copy requires matching byte sizes for each pair");
      if (&dsts[i]->context() != this)
        RETURN_ERROR("batch copy requires every destination on this context");
    }
    for (size_t i = 0, n = dsts.size(); i < n; ++i) {
      if (auto err = srcs[i]->copyTo(*dsts[i]))
        return err;
    }
    return success();
  }

  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize);
  inline ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                              const DeviceBuffer &src) {
    return enqueueCopyFromDevice(dst, src.ptr(), src.byteSize());
  }

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t value,
                                  size_t valueSize);
  ErrorOrSuccess enqueueSetMemory(const DeviceBuffer &dst, uint64_t val,
                                  size_t valSize);

  // ===-- Synchronization ----------------------------------------------=== //

  ErrorOrSuccess synchronize(void *external_stream = nullptr);

  // ===-- Device info --------------------------------------------------=== //

  ErrorOr<Device::MemInfo> getMemoryInfo() const;
  ErrorOr<const char *> getArchName() const;
  ErrorOrSuccess isCompatible() const;

  // ===-- Debugging helpers ---------------------------------------------=== //

  /// The debug allocator (MODULAR_DEBUG_DEVICE_ALLOCATOR) is a
  /// legacy-implementation-only diagnostic; the HAL backend reports it
  /// disabled.
  bool isDebugAllocatorEnabled(StringRef name) const { return false; }

  // ===-- Host memory / addressing --------------------------------------=== //

  /// Whether `ptr` is a device address owned by this context. The legacy
  /// default ("residency unknown -- caller should stage") is the conservative
  /// answer; a driver-backed residency query is a future optimization.
  bool ownsDeviceAddress(const void *ptr) const { return false; }

  /// Resolves a device pointer to its host-visible address on unified-memory
  /// devices. The HAL API exposes no such query yet, so this reports the
  /// legacy "unsupported" default.
  ErrorOr<void *> getHostAccessiblePointer(const void *devicePtr) {
    return Error("getHostAccessiblePointer is not supported on this device");
  }

  /// Grants the device access to a host range. The HAL API has no host-wrap
  /// primitive yet, so this reports the legacy "unsupported" default.
  ErrorOr<DeviceBufferRef> wrapHostMemory(void *ptr, size_t byteSize) {
    return Error("wrapHostMemory is not supported on this device");
  }

  /// Runs `handler` on a later poll, once the work enqueued so far completes.
  ErrorOrSuccess enqueueEventHandler(std::function<void()> handler);
  /// Runs the handlers whose work has completed, in enqueue order.
  ErrorOrSuccess pollPendingEvents();

  // ===-- Identity / Mojo interop ---------------------------------------=== //

  /// Contexts compare equal when they drive the same underlying stream,
  /// mirroring the legacy implementation.
  bool operator==(const DeviceContext &other) const;
  bool operator!=(const DeviceContext &other) const {
    return !(*this == other);
  }

  /// The heap-boxed Mojo HAL `DeviceContext` this object wraps. The graph
  /// runtime hands op code a C++ `DeviceContext*`; the Mojo side recovers its
  /// Mojo `DeviceContext` from this box.
  void *mojoHandle() const {
    if (!mojoCtx_)
      reportNoMojoHandle();
    return mojoCtx_;
  }

  /// Adopts a heap-boxed Mojo `DeviceContext` produced by the Mojo
  /// a boxed Mojo `DeviceContext` (the reverse of `mojoHandle()`): the returned
  /// C++ context takes ownership of `mojoBox` and releases it when its
  /// refcount reaches zero. Used by the graph runtime to materialize a C++
  /// `DeviceContext` from the opaque handle a kernel hands back.
  static ErrorOr<DeviceContextRef> adoptMojoBox(void *mojoBox);

private:
  friend class Device;
  friend class DeviceBuffer;
  friend class DeviceContextScope;

  DeviceContext(MojoHALContextBox *mojoCtx, DeviceRef device);

  /// Streams created via createStream(), shared with selectStream() views so
  /// every view resolves the same pool. `base` is the owning context's own
  /// stream, captured at construction (empty on capture failure, in which
  /// case selectStream(0) falls back to the context itself).
  struct StreamPool {
    std::mutex mu;
    DeviceStreamRef base;
    llvm::SmallVector<DeviceStreamRef> side;
  };

  /// View constructor (selectStream): adopts `mojoCtx` (a box whose current
  /// stream is the selected one) and shares the owning context's pool.
  DeviceContext(MojoHALContextBox *mojoCtx, DeviceRef device,
                std::shared_ptr<StreamPool> pool)
      : mojoCtx_(mojoCtx), streamPool_(std::move(pool)),
        device_(std::move(device)) {}

  /// Reports the `mojoHandle()` misuse above and aborts.
  [[noreturn]] static void reportNoMojoHandle();

  MojoHALContextBox *mojoCtx_;
  std::shared_ptr<StreamPool> streamPool_;
  DeviceRef device_;
  mutable std::string archName_;

  std::mutex pendingMu_;
  std::deque<std::pair<DeviceEventRef, std::function<void()>>> pending_;
};

} // namespace Driver
} // namespace M

#endif // MLRT_DRIVER_DEVICECONTEXT_HALDEVICECONTEXT_H
