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
// This file declares the DeviceGraphBuilder interface for explicit device graph
// construction.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H
#define MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "llvm/ADT/ArrayRef.h"

#include <chrono>
#include <optional>

// The graph builder drives legacy DeviceContext internals (DeviceFunction,
// device-graph capture) that the HAL DeviceContext does not expose; under
// --config=hal-device-context the whole builder compiles out.
#if !MODULAR_HAL_DEVICE_CONTEXT

namespace M::Driver {

class DeviceGraphBuilder;

/// Launch configuration for a single kernel-dispatch node.
struct DeviceFunctionLaunchParams {
  /// Grid dims
  uint32_t gridX = 0;
  uint32_t gridY = 0;
  uint32_t gridZ = 0;

  /// Block dims
  uint32_t blockX = 0;
  uint32_t blockY = 0;
  uint32_t blockZ = 0;

  uint32_t sharedMemBytes = 0;
  /// Backend-specific launch attributes (CUlaunchAttribute /
  /// hipLaunchAttribute).
  void *attrs = nullptr;
  uint32_t numAttrs = 0;
  /// Kernel arguments: array of pointers, one per argument.
  void **args = nullptr;
  uint32_t argCount = 0;
};

using DeviceGraphBuilderRef = RCRef<DeviceGraphBuilder>;

/// Opaque integer identifier for a node in an under-construction device graph.
using DeviceGraphNodeId = int32_t;

/// A lightweight handle identifying a node in an under-construction device
/// graph. Returned by DeviceGraphBuilder::addFunctionDirect(); can be passed
/// as a dependency when adding subsequent nodes to express the required
/// execution ordering. The dependency API will be added in a future revision.
struct DeviceGraphNode {
  DeviceGraphNodeId id;
};

/// Builder for explicit device graph construction.
///
/// Callers add individual typed nodes — kernels, memsets, memcopies,
/// allocations — and express dependency edges between them before
/// instantiating the graph.
///
/// Typical usage:
///
/// \code
///   auto builder = ctx->createGraphBuilder();
///   auto nodeA   = builder->addFunctionDirect(kernelA, paramsA);
///   auto nodeB   = builder->addFunctionDirect(kernelB, paramsB);
///   auto graph   = builder->instantiate();
///   graph->replay();
/// \endcode
class DeviceGraphBuilder : public ReferenceCounted<DeviceGraphBuilder> {
public:
  /// \param pool optional shared memory pool to allocate from; when null the
  /// builder creates a private one. Must belong to `context`'s device. See the
  /// `memoryPool` member for the sharing contract.
  DeviceGraphBuilder(DeviceContextRef context,
                     DeviceGraphMemoryPoolRef pool = {});

  virtual ~DeviceGraphBuilder();

  /// Returns the DeviceContext that owns this graph builder.
  virtual DeviceContext &context() const = 0;

  /// Returns the handle of the most recently added node, or `std::nullopt`
  /// if no nodes have been added yet.
  virtual std::optional<DeviceGraphNode> lastNode() const = 0;

  // Each node-creation method below takes an explicit \p dependencies list
  // of DeviceGraphNode handles previously returned by this builder. An empty
  // ArrayRef makes the new node a graph root with no predecessors; a
  // non-empty ArrayRef uses those exact handles as predecessor edges. There
  // is no implicit "chain after the last node" behavior — callers must wire
  // dependencies explicitly.

  /// Add a kernel-dispatch node to the graph.
  ///
  /// \param func   The device function (kernel) to launch.
  /// \param params Grid/block dimensions, shared memory, and kernel arguments.
  /// \param dependencies Explicit dependency list; see above.
  ///
  /// \returns A node handle on success, or an error if the underlying driver
  ///          call fails.
  virtual ErrorOr<DeviceGraphNode>
  addFunctionDirect(const DeviceFunction &func,
                    const DeviceFunctionLaunchParams &params,
                    llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  // Memcpy nodes — argument shapes mirror DeviceContext::enqueueCopy*.

  /// Add a host-to-device memcpy node.
  virtual ErrorOr<DeviceGraphNode>
  addCopyHostToDevice(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  ErrorOr<DeviceGraphNode>
  addCopyHostToDevice(const DeviceBuffer &dst, const void *src,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) {
    if (dst.context().device().api() != context().device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return addCopyHostToDevice(dst.ptr(), src, dst.byteSize(), dependencies);
  }

  /// Add a device-to-host memcpy node.
  virtual ErrorOr<DeviceGraphNode>
  addCopyDeviceToHost(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  ErrorOr<DeviceGraphNode>
  addCopyDeviceToHost(void *dst, const DeviceBuffer &src,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) {
    if (src.context().device().api() != context().device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return addCopyDeviceToHost(dst, src.ptr(), src.byteSize(), dependencies);
  }

  /// Add a device-to-device memcpy node. Both buffers must belong to a context
  /// using the same API as this builder; cross-context copies are not
  /// supported in graphs.
  ErrorOr<DeviceGraphNode>
  addCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                        llvm::ArrayRef<DeviceGraphNode> dependencies) {
    return addCopyDeviceToDevice(dst, src, 0, src.byteSize(), dependencies);
  }

  virtual ErrorOr<DeviceGraphNode>
  addCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                        size_t srcByteOffset, size_t srcByteSize,
                        llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  // Memset nodes — argument shapes mirror DeviceContext::enqueueSetMemory.

  /// Add a memset node. \p valSize must be 1, 2, 4, or 8 bytes; for 8-byte
  /// patterns the high and low halves must match (the streaming form of
  /// `enqueueSetMemory` issues two ops in that case, which cannot be
  /// represented as a single graph node).
  virtual ErrorOr<DeviceGraphNode>
  addSetMemory(void *dst, size_t byteSize, uint64_t val, size_t valSize,
               llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  ErrorOr<DeviceGraphNode>
  addSetMemory(const DeviceBuffer &dst, uint64_t val, size_t valSize,
               llvm::ArrayRef<DeviceGraphNode> dependencies) {
    if (dst.context().device().api() != context().device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return addSetMemory(dst.ptr(), dst.byteSize(), val, valSize, dependencies);
  }

  /// Add an empty (no-op) node to the graph. Empty nodes do no work at
  /// execution time and exist purely for transitive ordering: a single empty
  /// node fanned in from m predecessors and out to n successors expresses an
  /// m-to-n barrier with m+n edges instead of m*n edges, and can serve as a
  /// stable handle for "the completion of this phase" when the producer set
  /// is not visible to the consumer.
  ///
  /// \param dependencies Explicit dependency list; see above.
  virtual ErrorOr<DeviceGraphNode>
  addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  /// Instantiate the currently constructed device graph into its executable
  /// format. Can be subsequently replayed multiple times.
  virtual ErrorOr<DeviceGraphRef> instantiate() = 0;

  /// Returns a stream whose enqueued work is captured into the graph under
  /// construction rather than executed. This is what lets opaque libraries
  /// that enqueue on a raw stream (vendor BLAS) record into a device graph.
  ///
  /// The stream is in record mode for the builder's entire lifetime: it is
  /// created armed, `finalizeRecordingStream` harvests-and-re-arms it, and it
  /// is torn down only with the builder. Handed-out stream objects therefore
  /// never observe a non-recording stream; repeated calls return the same
  /// stream.
  ///
  /// Backends without capture support return an error; callers are expected
  /// to fall back (vendor BLAS call sites fall back to Mojo kernels).
  virtual ErrorOr<DeviceStreamRef> recordingStream();

  /// Harvests the work captured on the recording stream since the last
  /// harvest: materializes it as a single child-graph node ordered after
  /// \p dependencies, immediately re-arms the stream, and returns the node's
  /// handle. Returns nullopt — with the stream left armed — when no recording
  /// stream exists or nothing has been captured yet (a caller may still be
  /// preparing host state before enqueueing, e.g. cuBLAS's workspace setup,
  /// so an empty capture is never torn down out from under it).
  virtual ErrorOr<std::optional<DeviceGraphNode>>
  finalizeRecordingStream(llvm::ArrayRef<DeviceGraphNode> dependencies);

  /// Returns a `DeviceContext` facade that records operations enqueued through
  /// it into this builder instead of executing them eagerly. This lets code
  /// written against `DeviceContext` (the bulk of the stdlib and kernels)
  /// record into a device graph without modification. See
  /// `RecordingDeviceContext`.
  ///
  /// \param initialDep the node that recorded operations chain after. This
  /// builder does not model dependencies itself (that lives in the Mojo layer),
  /// so the caller passes a node — typically an `addEmpty` node seeded with the
  /// ambient/region predecessors — as the recording chain's root.
  ErrorOr<DeviceContextRef> recordingContext(DeviceGraphNode initialDep);

  /// Returns a newly allocated buffer, backed by pinned host memory when
  /// `isHost`. Memory allocated through `DeviceGraphBuilder` has a lifetime
  /// tied to the resulting `DeviceGraph`.
  ///
  /// Allocations are not recorded in the device graph in the current
  /// implementation. Rather, allocations are stored into a separate memory pool
  /// via DeviceGraphMemoryPool.
  ErrorOr<DeviceBufferRef> createBuffer(size_t numBytes, bool isHost);

  /// Add an output to the device graph.
  void addOutput(AsyncRT::AnyAsyncValueRef output);

  /// Return the number of outputs registered on the device graph.
  size_t numOutputs() const { return outputs.size(); }

  /// Registers a buffer as the stable location of one of the graph's inputs.
  ///
  /// The graph records this address rather than the caller's, so replay can be
  /// fed by copying the live input into it (see `mgp.device_graph.execute`).
  /// The buffer must come from `createBuffer` so its address stays reserved for
  /// the graph's lifetime, and the reference is retained: releasing it before
  /// `instantiate()` would return the block to the pool, where a later
  /// `createBuffer` could hand the same address out as a second input.
  ///
  /// Inputs are registered in graph-signature order.
  void addInput(DeviceBufferRef input);

  /// Registers an in-place input at the current position: the graph records
  /// the caller's live address directly and replay must NOT copy into it.
  ///
  /// Used for mutable inputs (e.g. a KV cache) the graph writes in place —
  /// a private stable copy would discard those writes. Such inputs contribute
  /// their address to the graph cache key instead, so a moved buffer forces a
  /// rebuild rather than replaying stale addresses. Stored as a null entry in
  /// `inputs` so the positional pairing with the execute op's operands is
  /// preserved.
  void addInPlaceInput() { inputs.push_back(DeviceBufferRef()); }

  /// Return the number of stable inputs registered on the device graph,
  /// including in-place (null) entries.
  size_t numInputs() const { return inputs.size(); }

  /// Records that the graph publishes `ptr` into the stable address slot at
  /// `stableAddress`. Both are device addresses. `ptr` must be graph stable (an
  /// `addInput` twin or a `createBuffer` allocation): the value is fixed once
  /// recorded, and the graph writes it into the slot before each replay via
  /// `DeviceGraph::patchStableAddresses`.
  void addStableAssignment(void **stableAddress, void *ptr);

  size_t numStableAssignments() const { return stableAssignments.size(); }

protected:
  /// Hands the recorded outputs to a subclass `instantiate()` so it can
  /// transfer their ownership to the instantiated `DeviceGraph`, whose stable
  /// backing memory must outlive the graph. Consumes the builder's outputs.
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> takeOutputs() {
    return std::move(outputs);
  }

  /// Counterpart of `takeOutputs` for the graph's stable inputs.
  llvm::SmallVector<DeviceBufferRef> takeInputs() { return std::move(inputs); }

  /// Counterpart of `takeOutputs` for the graph's slot assignments.
  llvm::SmallVector<DeviceGraph::StableAssignment> takeStableAssignments() {
    return std::move(stableAssignments);
  }

  DeviceGraphMemoryPoolRef takeMemoryPool() { return std::move(memoryPool); }

private:
  /// A device graph can produce zero or more outputs whose backing memory needs
  /// to be tracked so that it can be referenced when executing the device
  /// graph.
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> outputs;

  /// The graph's stable input locations, in graph-signature order. Held by
  /// reference so the pool cannot reissue these addresses mid-recording. A
  /// null entry marks an in-place input: the graph records the caller's live
  /// address and replay must not copy into that position.
  llvm::SmallVector<DeviceBufferRef> inputs;

  /// The slots the graph publishes into and the values it publishes, in
  /// `addStableAssignment` order. Handed to the graph by `instantiate()`.
  llvm::SmallVector<DeviceGraph::StableAssignment> stableAssignments;

  /// Memory pool for all allocations created during the recording process.
  /// Because memory addresses are recorded as part of a device graph, all
  /// memory used by the device graph must be reserved for the lifetime of the
  /// device graph. Null once `instantiate()` has handed it to the graph.
  ///
  /// The pool may be shared across builders (passed in at construction) so
  /// their graphs share activation memory: a block freed during builder A's
  /// recording -- or after A instantiates -- can be reissued to builder B, so
  /// graph A's replay writes memory graph B also uses. This is sound only
  /// while graphs sharing a pool replay serially in recording order and do
  /// not rely on transient allocations persisting across replays (the same
  /// assumption `RecordingDeviceContext::createHostBuffer` documents). Stable
  /// inputs and outputs are unaffected: the graph holds references to them,
  /// so their blocks never return to the pool while the graph lives.
  DeviceGraphMemoryPoolRef memoryPool;
};

/// A `DeviceContext` facade that records enqueued operations into a
/// `DeviceGraphBuilder` instead of executing them eagerly.
///
/// Returned by `DeviceGraphBuilder::recordingContext()`. The point is to let
/// the large body of code written against `DeviceContext` (the bulk of the
/// stdlib and kernels) record into a device graph unmodified, rather than
/// trait-ifying every kernel entry point.
///
/// Operations split three ways:
///   - Write / launch / order-only enqueues (`enqueueFunctionExecDirect`,
///     `enqueueCopyToDevice`, `enqueueCopyFromDevice`, `enqueueSetMemory`,
///     `memCopyDtoD`) become graph nodes. Each new node is chained after the
///     previous one (see `lastNode_`) to emulate the stream's in-order
///     execution.
///   - Setup / query calls (`memAlloc`, `memFree`, `loadFunction`,
///     `getAttribute`, `getDeviceMemoryInfo`, `stream`, ...) forward to the
///     builder's real backing context, so recorded nodes reference real
///     handles/addresses.
///   - Host-visible waits (`synchronize`, events, timers) hard-error, because
///     they cannot be satisfied at record time; a kernel that reads a value
///     back and branches on it must be hoisted out of the recorded region.
///
/// Assumes single-threaded recording (true for one forward pass).
class RecordingDeviceContext : public DeviceContext {
public:
  /// \param initialDep the node that the first recorded operation depends on;
  /// subsequent operations chain after each other. See
  /// `DeviceGraphBuilder::recordingContext`.
  RecordingDeviceContext(DeviceGraphBuilder &builder,
                         DeviceGraphNode initialDep);
  ~RecordingDeviceContext() override;

  /// Reseeds the recording chain root: the next recorded operation depends on
  /// `node` instead of the last recorded one. This is what lets the Mojo
  /// builder memoize a single recording context and reuse it across region
  /// scopes, whose nodes must chain after their own region's predecessors
  /// rather than after another region's last node.
  void reseed(DeviceGraphNode node) { lastNode_ = node; }

  /// Returns `ctx` as a recording context, or nullptr when it is not one.
  static RecordingDeviceContext *verified_cast(DeviceContext *ctx) {
    return DeviceContext::cast<RecordingDeviceContext>(ctx);
  }

  // ----- Buffer creation: owned by the underlying context ----- //

  // A buffer created through the recording view must belong to the backing
  // context, not to the view: the view cannot outlive the recorded region,
  // its `synchronize()` hard-errors (so host readback via the buffer would
  // fail), and the graph replays on the backing context's stream. Delegating
  // creation tags the buffer with the backing context.
  ErrorOr<DeviceBufferRef> enqueueCreateBuffer(size_t byteSize) override;
  ErrorOr<DeviceBufferRef> createHostBuffer(size_t byteSize) override;

  // ----- Recorded (write / launch / order-only) ----- //

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount, const size_t *argSizes) override;
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override;

  // ----- Forwarded to the backing context ----- //

  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes, StringRef debugLevel,
               int32_t optimizationLevel) override;
  ErrorOr<DeviceFunctionRef>
  loadFunctionWithDebugName(StringRef moduleName, StringRef functionName,
                            StringRef code, int32_t maxDynamicSharedBytes,
                            StringRef debugLevel, int32_t optimizationLevel,
                            StringRef debugName) override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  // Accessors to get at the underlying device context. This accessor allows
  // operations to get to the true device context backing the
  // RecordingDeviceContext.
  DeviceContext &backendContext() override;
  const DeviceContext &backendContext() const override;

  // Returns a stream which records operations into the recording device graph.
  //
  // This is not implemented for all backends; the operation forwards to
  // DeviceGraphBuilder::recordingStream, which produces a backend-specific
  // stream which records into a device graph. For CUDA and HIP, these are new
  // streams in record mode.
  //
  // This API allows the device graph infrastructure to work with vendor
  // APIs that do not support device graphs (e.g. cuBLAS), where record/replay
  // is the documented way to use the vendor library with device graphs.
  ErrorOr<DeviceStreamRef> stream() const override;

  // ----- Hard-error (non-recordable host-visible waits) ----- //

  ErrorOrSuccess synchronize() override;
  ErrorOr<DeviceEventRef> enqueueEvent() override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) override;
  ErrorOr<DeviceTimerRef> startTimer() override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) override;

protected:
  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) override;
  ErrorOrSuccess memWrap(void **devPtr, void *hostPtr,
                         size_t byteSize) override;
  ErrorOrSuccess memFree(void *ptr, Device::FreeKind kind) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;
  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
  void *driverContext() const override;

private:
  /// The current predecessor set for a new node: the last recorded node (or
  /// empty for the first), which is what serializes operations in enqueue
  /// order.
  llvm::SmallVector<DeviceGraphNode, 1> currentDeps() const;
  /// Advances `lastNode_` to `node` on success, or propagates its error.
  ErrorOrSuccess chain(ErrorOr<DeviceGraphNode> node);
  /// Folds any pending captured stream work (see
  /// `DeviceGraphBuilder::recordingStream`) into the graph and threads the
  /// resulting child node into the recording chain, so the next recorded
  /// operation depends on it. Called before every recorded operation; a no-op
  /// when nothing has been captured.
  ErrorOrSuccess flushCapture();

  DeviceGraphBuilderRef builder_;
  DeviceGraphNode lastNode_;
};

} // namespace M::Driver

#endif // MODULAR_HAL_DEVICE_CONTEXT

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H
