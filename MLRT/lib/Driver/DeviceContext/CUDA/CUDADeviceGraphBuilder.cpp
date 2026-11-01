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
// This file implements the CUDA device graph builder.
//
//===----------------------------------------------------------------------===//

#include "CUDADeviceGraphBuilder.h"
#include "CUDADeviceContext.h"
#include "CUDADriver.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstring>

using namespace M;
using namespace M::Driver;

// Local error-checking macro for CUDA driver results. Intentionally simpler
// than the full formatCUDAError() used in CUDADeviceContext.cpp; the numeric
// code is sufficient for graph-builder call sites.
#define CUDA_GRAPH_CHECK_ERR(cuda_call)                                        \
  do {                                                                         \
    CUDADriver::Result _result_ = (cuda_call);                                 \
    if (_result_ != CUDADriver::SUCCESS)                                       \
      RETURN_ERROR(llvm::Twine("CUDA graph builder call failed with error "    \
                               "code ") +                                      \
                   llvm::Twine(static_cast<unsigned>(_result_)));              \
  } while (0)

CUDADeviceGraphBuilder::CUDADeviceGraphBuilder(CUDADeviceContext &ctx,
                                               CUgraph graph,
                                               DeviceGraphMemoryPoolRef pool)
    : DeviceGraphBuilder(DeviceContextRef::copy(&ctx), std::move(pool)),
      ctx_(CUDADeviceContextRef::copy(&ctx)), graph_(graph) {}

CUDADeviceGraphBuilder::~CUDADeviceGraphBuilder() {
  if (!graph_ && !captureStream_)
    return;
  DeviceContextScope scope(*ctx_);
  // The recording stream is armed for the builder's whole lifetime: end the
  // open capture (discarding any unharvested orphan work) and destroy the
  // builder-owned stream. Best effort — the builder is being torn down. This
  // runs even when the graph was already consumed by instantiate().
  if (captureStream_) {
    CUgraph orphan = nullptr;
    ctx_->cudaDriver().cuStreamEndCapture(rawCaptureStream_, &orphan);
    if (orphan)
      ctx_->cudaDriver().cuGraphDestroy(orphan);
    ctx_->cudaDriver().cuStreamDestroy(rawCaptureStream_);
    rawCaptureStream_ = nullptr;
    // The wrapper is non-owning, so releasing it after the destroy is safe.
    captureStream_ = {};
  }
  if (graph_) {
    CUDADeviceContext::cast(*ctx_).cudaDriver().cuGraphDestroy(graph_);
    graph_ = nullptr;
  }
}

ErrorOr<DeviceGraphBuilderRef>
CUDADeviceGraphBuilder::create(CUDADeviceContext &ctx,
                               DeviceGraphMemoryPoolRef pool) {
  DeviceContextScope scope(ctx);
  RETURN_IF_ERROR(scope.err());

  CUgraph graph = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx.cudaDriver().cuGraphCreate(&graph, 0));

  return CUDADeviceGraphBuilderRef::create(ctx, graph, std::move(pool));
}

DeviceContext &CUDADeviceGraphBuilder::context() const { return *ctx_; }

std::optional<DeviceGraphNode> CUDADeviceGraphBuilder::lastNode() const {
  if (nodes_.empty())
    return std::nullopt;
  return DeviceGraphNode{static_cast<DeviceGraphNodeId>(nodes_.size() - 1)};
}

ErrorOr<llvm::SmallVector<CUgraphNode>> CUDADeviceGraphBuilder::resolveDeps(
    llvm::ArrayRef<DeviceGraphNode> deps) const {
  llvm::SmallVector<CUgraphNode> result;
  result.reserve(deps.size());
  for (const DeviceGraphNode &dep : deps) {
    if (dep.id < 0 || static_cast<size_t>(dep.id) >= nodes_.size())
      RETURN_ERROR(llvm::Twine("invalid DeviceGraphNode id ") +
                   llvm::Twine(dep.id));
    result.push_back(nodes_[dep.id]);
  }
  return result;
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addFunctionDirect(
    const DeviceFunction &func, const DeviceFunctionLaunchParams &params,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_KERNEL_NODE_PARAMS_v2 nodeParams = {
      .func = cudaFunc.cudaFunction(),
      .gridDimX = params.gridX,
      .gridDimY = params.gridY,
      .gridDimZ = params.gridZ,
      .blockDimX = params.blockX,
      .blockDimY = params.blockY,
      .blockDimZ = params.blockZ,
      .sharedMemBytes = params.sharedMemBytes,
      .kernelParams = params.args,
      .extra = nullptr,
  };

  ArrayRef<CUlaunchAttribute_st> attrs(
      static_cast<const CUlaunchAttribute *>(params.attrs), params.numAttrs);

  // A launch that allows programmatic stream serialization gets the edges
  // stream capture would give it: a programmatic edge from each kernel
  // predecessor, so the node may launch once that kernel's blocks trigger
  // their dependents or exit. The kernel's griddepcontrol.wait still waits
  // for the predecessors' completion. Edges from other node types stay full,
  // and the driver may then serialize the node anyway.
  bool programmatic = false;
  for (const CUlaunchAttribute &attr : attrs) {
    if (attr.id == CUlaunchAttributeID::
                       CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
      int allowed = 0;
      std::memcpy(&allowed, attr.value, sizeof(allowed));
      programmatic = allowed != 0;
    }
  }
  programmatic = programmatic &&
                 ctx_->cudaDriver().hasCuGraphAddDependenciesV2() &&
                 llvm::any_of(dependencies, [&](const DeviceGraphNode &dep) {
                   return kernelNodes_.contains(dep.id);
                 });

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  if (!programmatic) {
    CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddKernelNode(
        &node, graph_, deps.data(), deps.size(), &nodeParams));
  } else {
    CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddKernelNode(
        &node, graph_, nullptr, 0, &nodeParams));
    llvm::SmallVector<CUgraphNode> to(deps.size(), node);
    llvm::SmallVector<CUgraphEdgeData> edges;
    edges.reserve(deps.size());
    for (const DeviceGraphNode &dep : dependencies) {
      CUgraphEdgeData edge = {};
      if (kernelNodes_.contains(dep.id)) {
        edge.from_port = CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
        edge.type = CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
      }
      edges.push_back(edge);
    }
    CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddDependencies_v2(
        graph_, deps.data(), to.data(), edges.data(), deps.size()));
  }

  // cuGraphAddKernelNode has no launch-attribute parameter, so attributes are
  // applied to the created node one by one. Only a subset of launch attributes
  // is valid on a kernel node; the rest describe stream-launch scheduling and
  // have no device-graph node equivalent:
  //  - The programmatic-dependent-launch family (PROGRAMMATIC_STREAM_
  //    SERIALIZATION, PROGRAMMATIC_EVENT, LAUNCH_COMPLETION_EVENT) is not a
  //    node attribute. Programmatic stream serialization became the edges
  //    above; the event attributes have no graph form.
  //  - SYNCHRONIZATION_POLICY is a stream-scheduling hint with no node
  //    meaning.
  // An attribute outside the vendored enum is an error, not a silent drop.
  for (const CUlaunchAttribute &attr : attrs) {
    switch (attr.id) {
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_IGNORE:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY:
    case CUlaunchAttributeID::
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT:
      continue;
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
    case CUlaunchAttributeID::
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PRIORITY:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE:
    case CUlaunchAttributeID::
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphKernelNodeSetAttribute(
          node, attr.id, attr.value));
      continue;
    }
    RETURN_ERROR(llvm::Twine("unsupported launch attribute on a device-graph "
                             "kernel node: id ") +
                 llvm::Twine(static_cast<unsigned>(attr.id)));
  }

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  kernelNodes_.insert(id);

  return DeviceGraphNode{id};
}

ErrorOr<DeviceStreamRef> CUDADeviceGraphBuilder::recordingStream() {
  // The stream stays armed for the builder's lifetime, so a stream that
  // already exists is already recording.
  if (captureStream_)
    return captureStream_;

  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  // Builder-owned rather than pooled: the recording stream's lifetime is
  // exactly the builder's (destroyed in the destructor), not the context's.
  unsigned int constexpr kNonBlocking = 0x1;
  CUstream raw = nullptr;
  unsigned int constexpr kDefaultPriority = 0;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuStreamCreateWithPriority(
      &raw, kNonBlocking, kDefaultPriority));

  CUDADriver::Result beginResult = ctx_->cudaDriver().cuStreamBeginCapture(
      raw, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
  if (beginResult != CUDADriver::SUCCESS) {
    ctx_->cudaDriver().cuStreamDestroy(raw);
    RETURN_ERROR(llvm::Twine("cuStreamBeginCapture failed with error code ") +
                 llvm::Twine(static_cast<unsigned>(beginResult)));
  }

  rawCaptureStream_ = raw;
  captureStream_ =
      CUDADeviceStreamRef::create(CUDADeviceContext::cast(*ctx_), raw);
  return captureStream_;
}

ErrorOr<std::optional<DeviceGraphNode>>
CUDADeviceGraphBuilder::finalizeRecordingStream(
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  const std::optional<DeviceGraphNode> none = std::nullopt;
  if (!captureStream_)
    return none;

  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  // The capture's current leaf set is empty until the first enqueue lands on
  // the capture stream. Nothing to harvest then — leave the stream armed (a
  // caller may still be preparing host state before enqueueing, e.g.
  // cuBLAS's workspace setup).
  CUstreamCaptureStatus status =
      CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
  unsigned long long captureId = 0;
  CUgraph capturingGraph = nullptr;
  const CUgraphNode *leaves = nullptr;
  size_t numLeaves = 0;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuStreamGetCaptureInfo_v2(
      rawCaptureStream_, &status, &captureId, &capturingGraph, &leaves,
      &numLeaves));
  if (numLeaves == 0)
    return none;

  CUgraph childGraph = nullptr;
  CUDA_GRAPH_CHECK_ERR(
      ctx_->cudaDriver().cuStreamEndCapture(rawCaptureStream_, &childGraph));
  // Re-arm immediately so handed-out stream objects never observe a
  // non-recording stream, whatever happens to the harvested graph below.
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuStreamBeginCapture(
      rawCaptureStream_, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED));

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDADriver::Result addResult = ctx_->cudaDriver().cuGraphAddChildGraphNode(
      &node, graph_, deps.data(), deps.size(), childGraph);
  // The child is copied into the parent at node-creation time; the original
  // is destroyed regardless of the add's outcome.
  ctx_->cudaDriver().cuGraphDestroy(childGraph);
  if (addResult != CUDADriver::SUCCESS)
    RETURN_ERROR(llvm::Twine("cuGraphAddChildGraphNode failed with error "
                             "code ") +
                 llvm::Twine(static_cast<unsigned>(addResult)));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return std::optional<DeviceGraphNode>(DeviceGraphNode{id});
}

namespace {
// Build a 1D CUDA_MEMCPY3D_v2 covering \p byteSize bytes from \p src to \p dst
// with the given memory-type tags. Caller fills in src/dst pointer fields.
CUDA_MEMCPY3D_v2 make1DMemcpyParams(size_t byteSize) {
  CUDA_MEMCPY3D_v2 params = {};
  params.WidthInBytes = byteSize;
  params.Height = 1;
  params.Depth = 1;
  return params;
}
} // namespace

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addCopyHostToDevice(
    void *dst, const void *src, size_t byteSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUDA_MEMCPY3D_v2 params = make1DMemcpyParams(byteSize);
  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost = src;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = dst;

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemcpyNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addCopyDeviceToHost(
    void *dst, const void *src, size_t byteSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUDA_MEMCPY3D_v2 params = make1DMemcpyParams(byteSize);
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = const_cast<void *>(src);
  params.dstMemoryType = CU_MEMORYTYPE_HOST;
  params.dstHost = dst;

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemcpyNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addCopyDeviceToDevice(
    DeviceBuffer &dst, DeviceBuffer &src, size_t srcByteOffset,
    size_t srcByteSize, llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  // Graph DtoD nodes only support same-context copies; reject peer copies.
  // Resolve each buffer's context through backendContext() so a buffer wrapped
  // by a decorating context (e.g. the RecordingDeviceContext that routed this
  // copy here) validates against the driver-backed context it decorates.
  const auto *srcCtx =
      CUDADeviceContext::verified_cast(&src.context().backendContext());
  const auto *dstCtx =
      CUDADeviceContext::verified_cast(&dst.context().backendContext());
  if (!srcCtx || !dstCtx || srcCtx->cudaContext() != ctx_->cudaContext() ||
      dstCtx->cudaContext() != ctx_->cudaContext()) {
    RETURN_ERROR("device-to-device graph copy requires src and dst buffers to "
                 "share the builder's CUDA context");
  }
  if (srcByteOffset + srcByteSize > src.byteSize()) {
    RETURN_ERROR("source range exceeds source buffer size");
  }
  if (srcByteSize != dst.byteSize()) {
    RETURN_ERROR("source and destination byte sizes must match");
  }

  CUDA_MEMCPY3D_v2 params = make1DMemcpyParams(srcByteSize);
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = static_cast<uint8_t *>(src.ptr()) + srcByteOffset;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = dst.ptr();

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemcpyNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addSetMemory(
    void *dst, size_t byteSize, uint64_t val, size_t valSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUDA_MEMSET_NODE_PARAMS params = {};
  params.dst = dst;
  params.pitch = 0;
  params.height = 1;

  switch (valSize) {
  case 1:
    params.elementSize = 1;
    params.value = static_cast<uint8_t>(val);
    params.width = byteSize;
    break;
  case 2:
    if ((byteSize & 1) != 0)
      RETURN_ERROR("byteSize of buffer is misaligned");
    params.elementSize = 2;
    params.value = static_cast<uint16_t>(val);
    params.width = byteSize >> 1;
    break;
  case 4:
    if ((byteSize & 3) != 0)
      RETURN_ERROR("byteSize of buffer is misaligned");
    params.elementSize = 4;
    params.value = static_cast<uint32_t>(val);
    params.width = byteSize >> 2;
    break;
  case 8: {
    if ((byteSize & 7) != 0)
      RETURN_ERROR("byteSize of buffer is misaligned");
    uint32_t hiBits = static_cast<uint32_t>(val >> 32);
    uint32_t loBits = static_cast<uint32_t>(val & 0xfffffffful);
    if (hiBits != loBits) {
      RETURN_ERROR("8-byte memset with differing high/low halves cannot be "
                   "expressed as a single graph node");
    }
    params.elementSize = 4;
    params.value = loBits;
    params.width = byteSize >> 2;
    break;
  }
  default:
    RETURN_ERROR("unsupported valSize");
  }

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemsetNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode>
CUDADeviceGraphBuilder::addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddEmptyNode(
      &node, graph_, deps.data(), deps.size()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphRef> CUDADeviceGraphBuilder::instantiate() {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  // Fold any trailing captured stream work in before freezing the graph. The
  // insertion-order tail stands in for the recording chain here; in the
  // serial graphs synthesis builds they are the same node.
  llvm::SmallVector<DeviceGraphNode, 1> tailDeps;
  if (std::optional<DeviceGraphNode> last = lastNode())
    tailDeps.push_back(*last);
  ErrorOr<std::optional<DeviceGraphNode>> flushed =
      finalizeRecordingStream(tailDeps);
  if (flushed.isError())
    return flushed.takeError();

  CUgraph rawGraph = graph_;
  graph_ = nullptr;

  DEFINE_OR_RETURN_ERROR(DeviceStreamRef, stream, ctx_->stream());

  auto graph = CUDADeviceGraphRef::create(DeviceContextRef(ctx_.copy()),
                                          rawGraph, std::move(stream));
  RETURN_IF_ERROR(graph->instantiate());
  // Transfer the recorded inputs and outputs to the graph so their backing
  // memory outlives this builder.
  graph->setInputs(takeInputs());
  graph->setOutputs(takeOutputs());
  graph->setStableAssignments(takeStableAssignments());
  graph->setMemoryPool(takeMemoryPool());
  return DeviceGraphRef(std::move(graph));
}
