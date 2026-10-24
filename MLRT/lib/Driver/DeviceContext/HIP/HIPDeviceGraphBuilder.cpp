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
// This file implements the HIP device graph builder.
//
//===----------------------------------------------------------------------===//

#include "HIPDeviceGraphBuilder.h"
#include "HIPDeviceContext.h"
#include "HIPDriver.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "llvm/ADT/Twine.h"

using namespace M;
using namespace M::Driver;

#define HIP_GRAPH_CHECK_ERR(hip_call)                                          \
  do {                                                                         \
    HIPDriver::Result _result_ = (hip_call);                                   \
    if (_result_ != HIPDriver::SUCCESS)                                        \
      RETURN_ERROR(llvm::Twine("HIP graph builder call failed with error "     \
                               "code ") +                                      \
                   llvm::Twine(static_cast<unsigned>(_result_)));              \
  } while (0)

// ===== HIPDeviceGraphBuilder ===== //

HIPDeviceGraphBuilder::HIPDeviceGraphBuilder(HIPDeviceContext &ctx,
                                             hipGraph_t graph,
                                             DeviceGraphMemoryPoolRef pool)
    : DeviceGraphBuilder(DeviceContextRef::copy(&ctx), std::move(pool)),
      ctx_(HIPDeviceContextRef::copy(&ctx)), graph_(graph) {}

HIPDeviceGraphBuilder::~HIPDeviceGraphBuilder() {
  if (!graph_)
    return;
  DeviceContextScope scope(*ctx_);
  ctx_->hipDriver().hipGraphDestroy(graph_);
  graph_ = nullptr;
}

ErrorOr<DeviceGraphBuilderRef>
HIPDeviceGraphBuilder::create(HIPDeviceContext &ctx,
                              DeviceGraphMemoryPoolRef pool) {
  DeviceContextScope scope(ctx);
  RETURN_IF_ERROR(scope.err());

  hipGraph_t graph = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx.hipDriver().hipGraphCreate(&graph, 0));

  return HIPDeviceGraphBuilderRef::create(ctx, graph, std::move(pool));
}

DeviceContext &HIPDeviceGraphBuilder::context() const { return *ctx_; }

std::optional<DeviceGraphNode> HIPDeviceGraphBuilder::lastNode() const {
  if (nodes_.empty())
    return std::nullopt;
  return DeviceGraphNode{static_cast<DeviceGraphNodeId>(nodes_.size() - 1)};
}

ErrorOr<llvm::SmallVector<hipGraphNode_t>>
HIPDeviceGraphBuilder::resolveDeps(llvm::ArrayRef<DeviceGraphNode> deps) const {
  llvm::SmallVector<hipGraphNode_t> result;
  result.reserve(deps.size());
  for (const DeviceGraphNode &dep : deps) {
    if (dep.id < 0 || static_cast<size_t>(dep.id) >= nodes_.size())
      RETURN_ERROR(llvm::Twine("invalid DeviceGraphNode id ") +
                   llvm::Twine(dep.id));
    result.push_back(nodes_[dep.id]);
  }
  return result;
}

ErrorOr<DeviceGraphNode> HIPDeviceGraphBuilder::addFunctionDirect(
    const DeviceFunction &func, const DeviceFunctionLaunchParams &params,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  const auto &hipFunc = HIPDeviceFunction::cast(func);

  hipKernelNodeParams nodeParams = {
      .blockDim = {params.blockX, params.blockY, params.blockZ},
      .extra = nullptr,
      .func = reinterpret_cast<void *>(hipFunc.hipFunction()),
      .gridDim = {params.gridX, params.gridY, params.gridZ},
      .kernelParams = params.args,
      .sharedMemBytes = params.sharedMemBytes,
  };

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<hipGraphNode_t>, deps,
                         resolveDeps(dependencies));
  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddKernelNode(
      &node, graph_, deps.data(), deps.size(), &nodeParams));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> HIPDeviceGraphBuilder::addCopyHostToDevice(
    void *dst, const void *src, size_t byteSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<hipGraphNode_t>, deps,
                         resolveDeps(dependencies));
  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddMemcpyNode1D(
      &node, graph_, deps.data(), deps.size(), dst, src, byteSize,
      hipMemcpyKind::hipMemcpyHostToDevice));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> HIPDeviceGraphBuilder::addCopyDeviceToHost(
    void *dst, const void *src, size_t byteSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<hipGraphNode_t>, deps,
                         resolveDeps(dependencies));
  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddMemcpyNode1D(
      &node, graph_, deps.data(), deps.size(), dst, src, byteSize,
      hipMemcpyKind::hipMemcpyDeviceToHost));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> HIPDeviceGraphBuilder::addCopyDeviceToDevice(
    DeviceBuffer &dst, DeviceBuffer &src, size_t srcByteOffset,
    size_t srcByteSize, llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  // Graph DtoD nodes only support same-context copies; reject peer copies.
  // Resolve each buffer's context through backendContext() so a buffer wrapped
  // by a decorating context (e.g. the RecordingDeviceContext that routed this
  // copy here) validates against the driver-backed context it decorates.
  if (HIPDeviceContext::verified_cast(&src.context().backendContext()) !=
          ctx_.getPointer() ||
      HIPDeviceContext::verified_cast(&dst.context().backendContext()) !=
          ctx_.getPointer()) {
    RETURN_ERROR("device-to-device graph copy requires src and dst buffers to "
                 "share the builder's HIP context");
  }
  if (srcByteOffset + srcByteSize > src.byteSize()) {
    RETURN_ERROR("source range exceeds source buffer size");
  }
  if (srcByteSize != dst.byteSize()) {
    RETURN_ERROR("source and destination byte sizes must match");
  }

  void *srcPtr = static_cast<uint8_t *>(src.ptr()) + srcByteOffset;

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<hipGraphNode_t>, deps,
                         resolveDeps(dependencies));
  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddMemcpyNode1D(
      &node, graph_, deps.data(), deps.size(), dst.ptr(), srcPtr, srcByteSize,
      hipMemcpyKind::hipMemcpyDeviceToDevice));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> HIPDeviceGraphBuilder::addSetMemory(
    void *dst, size_t byteSize, uint64_t val, size_t valSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  hipMemsetParams params = {};
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

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<hipGraphNode_t>, deps,
                         resolveDeps(dependencies));
  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddMemsetNode(
      &node, graph_, deps.data(), deps.size(), &params));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode>
HIPDeviceGraphBuilder::addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<hipGraphNode_t>, deps,
                         resolveDeps(dependencies));
  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddEmptyNode(
      &node, graph_, deps.data(), deps.size()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphRef> HIPDeviceGraphBuilder::instantiate() {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  hipGraph_t rawGraph = graph_;
  graph_ = nullptr;

  DEFINE_OR_RETURN_ERROR(DeviceStreamRef, stream, ctx_->stream());

  auto graph = HIPDeviceGraphRef::create(DeviceContextRef(ctx_.copy()),
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
