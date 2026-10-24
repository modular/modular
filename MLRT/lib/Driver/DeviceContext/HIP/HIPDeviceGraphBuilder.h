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
// This file declares the HIP device graph builder.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICEGRAPHBUILDER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICEGRAPHBUILDER_H

#include "HIPDeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "llvm/ADT/SmallVector.h"

namespace M::Driver {

class HIPDeviceGraphBuilder;

using HIPDeviceGraphBuilderRef = RCRef<HIPDeviceGraphBuilder>;

/// HIP implementation of DeviceGraphBuilder.
///
/// Wraps a hipGraph_t under construction. Use create() to allocate a new
/// empty graph; each addFunctionDirect() call appends one kernel node.
class HIPDeviceGraphBuilder : public DeviceGraphBuilder {
public:
  /// Construct a builder that takes ownership of an already-created \p graph.
  /// Prefer create() to allocate both together.
  HIPDeviceGraphBuilder(HIPDeviceContext &ctx, hipGraph_t graph,
                        DeviceGraphMemoryPoolRef pool = {});
  ~HIPDeviceGraphBuilder() override;

  /// Allocate a new empty hipGraph_t and return a builder for it.
  static ErrorOr<DeviceGraphBuilderRef>
  create(HIPDeviceContext &ctx, DeviceGraphMemoryPoolRef pool = {});

  DeviceContext &context() const override;

  std::optional<DeviceGraphNode> lastNode() const override;

  ErrorOr<DeviceGraphNode>
  addFunctionDirect(const DeviceFunction &func,
                    const DeviceFunctionLaunchParams &params,
                    llvm::ArrayRef<DeviceGraphNode> dependencies) override;

  using DeviceGraphBuilder::addCopyHostToDevice;
  ErrorOr<DeviceGraphNode>
  addCopyHostToDevice(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  using DeviceGraphBuilder::addCopyDeviceToHost;
  ErrorOr<DeviceGraphNode>
  addCopyDeviceToHost(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  using DeviceGraphBuilder::addCopyDeviceToDevice;
  ErrorOr<DeviceGraphNode>
  addCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                        size_t srcByteOffset, size_t srcByteSize,
                        llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  ErrorOr<DeviceGraphNode>
  addSetMemory(void *dst, size_t byteSize, uint64_t val, size_t valSize,
               llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  ErrorOr<DeviceGraphNode>
  addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) override;

  ErrorOr<DeviceGraphRef> instantiate() override;

  hipGraph_t hipGraph() const { return graph_; }

  /// Returns the raw hipGraphNode_t for a previously added node.
  hipGraphNode_t hipNode(DeviceGraphNodeId id) const { return nodes_[id]; }

private:
  /// Resolves a user-provided dependency list into raw hipGraphNode_t
  /// handles. An empty `ArrayRef` produces no dependencies (graph root).
  ErrorOr<llvm::SmallVector<hipGraphNode_t>>
  resolveDeps(llvm::ArrayRef<DeviceGraphNode> deps) const;

  HIPDeviceContextRef ctx_;
  hipGraph_t graph_;
  llvm::SmallVector<hipGraphNode_t> nodes_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICEGRAPHBUILDER_H
