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

#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "Mojo/TransformUtils/CallGraphUtils.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"

#define DEBUG_TYPE "globalize-slot-indices"

namespace M::MGP {

#define GEN_PASS_DEF_GLOBALIZESLOTINDICESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

}; // namespace M::MGP

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Call Graph Definition for MGP Models
//===----------------------------------------------------------------------===//

struct MGPCallGraphNode
    : public KGEN::SCCNode<MGPCallGraphNode, ModelOp, CallOp> {
  using SCCNode::SCCNode;

  LogicalResult createSubgraphOffset(SCCNode::EdgeT call,
                                     uint64_t &modelSlotUsage) const;

  bool visited = false;

  ContextCreateOp contextCreateOp;
  uint64_t slotsUsed = 0;
};

LogicalResult
MGPCallGraphNode::createSubgraphOffset(MGPCallGraphNode::EdgeT edge,
                                       uint64_t &modelSlotUsage) const {
  OpBuilder execBuilder(edge.call);
  auto slotOffsetOp = mlir::index::ConstantOp::create(
      execBuilder, edge.call->getLoc(), modelSlotUsage);
  edge.call->insertOperands(edge.call.getNumOperands(),
                            slotOffsetOp.getResult());
  edge.call.setIsGlobalizedIndex(true);

  modelSlotUsage += edge.node->slotsUsed;

  return success();
}

struct MGPCallGraph : public KGEN::SCCGraph<MGPCallGraph, MGPCallGraphNode,
                                            /*BottomUpRewrite=*/true> {
  using Base = KGEN::SCCGraph<MGPCallGraph, MGPCallGraphNode, true>;
  using Base::Base;

  bool doAnalysis(MGPCallGraphNode *node);
  void doRewrite(const MGPCallGraphNode *node);

  std::atomic<int64_t> numErrors{0};
};

bool MGPCallGraph::doAnalysis(MGPCallGraphNode *node) {
  if (node->visited)
    return false;

  node->contextCreateOp = node->func.getInitContext();
  if (!node->contextCreateOp) {
    node->func.emitError("unable to find context");
    ++numErrors;
  } else {
    node->slotsUsed = node->contextCreateOp.getNumSlots();
  }

  node->visited = true;
  return true;
}

void MGPCallGraph::doRewrite(const MGPCallGraphNode *node) {
  if (numErrors > 0)
    return;

  uint64_t modelSlotsUsage = node->slotsUsed;
  for (EdgeT edge : node->callsites) {
    if (failed(node->createSubgraphOffset(edge, modelSlotsUsage))) {
      edge.call->emitError("unable to inline model call operation");
      ++numErrors;
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// GlobalizeSlotIndicesPass
//===----------------------------------------------------------------------===//

namespace {

struct GlobalizeSlotIndicesPass
    : public impl::GlobalizeSlotIndicesPassBase<GlobalizeSlotIndicesPass> {
  using GlobalizeSlotIndicesPassBase::GlobalizeSlotIndicesPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto models = moduleOp.getOps<ModelOp>();
    if (models.empty() || llvm::hasSingleElement(models))
      return;

    AsyncRT::CPUDevice &cpuDevice =
        *loadContext(&getContext())->get<AsyncRT::CPUDevice>();

    SymbolTable symtab(moduleOp);

    MGPCallGraph callGraph;
    callGraph.build(moduleOp, symtab);
    callGraph.run(cpuDevice);

    if (callGraph.numErrors > 0)
      return signalPassFailure();
  }
};

} // namespace
