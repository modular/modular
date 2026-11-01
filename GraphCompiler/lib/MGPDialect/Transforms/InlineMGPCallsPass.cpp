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

#define DEBUG_TYPE "inline-mgp-calls"

namespace M::MGP {

#define GEN_PASS_DEF_INLINEMGPCALLSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Slot offsetting helper functions
//===----------------------------------------------------------------------===//

/// Offsets the slot index attributes of MGP ops nested in `op`.
void offsetRuntimeSlot(Operation *op, uint64_t slotOffset) {
  op->walk([&](Operation *op) {
    if (auto bufferCacheOp = dyn_cast<BufferCacheOp>(op)) {
      uint64_t oldSlot = bufferCacheOp.getBBufferSlot();
      bufferCacheOp.setBBufferSlot(oldSlot + slotOffset);
    } else if (auto runtimeCreateOp = dyn_cast<RuntimeCreateOp>(op)) {
      uint64_t oldSlot = runtimeCreateOp.getARuntimeSlot();
      runtimeCreateOp.setARuntimeSlot(oldSlot + slotOffset);
    } else if (auto moduleLoadOp = dyn_cast<ModuleLoadOp>(op)) {
      uint64_t oldSlot = moduleLoadOp.getARuntimeSlot();
      moduleLoadOp.setARuntimeSlot(oldSlot + slotOffset);
    } else if (auto executeGenericOp = dyn_cast<ExecuteGenericOp>(op)) {
      uint64_t oldSlot = executeGenericOp.getARuntimeSlot();
      executeGenericOp.setARuntimeSlot(oldSlot + slotOffset);
    }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Call Graph Definition for MGP Models
//===----------------------------------------------------------------------===//

struct MGPCallGraphNode
    : public KGEN::SCCNode<MGPCallGraphNode, ModelOp, CallOp> {
  using SCCNode::SCCNode;

  LogicalResult inlineSubgraph(SCCNode::EdgeT call,
                               uint64_t &modelSlotUsage) const;

  bool visited = false;

  ContextCreateOp contextCreateOp;
  uint64_t slotsUsed = 0;

  /// Nonzero if the node was inlined and may be deleted afterwards.
  uint64_t numTimesInlined = 0;
};

LogicalResult MGPCallGraphNode::inlineSubgraph(MGPCallGraphNode::EdgeT edge,
                                               uint64_t &modelSlotUsage) const {
  // The subgraph's init ops are cloned into the parent's init and its execute
  // ops at the call site, with slots offset past the parent's and placeholder
  // weight names prefixed with the call prefix.
  ModelOp subgraph = edge.node->func;
  ModelOp parentModelOp = this->func;
  CallOp callOp = edge.call;

  if (!subgraph)
    return callOp.emitError()
           << "failed to find symbol " << callOp.getCalleeAttr();

  ContextCreateOp parentContextCreateOp = parentModelOp.getInitContext();

  ContextCreateOp subgraphContextCreateOp = subgraph.getInitContext();

  uint64_t subgraphSlotsUsed = edge.node->slotsUsed;

  uint64_t slotOffset = modelSlotUsage;

  Block *parentInit = parentModelOp.getInitBlock();

  OpBuilder initBuilder(&parentInit->back()); // Before the terminator.
  IRMapping initMap;

  // Assumes one mgp.context.create per model.
  if (parentContextCreateOp && subgraphContextCreateOp)
    initMap.map(subgraphContextCreateOp.getResult(),
                parentContextCreateOp.getResult());

  initMap.map(subgraph.getInitBlock()->getArgument(0),
              parentInit->getArgument(0));

  for (auto subgraphArg : subgraph.getInitBlock()->getArguments())
    if (isa<DeviceContextType>(subgraphArg.getType()))
      for (auto parentArg : parentInit->getArguments())
        if (subgraphArg.getType() == parentArg.getType())
          initMap.map(subgraphArg, parentArg);

  for (auto subgraphArg : subgraph.getInitBlock()->getArguments())
    if (isa<DeviceContextType>(subgraphArg.getType()))
      if (!initMap.contains(subgraphArg))
        return subgraph.emitError()
               << "failed to map device context for argument "
               << subgraphArg.getArgNumber();

  // DCE strips this if nothing is a placeholder.
  StringRef prefix = callOp.getPrefix();
  auto mojoStringType = OpaqueType::get(
      initBuilder.getContext(), initBuilder.getStringAttr("mojo_string"), {});
  Value prefixStringValue = StringCreateOp::create(initBuilder, callOp.getLoc(),
                                                   mojoStringType, prefix);

  for (Operation *subOp : getDataflowOrder(subgraph.getInitBlock())) {
    // The parent already has a context.
    if (isa<ContextCreateOp>(subOp))
      continue;

    // The parent's init terminator waits on the subgraph's init chain.
    if (isa<MEF::OutputOp>(subOp)) {
      Value subgraphOutputChain = initMap.lookup(subOp->getOperand(0));
      assert(subgraphOutputChain &&
             "Subgraph init block is missing an output chain");

      OpOperand &parentOutputChain =
          parentInit->getTerminator()->getOpOperand(0);
      auto newInit = GRT::ChainCreateOp::create(
          initBuilder, parentInit->getTerminator()->getLoc(),
          ValueRange{parentOutputChain.get(), subgraphOutputChain});

      parentOutputChain.set(newInit);
      continue;
    }

    Operation *clonedOp = initBuilder.clone(*subOp, initMap);
    offsetRuntimeSlot(clonedOp, slotOffset);

    if (auto bufConstExtOp = dyn_cast<BufferConstantExternalOp>(clonedOp)) {
      if (bufConstExtOp.getEIsPlaceholder()) {
        Value nameValue = bufConstExtOp.getName();

        // The concat must dominate its use.
        auto priorInsertionPoint = initBuilder.saveInsertionPoint();
        initBuilder.setInsertionPoint(bufConstExtOp);

        Value concatenatedName =
            StringConcatOp::create(initBuilder, callOp.getLoc(), mojoStringType,
                                   prefixStringValue, nameValue);

        initBuilder.restoreInsertionPoint(priorInsertionPoint);

        bufConstExtOp.getNameMutable().assign(concatenatedName);
      }
    }
  }

  Block *subgraphExec = subgraph.getExecuteBlock();
  OpBuilder execBuilder(callOp);
  IRMapping valueMap;
  valueMap.map(subgraphExec->getArguments(), callOp.getOperands());

  for (Operation *subOp : getDataflowOrder(subgraphExec)) {
    if (subOp == subgraphExec->getTerminator())
      continue;

    Operation *clonedOp = execBuilder.clone(*subOp, valueMap);
    offsetRuntimeSlot(clonedOp, slotOffset);
  }

  auto subgraphTerminator = cast<MEF::OutputOp>(subgraphExec->getTerminator());

  auto newResults =
      llvm::map_range(subgraphTerminator.getOperands(),
                      [&](Value v) { return valueMap.lookup(v); });

  callOp.replaceAllUsesWith(newResults);

  modelSlotUsage += subgraphSlotsUsed;
  callOp.erase();
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

  if (node->scc->hasRecursion) {
    auto scc = node->scc;

    ModuleOp module = node->func.getParentOp();

    mlir::InFlightDiagnostic diag =
        module.emitError("Cycle detected in MGP call graph, cannot inline");
    diag << ". Involved models: ";
    llvm::interleaveComma(scc->nodes, diag, [&](MGPCallGraphNode *node) {
      diag << "@" << node->func.getSymName();
      node->func->emitError() << "Model involved in the call graph cycle.";
    });

    ++numErrors;
  }

  {
    node->contextCreateOp = node->func.getInitContext();
    if (!node->contextCreateOp) {
      node->func.emitError("unable to find context");
      ++numErrors;
    } else {
      node->slotsUsed = node->contextCreateOp.getNumSlots();
    }
  }

  node->visited = true;
  return true;
}

void MGPCallGraph::doRewrite(const MGPCallGraphNode *node) {
  if (numErrors > 0)
    return;

  uint64_t modelSlotsUsage = node->slotsUsed;
  for (EdgeT edge : node->callsites) {
    if (failed(node->inlineSubgraph(edge, modelSlotsUsage))) {
      edge.call->emitError("unable to inline model call operation");
      ++numErrors;
      return;
    }

    edge.node->numTimesInlined++;
  }

  ContextCreateOp op = node->contextCreateOp;
  op.setNumSlots(modelSlotsUsage);
}

} // namespace M::MGP

using namespace M;
using namespace M::MGP;

namespace {

//===----------------------------------------------------------------------===//
// InlineMGPCallsPass
//===----------------------------------------------------------------------===//

struct InlineMGPCallsPass
    : public impl::InlineMGPCallsPassBase<InlineMGPCallsPass> {
  using InlineMGPCallsPassBase::InlineMGPCallsPassBase;

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

    // Erase inlined subgraphs that are no longer used.
    for (auto &[model, node] : callGraph.nodes) {
      if (node.numTimesInlined == 0)
        continue;

      if (SymbolTable::symbolKnownUseEmpty(model, moduleOp) &&
          model.getIsSubgraph()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Erasing inlined model @" << model.getSymName() << "\n");
        model.erase();
      } else {
        LLVM_DEBUG(
            llvm::dbgs()
            << "Model @" << model.getSymName()
            << " still has uses or is public after inlining, not erasing.\n");
      }
    }
  }
};

} // namespace
