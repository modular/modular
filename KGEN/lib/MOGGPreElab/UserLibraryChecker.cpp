//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "UserLibraryChecker.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <list>

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace {
struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, GeneratorOp, CallOp> {
  using CallGraphNodeBase::CallGraphNodeBase;
};
} // namespace

namespace M::KGEN::MOGGPreElab {
struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  const SymbolTable &symtab;
};
} // namespace M::KGEN::MOGGPreElab

UserLibraryChecker::UserLibraryChecker(ModuleOp module,
                                       const SymbolTable &symtab)
    : cg(new CallGraph(symtab)) {
  cg->build(module, symtab);
  for (auto gen : module.getOps<GeneratorOp>())
    for (auto region : gen.getOps<ParamDeclareRegionOp>())
      paramDeclRegions.push_back(region);
}

UserLibraryChecker::~UserLibraryChecker() = default;

LogicalResult UserLibraryChecker::run() {
  if (failed(checkCallsiteLocation()))
    return failure();

  return success();
}

static LogicalResult checkCallsiteErrorInternal(CallOp call, GeneratorOp root,
                                                GeneratorOp gen) {
  auto checkFunc = [&](const StringLiteral &decorator, StringRef msg,
                       GeneratorOp root) -> LogicalResult {
    if (gen->hasAttr(decorator)) {
      mlir::emitError(call->getLoc(), msg)
          << ", see kernel at " << root->getLoc() << ".";
      return failure();
    }
    return success();
  };

  if (failed(checkFunc(Decorators::TENSOR_ALLOC.attr,
                       "Tensor allocations are currently only supported inside "
                       "the top level kernel",
                       root)))
    return failure();
  if (failed(checkFunc(Decorators::ENABLE_FUSION.attr,
                       "Calling enable_fusion outside of kernel entry point is "
                       "not supported",
                       root)))
    return failure();

  return success();
}

static LogicalResult checkParamRegionCallsiteLocation(
    CallGraph *cg,
    const llvm::SmallVectorImpl<ParamDeclareRegionOp> &paramDeclRegions,
    llvm::SmallPtrSetImpl<CallGraphNode *> &visited) {

  llvm::SmallVector<ParamDeclareRegionOp> registeredRegions;
  LogicalResult res = success();

  for (ParamDeclareRegionOp region : paramDeclRegions) {
    // Check if the region is inside a kenrel.
    if (isKernel(region->getParentOfType<GeneratorOp>()))
      registeredRegions.push_back(region);
  }

  for (ParamDeclareRegionOp region : registeredRegions) {
    // Checks and enqueues calls inside the ParamDeclareRegionOp within a
    // registered kernel.
    for (CallOp call : region.getOps<CallOp>()) {
      auto gen =
          cg->symtab.lookup(call.getCallee().getSymbol().getLeafReference());
      assert(gen && "invalid IR?");
      if (isa<ExternGeneratorOp>(gen))
        continue;
      if (failed(checkCallsiteErrorInternal(
              call, region->getParentOfType<GeneratorOp>(),
              cast<GeneratorOp>(gen))))
        res = failure();
    }
  }

  return res;
}

static LogicalResult checkGeneratorCallsiteLocation(
    CallGraph *cg, llvm::SmallPtrSetImpl<CallGraphNode *> &visited) {

  llvm::SmallVector<GeneratorOp> kernels;
  std::list<CallGraphNode *> queue;
  LogicalResult res = success();

  for (auto &[gen, node] : cg->nodes) {
    // Scan through the decorators to see if the kernel is registered.
    if (isKernel(gen))
      kernels.push_back(gen);
  }

  for (GeneratorOp gen : kernels) {
    // Traverse and check the call graph starting from this kernel.
    CallGraphNode *root = &cg->nodes.find(gen)->second;
    queue.push_back(root);
    while (!queue.empty()) {
      CallGraphNode *caller = queue.front();
      queue.pop_front();
      for (CallGraphNode::EdgeT edge : caller->callsites) {
        CallGraphNode *callee = edge.node;

        if (visited.contains(callee))
          continue;

        if (caller != root) {
          if (failed(checkCallsiteErrorInternal(edge.call, root->func,
                                                callee->func))) {
            res = failure();
            continue;
          }
          visited.insert(callee);
        }

        queue.push_back(callee);
      }
    }
  }

  return res;
}

LogicalResult UserLibraryChecker::checkCallsiteLocation() {
  llvm::SmallPtrSet<CallGraphNode *, 32> visited;
  LogicalResult res = success();
  if (failed(checkGeneratorCallsiteLocation(cg.get(), visited)))
    res = failure();
  if (failed(checkParamRegionCallsiteLocation(cg.get(), paramDeclRegions,
                                              visited)))
    res = failure();

  return res;
}
