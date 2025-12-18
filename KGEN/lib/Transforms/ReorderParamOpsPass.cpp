//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass reorders parameter declaration operations within each parameter
// scope by moving them to the beginning of the scope. Parameter definitions
// can be scattered throughout a function body and not necessarily follow a
// define-before-use order. This pass ensures that these ops are ordered
// correctly.
//
// The pass also ensures that parameter assertion operations are lifted as early
// as possible within their scopes so that they can be checked before any
// further operations that reference the same parameters are evaluated by the
// elaborator.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Compiler/Threading.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_REORDERPARAMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class ReorderParamOps : public impl::ReorderParamOpsBase<ReorderParamOps> {
public:
  using ReorderParamOpsBase::ReorderParamOpsBase;
  ReorderParamOps(bool disableVerifier)
      : ReorderParamOpsBase(), disableVerifier(disableVerifier) {}
  void runOnOperation() override;

private:
  bool disableVerifier;
};
} // namespace

static LogicalResult processRegion(Region *region,
                                   ParameterUseDefGraph &graph) {
  for (auto &[nestedRegion, nestedGraph] : graph.nestedScopes) {
    if (failed(processRegion(nestedRegion, nestedGraph)))
      return failure();
  }

  llvm::SetVector<Operation *, SmallVector<Operation *, 8>,
                  SmallPtrSet<Operation *, 8>>
      defOps;

  // Map of parameter name to its index in the parameter list.
  llvm::DenseMap<StringAttr, size_t> paramIndex;
  for (auto [i, param] : llvm::enumerate(graph.params)) {
    paramIndex[param] = i;
    auto it = graph.defs.find(param);
    assert(it != graph.defs.end());
    // Ignore the scope parent operation. Input parameters are set contextually.
    if (it->second.defOp == region->getParentOp() ||
        it->second.defOp->getParentRegion() != region)
      continue;
    defOps.insert(it->second.defOp);
  }

  Operation *currOp = &region->getBlocks().front().front();

  for (auto op : llvm::reverse(defOps)) {
    op->moveBefore(currOp);
    currOp = op;
  }

  // Move ParamAssertOps after the definition of the last parameter (determined
  // by paramIndex) they reference. Walk in reverse to ensure that we do not
  // change the relative order of assert ops that reference the same set of
  // parameters.
  for (auto &[op, uses] : llvm::reverse(graph.opUses)) {
    if (!isa<ParamAssertOp>(op))
      continue;

    // Index of the last parameter referenced by the assert.
    // -1 means it does not reference any parameters defined in this scope (i.e.
    // they were all defined in an outer scope), and this op should be moved to
    // the start of this scope.
    ssize_t moveBeforeIndex = -1;
    for (auto use : uses) {
      ssize_t currMoveBeforeIndex = paramIndex.lookup_or(use.getName(), -1);
      if (currMoveBeforeIndex > moveBeforeIndex)
        moveBeforeIndex = currMoveBeforeIndex;
    }

    if (moveBeforeIndex == -1) {
      op->moveBefore(&region->getBlocks().front().front());
    } else {
      Operation *defOp = graph.defs[graph.params[moveBeforeIndex]].defOp;
      if (defOp == region->getParentOp())
        op->moveBefore(&region->getBlocks().front().front());
      else
        op->moveAfter(defOp);
    }
  }

  return success();
}

void ReorderParamOps::runOnOperation() {
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  auto workFunc = [&](auto &cache, GeneratorOp func) {
    ParameterUseDefGraph graph(func.getBodyRegion());
    graph.calculate(cache);
    (void)processRegion(&func.getBodyRegion(), graph);
  };

  std::vector<GeneratorOp> work;
  llvm::append_range(work, getOperation().getOps<GeneratorOp>());
  parallelForEach(&getContext(), work, workFunc, paramCache);

  if (disableVerifier) {
    // This effectively disable the verifier, because MLIR assumes that
    // if the pass said that it preserved all analyses then it can't have
    // permuted the IR. Hence no need to verify.
    markAllAnalysesPreserved();
  }
}

std::unique_ptr<mlir::Pass> KGEN::createReorderParamOps(bool disableVerifier) {
  return std::make_unique<ReorderParamOps>(disableVerifier);
}
