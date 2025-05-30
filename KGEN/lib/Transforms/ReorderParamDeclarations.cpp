//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// It is common for functions to end up with a lot of parameters that are unused
// in the function body.  This can happen when they are defined on a highly
// parameterized struct, for example, because the function will get all of the
// parameters from that struct.
//
// This pass scans the IR and removes these unused parameters (and also function
// arguments) to reduce burden on the elaborator.  This reduces the amount of
// function clones produced, which reduces compile time and code size.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
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
#define GEN_PASS_DEF_REORDERPARAMDECLARATIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class ReorderParamDeclarations
    : public impl::ReorderParamDeclarationsBase<ReorderParamDeclarations> {
public:
  using ReorderParamDeclarationsBase::ReorderParamDeclarationsBase;
  ReorderParamDeclarations(bool disableVerifier)
      : ReorderParamDeclarationsBase(), disableVerifier(disableVerifier) {}
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

  for (StringAttr param : graph.params) {
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

  return success();
}

void ReorderParamDeclarations::runOnOperation() {
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

std::unique_ptr<mlir::Pass>
KGEN::createReorderParamDeclarations(bool disableVerifier) {
  return std::make_unique<ReorderParamDeclarations>(disableVerifier);
}
