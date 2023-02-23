//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENParameters.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_VERIFYPARAMETERS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct VerifyParametersPass : impl::VerifyParametersBase<VerifyParametersPass> {
  using VerifyParametersBase::VerifyParametersBase;

  void runOnOperation() override {
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    mlir::LockedSymbolTableCollection sharedSymtabs(analysis.getSymbolTables());
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

    std::vector<Region *> declRegions;
    for (auto decl : getOperation().getOps<DeclInterface>())
      for (Region &region : decl->getRegions())
        declRegions.push_back(&region);
    auto workFunc = [&sharedSymtabs, &paramCache](Region *declRegion) {
      ParameterUseDefGraph graph(*declRegion);
      ParameterCollector::Analysis cache = paramCache;
      return graph.verify(sharedSymtabs, cache);
    };
    if (failed(mlir::failableParallelForEach(&getContext(), declRegions,
                                             workFunc)))
      return signalPassFailure();

    // This pass does not modify any IR, so mark all analyses as preserved. In
    // addition, this signals the pass manager that the MLIR verifier need not
    // run after this pass.
    markAllAnalysesPreserved();
  }
};
} // namespace
