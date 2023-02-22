//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENParameters.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
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
    auto &symtab = getAnalysis<mlir::SymbolTableAnalysis>();
    auto &cache = getAnalysis<ParameterCollector::Analysis>();
    for (auto decl : getOperation().getOps<DeclInterface>()) {
      for (Region &region : decl->getRegions()) {
        ParameterUseDefGraph graph(region);
        if (failed(graph.verify(symtab.getSymbolTables(), cache)))
          return signalPassFailure();
      }
    }
    // This pass does not modify any IR, so mark all analyses as preserved. In
    // addition, this signals the pass manager that the MLIR verifier need not
    // run after this pass.
    markAllAnalysesPreserved();
  }
};
} // namespace
