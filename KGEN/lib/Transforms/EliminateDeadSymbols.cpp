//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_ELIMINATEDEADSYMBOLS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct EliminateDeadSymbolsPass
    : M::KGEN::impl::EliminateDeadSymbolsBase<EliminateDeadSymbolsPass> {
  void runOnOperation() override;
};
} // namespace

void EliminateDeadSymbolsPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  if (!theModule.getOps<GeneratorOp>().empty()) {
    mlir::emitError(theModule.getLoc())
        << "cannot run EliminateDeadSymbols before elaboration";
    return signalPassFailure();
  }

  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();

  DenseSet<StringAttr> usedSymbols;
  // The base of the export set is the used symbols.
  theModule.walk([&](ExportOp exportOp) {
    usedSymbols.insert(
        cast<FlatSymbolRefAttr>(exportOp.getExported()).getAttr());
  });

  // Now walk the used symbols and find symbols that they use.
  llvm::SetVector<StringAttr> worklist = {usedSymbols.begin(),
                                          usedSymbols.end()};
  while (!worklist.empty()) {
    StringAttr symbolRef = worklist.pop_back_val();
    auto callee = analysis.getTopLevelSymbolTable().lookup<FuncOp>(symbolRef);
    if (!callee)
      continue;
    // Walk the callee and add any symbol uses to the worklist as long as
    // we haven't already seen them.
    callee.walk([&](Operation *op) {
      op->getAttrDictionary().walkSubAttrs([&](Attribute attr) {
        if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr)) {
          if (usedSymbols.insert(symbolRef.getRootReference()).second)
            worklist.insert(symbolRef.getRootReference());
        }
      });
    });
  }

  // OK, we have all the used symbols. Now, just erase ones that aren't in
  // there.
  for (auto sym :
       llvm::make_early_inc_range(theModule.getOps<mlir::SymbolOpInterface>()))
    if (!usedSymbols.contains(sym.getNameAttr()))
      analysis.getTopLevelSymbolTable().erase(sym);
}
