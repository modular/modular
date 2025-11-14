//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/EliminateDeadSymbolUtils.h"
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

  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();

  DenseSet<StringAttr> usedSymbols = KGEN::getUsedSymbols(analysis, theModule);
  // OK, we have all the used symbols. Now, just erase ones that aren't in
  // there.
  unsigned numErased = 0;
  for (auto sym : llvm::make_early_inc_range(
           theModule.getOps<mlir::SymbolOpInterface>())) {
    if (!usedSymbols.contains(sym.getNameAttr())) {
      analysis.getTopLevelSymbolTable().erase(sym);
      ++numErased;
    }
  }
  this->numErased = numErased;
}
