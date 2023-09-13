//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
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

  DenseSet<StringAttr> usedSymbols;
  for (auto symbol : theModule.getOps<ExportInterface>())
    if (symbol.isExported())
      usedSymbols.insert(symbol.getLinkageNameAttr());

  // Now walk the used symbols and find symbols that they use.
  llvm::SetVector<StringAttr> worklist = {usedSymbols.begin(),
                                          usedSymbols.end()};
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](FlatSymbolRefAttr ref) {
    if (usedSymbols.insert(ref.getAttr()).second)
      worklist.insert(ref.getAttr());
  });
  while (!worklist.empty()) {
    StringAttr symbolRef = worklist.pop_back_val();
    Operation *callee = analysis.getTopLevelSymbolTable().lookup(symbolRef);
    if (!callee)
      continue;
    // Walk the callee and add any symbol uses to the worklist as long as
    // we haven't already seen them.
    callee->walk([&](Operation *op) {
      walker.walk(op->getAttrDictionary());
      for (Type type : op->getResultTypes())
        walker.walk(type);
      for (Region &region : op->getRegions())
        for (Type type : region.getArgumentTypes())
          walker.walk(type);
    });
  }

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
