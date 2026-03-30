//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/EliminateDeadSymbolUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

using namespace M;
using namespace KGEN;

DenseSet<StringAttr> KGEN::getUsedSymbols(mlir::SymbolTableAnalysis &analysis,
                                          ModuleOp theModule) {
  DenseSet<StringAttr> usedSymbols;
  for (auto symbol : theModule.getOps<ExportInterface>())
    if (symbol.isExported())
      usedSymbols.insert(symbol.getSymNameAttr());

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
      // Also walk the location: FusedLoc<DISubprogramAttr> can embed symbol
      // references in DISubroutineType argument/result types that would
      // otherwise be missed.
      walker.walk(op->getLoc());
      for (Type type : op->getResultTypes())
        walker.walk(type);
      for (Region &region : op->getRegions())
        for (Type type : region.getArgumentTypes())
          walker.walk(type);
    });
  }
  return usedSymbols;
}
