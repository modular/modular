//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Threading.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_EXTERNALIZEPRECOMPILEDFUNCTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ExternalizePrecompiledFunctionsPass
    : M::KGEN::impl::ExternalizePrecompiledFunctionsBase<
          ExternalizePrecompiledFunctionsPass> {
  void runOnOperation() override;
};
} // namespace

void ExternalizePrecompiledFunctionsPass::runOnOperation() {
  auto theModule = cast<ModuleOp>(getOperation());
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  OpBuilder b(theModule.getContext());
  for (auto func : llvm::make_early_inc_range(theModule.getOps<FuncOp>())) {
    // No precompiled body ref, move along.
    if (!func.getPrecompiledBodyRef().has_value())
      continue;

    // Replace it with a kgen.extern.func.
    b.setInsertionPoint(func);
    auto externFunc = b.create<ExternFuncOp>(func.getLoc(), func.getNameAttr(),
                                             func.getSignature().getValues(),
                                             func.getPrecompiledBodyRefAttr());
    symtab.remove(func);
    symtab.insert(externFunc);
    func->erase();
  }
}
