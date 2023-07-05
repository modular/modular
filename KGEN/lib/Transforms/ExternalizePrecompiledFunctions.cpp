//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
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
  ModuleOp theModule = cast<ModuleOp>(getOperation());
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // For each function in the module, check if it's been externalized. This is
  // signified by having the precompiledBodyRef attribute.
  DenseMap<StringAttr, SymbolRefAttr> calleeToPrecompiledBodyRef;
  for (auto func : llvm::make_early_inc_range(theModule.getOps<FuncOp>())) {
    // Delete the functions we put in the map because we don't need them anymore
    // - we're about to remove them and replace calls to them with
    // pop.external_call.
    if (SymbolRefAttr precompiledRef = func.getPrecompiledBodyRefAttr()) {
      calleeToPrecompiledBodyRef[func.getSymNameAttr()] = precompiledRef;
      symtab.erase(func);
    }
  }

  // Now, we erase the functions that have already been compiled. This will
  // replace all calls to the erased functions with pop.external_call ops
  // that use the kgen.link directive generated for the package.
  auto workFn = [&](FuncOp func) {
    // Walk the calls in the function and replace calls to the thing that was
    // erased with pop.external_call that references the linked binary.
    func.walk([&](CallOp call) {
      StringAttr callee =
          cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getAttr();
      SymbolRefAttr precompiledRef = calleeToPrecompiledBodyRef.lookup(callee);
      if (!precompiledRef)
        return;

      // Replace the call with a pop.external_call, marked as imported from the
      // library we have linked.
      OpBuilder b(call);
      auto externalCall = b.create<POP::ExternalCallOp>(
          call.getLoc(), call.getResultTypes(), callee.getValue(),
          call.getOperands(),
          cast<FlatSymbolRefAttr>(precompiledRef).getAttr().getValue());
      call.replaceAllUsesWith(externalCall);
      call.erase();
    });
  };

  mlir::parallelForEach(theModule.getContext(),
                        theModule.getOps<FuncOp>(), workFn);
}
