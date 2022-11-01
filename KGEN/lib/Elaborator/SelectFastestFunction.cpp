//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SelectFastestFunction.h"

#include "KGEN/ExecutionEngine.h"
#include "Support/MicroBenchmark.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "select-fastest-function"

using namespace M;
using namespace KGEN;

ErrorOr<size_t>
M::KGEN::evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                                 ArrayRef<FuncOp> specializations) {
  // Create the execution engine.
  auto engineOr = ExecutionEngine::create();
  if (failed(engineOr))
    return engineOr.takeError();
  ExecutionEngine engine = std::move(*engineOr);

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);
  if (ErrorOrSuccess err =
          engine.add(cast<ModuleOp>(symtab.getOp()), funcsToCompile))
    return err.takeError();

  // Get pointers to all the candidates.
  SmallVector<void *> candidatePtrs;
  for (FuncOp candidate : specializations) {
    ErrorOr<CompiledFunc> func =
        engine.lookup(candidate.getSymName(), candidate);
    if (func.isError())
      return func.takeError();

    candidatePtrs.push_back(func->getFunctionPointer());
  }

  // Lookup the evaluator function
  ErrorOr<CompiledFunc> evaluatorFunc =
      engine.lookup(evaluator.getSymName(), evaluator);
  if (evaluatorFunc.isError())
    return evaluatorFunc.takeError();

  // Invoke the evaluator.
  ssize_t bestIdx = evaluatorFunc->invoke<ssize_t, void **, ssize_t>(
      candidatePtrs.data(), candidatePtrs.size());
  if (bestIdx == -1)
    return Error("user-provided evaluator returned failure");
  if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= candidatePtrs.size())
    return Error("user-provided evaluator returned an erroneous result");

  LLVM_DEBUG({
    llvm::dbgs() << "Fastest implementation:\n";
    specializations[bestIdx]->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Return the best kernel.
  return bestIdx;
}
