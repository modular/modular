//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SelectFastestFunction.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/MicroBenchmark.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "select-fastest-function"

using namespace M;
using namespace KGEN;

ErrorOr<size_t>
M::KGEN::evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                                 LLCL::Runtime &runtime,
                                 ArrayRef<FuncOp> specializations) {
  // Create the execution engine.
  UNWRAP_ERROR(engine, ExecutionEngine::create(CompilationOptions()));

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);
  if (auto err = engine.add(runtime, symtab, funcsToCompile,
                            "evaluateSpecializations"))
    return err.takeError();

  // Get pointers to all the candidates.
  SmallVector<void *> candidatePtrs;
  for (FuncOp candidate : specializations) {
    UNWRAP_ERROR(func, engine.lookup("evaluateSpecializations", candidate));
    candidatePtrs.push_back(func.getFunctionPointer());
  }

  // Lookup the evaluator function
  UNWRAP_ERROR(evaluatorFunc,
               engine.lookup("evaluateSpecializations", evaluator));

  // Invoke the evaluator.
  ssize_t bestIdx = evaluatorFunc.invoke<ssize_t, void **, ssize_t>(
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
