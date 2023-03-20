//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SelectFastestFunction.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/MicroBenchmark.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "select-fastest-function"

using namespace M;
using namespace KGEN;

static ErrorOr<Cache::BufferRef>
produceArchiveFromExports(LLCL::Runtime &runtime, SymbolTable &symtab,
                          TargetInfoAttr target, ArrayRef<FuncOp> exports) {
  // Create the set of symbols to export.
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols;
  for (auto e : exports) {
    StringAttr symName = e.getSymNameAttr();
    exportedSymbols.insert({symName, ExportedSymbol(symName)});
  }

  mlir::PassManager mgr(target.getContext());
  auto compilerOr =
      ObjectCompiler::create(runtime, mgr, ".kgen_cache", CompilationOptions());
  if (failed(compilerOr))
    return compilerOr.takeError();
  auto compiler = std::make_unique<ObjectCompiler>(std::move(*compilerOr));

  // Produce a standalone archive for all the exports.
  auto archiveOr = compiler->produceStandaloneArchive(
      symtab, std::move(exportedSymbols), /*isJIT=*/true);
  if (failed(archiveOr))
    return Error("failed to produce standalone archive");

  return archiveOr.takeValue();
}

ErrorOr<size_t>
M::KGEN::evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                                 LLCL::Runtime &runtime, TargetInfoAttr target,
                                 ArrayRef<FuncOp> specializations) {
  // Create the execution engine.
  UNWRAP_ERROR(engine, ExecutionEngine::create(CompilationOptions()));

  // TODO (8082): This should not be necessary.
  std::vector<std::pair<StringLiteral, void *>> compilerRTFunctions;
  KGEN::registerIntelAMX(compilerRTFunctions);
  KGEN::registerLLCL(compilerRTFunctions);
  KGEN::registerMemory(compilerRTFunctions);
  KGEN::registerPrint(compilerRTFunctions);
  KGEN::registerSystem(compilerRTFunctions);
  KGEN::registerTracing(compilerRTFunctions);
  for (auto [name, ptr] : compilerRTFunctions)
    if (auto err = engine.add("evaluateSpecializations", name, ptr))
      return err.takeError();

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);
  SmallVector<void *> candidatePtrs;
  {
    TimeTraceScope<> traceScope("compile-specializations");
    auto archiveOr =
        produceArchiveFromExports(runtime, symtab, target, funcsToCompile);
    if (archiveOr.isError())
      return archiveOr.takeError();

    if (auto err = engine.add("evaluateSpecializations", archiveOr.takeValue()))
      return err.takeError();

    // Get pointers to all the candidates.
    for (FuncOp candidate : specializations) {
      UNWRAP_ERROR(func, engine.lookup("evaluateSpecializations",
                                       candidate.getNameAttr()));
      candidatePtrs.push_back(func.getFunctionPointer());
    }
  }

  // Lookup the evaluator function
  UNWRAP_ERROR(evaluatorFunc, engine.lookup("evaluateSpecializations",
                                            evaluator.getNameAttr()));

  // Invoke the evaluator.
  ssize_t bestIdx;
  {
    TimeTraceScope<> traceScope("execute-specializations");
    bestIdx = evaluatorFunc.invoke<ssize_t, void **, ssize_t>(
        candidatePtrs.data(), candidatePtrs.size());
  }
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
