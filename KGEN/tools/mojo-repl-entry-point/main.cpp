//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines an entry point for a dummy executable used by the Mojo
// REPL. This provides an anchor point for the debugger to run REPL expressions,
// as LLDB requires an in-memory target.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

/// Wrapper function to avoid return type ABI issues when attempting to evaluate
/// specializations. Simply returns the evaluator's result as an out-param.
MODULAR_EXPORT LLVM_ATTRIBUTE_USED void
lldb_evaluate_specializations(ssize_t (*evaluator)(void **, ssize_t),
                              void **specializations,
                              int64_t numSpecializations, uint64_t *best) {
  *best = evaluator(specializations, numSpecializations);
}

/// Entry point that LLDB should stop at before evaluating expressions. It's
/// guaranteed that all required setup happens before this function is called.
MODULAR_EXPORT LLVM_ATTRIBUTE_USED LLVM_ATTRIBUTE_NOINLINE int
mojo_repl_main() {
  return 0;
}

/// Ensure our exported functions aren't DCE'd so we can find it from the REPL.
static void forceLinkExportedSymbols() {
  llvm::nulls() << (void *)&lldb_evaluate_specializations
                << (void *)&mojo_repl_main;
}

//===----------------------------------------------------------------------===//
// CompilerRT
//===----------------------------------------------------------------------===//

/// Forcibly link in the compiler-rt runtime functions. This allows Mojo code
/// running in the repl to use the compiler-rt runtime functions.
static void forceLinkCompilerRT() {
  llvm::nulls() << (void *)&KGEN::registerIntelAMX
                << (void *)&KGEN::registerLLCL << (void *)&KGEN::registerPython
                << (void *)&KGEN::registerMemory
                << (void *)&KGEN::registerRandom
                << (void *)&KGEN::registerSystem
                << (void *)&KGEN::registerTracing;
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

int main() {
  forceLinkExportedSymbols();
  forceLinkCompilerRT();
  KGEN_CompilerRT_Python_SetPythonPath();
  return mojo_repl_main();
}
