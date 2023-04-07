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

#include "KGEN/CompilerRT.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// CompilerRT
//===----------------------------------------------------------------------===//

/// Forcibly link in the compiler-rt runtime functions. This allows Mojo code
/// running in the repl to use the compiler-rt runtime functions.
static void forceLinkCompilerRT() {
  llvm::nulls() << (void *)&KGEN::registerIntelAMX
                << (void *)&KGEN::registerLLCL << (void *)&KGEN::registerMemory
                << (void *)&KGEN::registerPrint << (void *)&KGEN::registerRandom
                << (void *)&KGEN::registerSystem
                << (void *)&KGEN::registerTracing;
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

int main() {
  forceLinkCompilerRT();
  return 0;
}
