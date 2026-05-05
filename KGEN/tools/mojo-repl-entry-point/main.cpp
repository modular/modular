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

#include "Init/Init.h"
#include "KGEN/CompilerRT/Registration.h"
#include "Support/Context.h"
#include "llvm/Support/raw_ostream.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

using namespace M;

/// Entry point that LLDB should stop at before evaluating expressions. It's
/// guaranteed that all required setup happens before this function is called.
MODULAR_EXPORT LLVM_ATTRIBUTE_USED LLVM_ATTRIBUTE_NOINLINE int
mojo_repl_main() {
  return 0;
}

/// Ensure our exported functions aren't DCE'd so we can find it from the REPL.
static void forceLinkExportedSymbols() {
  llvm::nulls() << (void *)&mojo_repl_main;
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

#if defined(_WIN32)
int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine,
            int nShowCmd) {
#else
int main() {
#endif
  forceLinkExportedSymbols();
  KGEN_CompilerRT_Python_SetPythonPath();

  // Create our context for execution.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo-repl",
      Init::Options().withCPUDeviceOptions(MLRT::CPUDeviceOptions()));
  if (ctxOr.isError()) {
    llvm::errs() << "unable to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  ContextRef ctx = std::move(*ctxOr);
  MLRT::CPUDevice *cpuDevice = ctx->get<MLRT::CPUDevice>();

  // In order to ensure that mojo has a cpuDevice for execution, inject
  // a global value. Normally this would be set by Mojo during startup,
  // but given that we are skipping that dance we can set it here.
  KGEN_CompilerRT_InsertGlobal("Runtime", static_cast<void *>(cpuDevice));

  return mojo_repl_main();
}
