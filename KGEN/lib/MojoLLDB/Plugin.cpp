//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the main plugin entry point for the various Mojo LLDB
// extensions.
//
//===----------------------------------------------------------------------===//

#include "REPL/MojoREPL.h"
#include "Support/SymbolExport.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace M::KGEN::Mojo;

//===--------------------------------------------------------------===//
// Plugin Initialization
//===--------------------------------------------------------------===//

MODULAR_EXPORT bool LLDBPluginInitialize() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Initialize the various plugin components.
  MojoREPL::Initialize();
  return true;
}

MODULAR_EXPORT void LLDBPluginTerminate() { MojoREPL::Terminate(); }
