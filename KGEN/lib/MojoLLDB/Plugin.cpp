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
#include "TypeSystem/MojoTypeSystem.h"
#include "lldb/API/SBDebugger.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace M::KGEN::Mojo;

//===--------------------------------------------------------------===//
// Plugin Initialization
//===--------------------------------------------------------------===//

/// LLDB has two different types of plugin initialization, we support them both
/// here to provide flexibility for users.

MODULAR_EXPORT bool LLDBPluginInitialize() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  LLVMLinkInMCJIT();

  // Initialize the various plugin components.
  MojoTypeSystem::Initialize();
  MojoREPL::Initialize();
  return true;
}

MODULAR_EXPORT void LLDBPluginTerminate() {
  MojoREPL::Terminate();
  MojoTypeSystem::Terminate();
}

namespace lldb {
MODULAR_VISIBILITY_EXPORT bool PluginInitialize(SBDebugger debugger) {
  return LLDBPluginInitialize();
}
} // namespace lldb
