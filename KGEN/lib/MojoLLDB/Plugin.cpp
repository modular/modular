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

#include "./Language/MojoLanguage.h"
#include "Commands/CommandObjectLLVMDebug.h"
#include "Commands/CommandObjectMojo.h"
#include "REPL/MojoREPL.h"
#include "Support/CrashReporting.h"
#include "Support/SymbolExport.h"
#include "TypeSystem/MojoTypeSystem.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBDebugger.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace M::KGEN::Mojo;

//===--------------------------------------------------------------===//
// Plugin Initialization
//===--------------------------------------------------------------===//

/// LLDB has two different types of plugin initialization, we support them both
/// here to provide flexibility for users. However, as we have the public API
/// enabled, initialization will go through `lldb::PluginInitialize`.

MODULAR_EXPORT bool LLDBPluginInitialize() {
  // initCrashpadForProgram should only really be used when we "own" the
  // program, and that's not necessarily the case for LLDB... but we have no
  // real better place to put this, since the only better place ('main'
  // function of the LLDB driver) is upstream and hard to patch in our build.
  initCrashpadForProgram("lldb", "mojo-lldb");

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  LLVMLinkInMCJIT();

  // Initialize the various plugin components.
  MojoTypeSystem::Initialize();
  MojoREPL::Initialize();
  MojoLanguage::Initialize();
  return true;
}

MODULAR_EXPORT void LLDBPluginTerminate() {
  MojoREPL::Terminate();
  MojoTypeSystem::Terminate();
  MojoLanguage::Terminate();
}

namespace lldb {
MODULAR_VISIBILITY_EXPORT bool PluginInitialize(SBDebugger debugger) {
  if (!LLDBPluginInitialize())
    return false;

  registerMojoCommands(debugger);
  registerLLVMDebugCommands(debugger);
  return true;
}
} // namespace lldb
