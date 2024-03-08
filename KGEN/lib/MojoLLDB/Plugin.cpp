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
#include "LLCL/Runtime/Runtime.h"
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

/// Get the global LLCL runtime to be used inside the plugin.
static LLCL::Runtime &getOrCreateGlobalRuntime() {
  static ConditionallyOwnedPointer<LLCL::Runtime> runtime =
      LLCL::createRuntimeIfNeeded(
          LLCL::RuntimeOptions().withMainWillNotDonate());
  return *runtime;
}

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

  // We need to create a global runtime for the bits to work with. This is a
  // bit strange, but there's no better place for it.
  getOrCreateGlobalRuntime();

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

static void enableJITDebugging(lldb::SBDebugger &debugger) {
  // FIXME(21178): Implement a smarter JIT loader plugin.
  // JIT debugging works via the JITLoaderGDB LLDB plugin: whenever a module
  // is loaded, the plugin will look for some specific symbols in the symbol
  // table of the module, which causes some computation to be done. Fortunately
  // this doesn't trigger debug info lookups, but it still might cause some
  // unwanted performance degradation when doing remote debugging and symbol
  // tables are not available locally, or when there are individual modules of
  // tens of GB in size. Two ideas of how to diminish the slowdown when the
  // time comes:
  //  - Add a special section in the module in question so that JITLoaderGDB
  //    filters out modules without this section. This will reduce the amount of
  //    unneeded lookups.
  //  - Add a regex feature so that JITLoaderGDB only does the lookup in modules
  //    whose name matches the regex.
  lldb::SBExecutionContext exeCtx;
  lldb::SBCommandReturnObject result;
  debugger.GetCommandInterpreter().HandleCommand(
      "settings set plugin.jit-loader.gdb.enable on", exeCtx, result);
  if (result.GetStatus() == lldb::eReturnStatusFailed) {
    llvm::errs() << "error: " << result.GetError()
                 << "\nDebugging of JITted programs might not work.";
  }
}

namespace lldb {
MODULAR_VISIBILITY_EXPORT bool PluginInitialize(SBDebugger debugger) {
  if (!LLDBPluginInitialize())
    return false;

  registerMojoCommands(debugger, getOrCreateGlobalRuntime());
  registerLLVMDebugCommands(debugger);
  // We enable JIT debugging here so that this feature doesn't depend on
  // lldb init files or how LLDB was launched.
  enableJITDebugging(debugger);
  return true;
}
} // namespace lldb

// FIXME: This is a workaround for LLDB's plugin detection mechanism, which
// currently hardcodes the unix mangling of the function name.
#if defined(_WIN32)
MODULAR_EXPORT bool
_ZN4lldb16PluginInitializeENS_10SBDebuggerE(lldb::SBDebugger debugger) {
  return lldb::PluginInitialize(debugger);
}
#endif
