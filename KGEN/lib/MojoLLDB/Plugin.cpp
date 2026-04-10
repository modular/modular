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

#include "KGEN/MojoLLDB/Plugin.h"
#include "Commands/CommandObjectLLVMDebug.h"
#include "Commands/CommandObjectMojo.h"
#include "Init/Init.h"
#include "Language/MojoLanguage.h"
#include "Language/MojoLanguageRuntime.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "REPL/MojoREPL.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/SymbolExport.h"
#include "TypeSystem/MojoTypeSystem.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBDebugger.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>

using namespace M;
using namespace M::KGEN::Mojo;

//===--------------------------------------------------------------===//
// Plugin Initialization
//===--------------------------------------------------------------===//

static std::atomic<bool> g_plugin_initialized{false};
static std::atomic<M::Context *> existingContext;

// One-time initialisation flag for the global AsyncRT context.
static std::once_flag g_context_init_flag;

void M::KGEN::setLLDBPluginContext(ContextRef ctx) {
  auto oldCtx = ContextRef::take(existingContext.exchange(ctx.release()));
  // Let oldCtx get disposed to decrement the reference count on the previous
  // value, if any.
}

static ErrorOr<ContextRef> getOrCreateGlobalContext() {
  // Fast path: already initialised.
  if (auto ctx = ContextRef::copy(existingContext.load()))
    return ctx;

  // Crash reporting should only really be used when we "own" the program, and
  // that's not necessarily the case for LLDB... but we have no real better
  // place to put this, since the only better place ('main' function of the
  // LLDB driver) is upstream and hard to patch in our build.
  //
  // Ownership note: existingContext is the *sole* holder of the ContextRef.
  // We deliberately avoid keeping a separate static copy so that releasing
  // existingContext (ref-count → 0) immediately triggers Runtime::~Runtime()
  // → workQueue->shutdown(), which synchronously joins all AsyncRT worker
  // threads.  See the AddDestroyCallback in PluginInitialize for why this
  // ordering matters.
  std::call_once(g_context_init_flag, []() {
    auto ctxOr = Init::createContext(
        "mojo-lldb-plugin",
        Init::Options().withRuntimeOptions(AsyncRT::RuntimeOptions()
                                               .withCPUAffinity(false)
                                               .withMainWillNotDonate()));
    if (ctxOr.isError()) {
      llvm::errs() << "Failed to create mojo-lldb-plugin context: "
                   << ctxOr.getError() << "\n";
    } else {
      // Move ownership directly into existingContext — no extra copy retained.
      M::KGEN::setLLDBPluginContext(ctxOr.takeValue());
    }
  });

  if (auto ctx = ContextRef::copy(existingContext.load()))
    return ctx;
  return Error("failed to create mojo-lldb-plugin context (see stderr)");
}

static ContextRef getGlobalContext() {
  return getOrCreateGlobalContext().takeValue();
}

/// LLDB has two different types of plugin initialization, we support them both
/// here to provide flexibility for users. However, as we have the public API
/// enabled, initialization will go through `lldb::PluginInitialize`.

MODULAR_EXPORT bool LLDBPluginInitialize() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  LLVMLinkInMCJIT();

  // Ensure we have a legitimate context.
  auto ctxOr = getOrCreateGlobalContext();
  if (ctxOr.isError()) {
    llvm::errs() << "context error: " << ctxOr.getError() << "\n";
    return false;
  }

  // Initialize the various plugin components.
  MojoTypeSystem::Initialize(&getGlobalContext);
  MojoREPL::Initialize();
  MojoLanguage::Initialize();
  MojoLanguageRuntime::Initialize();
  g_plugin_initialized = true;
  return true;
}

MODULAR_EXPORT void LLDBPluginTerminate() {
  bool expected = true;
  if (!g_plugin_initialized.compare_exchange_strong(expected, false))
    return;
  // Reverse of initialization order (LIFO).
  MojoLanguageRuntime::Terminate();
  MojoLanguage::Terminate();
  MojoREPL::Terminate();
  MojoTypeSystem::Terminate();
  // Release the AsyncRT context as a safety fallback for callers that do not
  // go through PluginInitialize / AddDestroyCallback (e.g. LLDBPluginInitialize
  // called directly).  If the destroy callback already ran this is a no-op.
  M::KGEN::setLLDBPluginContext(ContextRef{});
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

// Disable unsupported DW_FORM value warnings, which are common in magic SDKs
// on Linux.
static void disableUnsupportedDWFormValueWarnings(lldb::SBDebugger &debugger) {
  lldb::SBExecutionContext exeCtx;
  lldb::SBCommandReturnObject result;
  debugger.GetCommandInterpreter().HandleCommand(
      "settings set plugin.symbol-file.dwarf.emit-unsupported-dwform-value "
      "false",
      exeCtx, result);
}

namespace lldb {
MODULAR_VISIBILITY_EXPORT bool PluginInitialize(SBDebugger debugger) {
  if (!LLDBPluginInitialize())
    return false;
  // LLVM's LoadPlugin lambda never calls LLDBPluginTerminate for dynamically
  // loaded plugins, so register atexit to clean up before static destructors.
  static bool registered = false;
  if (!registered) {
    std::atexit(LLDBPluginTerminate);
    registered = true;
  }

  // When this debugger is destroyed (SBDebugger::Destroy), release the AsyncRT
  // context before SBDebugger::Terminate() is called.  existingContext is the
  // sole ContextRef holder, so releasing it drops the ref-count to zero, which
  // triggers Runtime::~Runtime() -> workQueue->shutdown().  shutdown() blocks
  // until all AsyncRT worker threads have joined.  Without this sequencing the
  // live worker threads race with LLDB's internal thread pool teardown inside
  // Debugger::Terminate() and corrupt the heap.
  debugger.AddDestroyCallback(
      [](lldb::user_id_t, void *) {
        M::KGEN::setLLDBPluginContext(ContextRef{});
      },
      /*baton=*/nullptr);

  registerMojoCommands(debugger, getGlobalContext());
  registerLLVMDebugCommands(debugger);
  // We enable JIT debugging here so that this feature doesn't depend on
  // lldb init files or how LLDB was launched.
  enableJITDebugging(debugger);
  disableUnsupportedDWFormValueWarnings(debugger);
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
