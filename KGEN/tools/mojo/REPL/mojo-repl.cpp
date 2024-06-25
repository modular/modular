//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-repl.h"
#include "../../common/Telemetry.h"
#include "../Common/LLDB.h"
#include "LLCL/Init/Init.h"
#include "llvm/Option/ArgList.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "REPL/REPLOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct REPLOptTable : public llvm::opt::PrecomputedOptTable {
  REPLOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Launches the Mojo REPL, which is in fact an invocation of
/// `lldb --repl-language mojo`. Exits unsuccessfully if LLDB could not be found
/// in the user's PATH.
static int repl(const State &state) {
  // Parse command line arguments. We forward most arguments to the underlying
  // invocation of lldb, and so don't check for invalid options.
  REPLOptTable options;
  unsigned unused = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, unused, unused);

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext("mojo", Init::Options());
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Initialize telemetry.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "REPL/REPLOptionsHelpText.inc"
    );
  }

  SmallVector<std::string> lldbArgs = {"--one-line-before-file",
                                       "settings set show-progress false",
                                       "--repl-language", "mojo", "--repl"};
  llvm::append_range(lldbArgs, state.arguments);
  return invokeLLDB(state, lldbArgs);
}

void M::registerREPLSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("repl", repl);
}
