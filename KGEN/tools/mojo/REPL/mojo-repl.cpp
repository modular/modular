//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-repl.h"
#include "../Common/Telemetry.h"

#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Program.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "REPL/REPLOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct REPLOptTable : public llvm::opt::PrecomputedOptTable {
  REPLOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Launches the Mojo REPL, which is in fact an invocation of
/// `lldb --repl-language mojo`. Exits unsuccessfully if lldb could not be found
/// in the user's PATH.
static int repl(const State &state) {
  // Parse command line arguments. We forward most arguments to the underlying
  // invocation of lldb, and so don't check for invalid options.
  REPLOptTable options;
  unsigned unused = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, unused, unused);

  if (args.hasArg(options::OPT_help, options::OPT_help_text)) {
    return state.printHelp(/*plainText=*/args.hasArg(options::OPT_help_text),
#include "REPL/REPLOptionsHelpText.inc"
    );
  }

  // Initialize telemetry.
  auto telemetryCtx = LLCL::RCRef<Telemetry::TelemetryContext>::create();
  initializeTelemetry(telemetryCtx.copy(), state, args);

  // Find the path to lldb.
  llvm::ErrorOr<std::string> lldb = llvm::sys::findProgramByName("lldb");
  if (!lldb)
    return state.reportError("lldb must exist in our PATH to launch the REPL");

  // We forward all unparsed command line arguments to lldb, as values for the
  // `--repl` option.
  SmallVector<StringRef> lldbArgs(state.arguments);
  lldbArgs.insert(lldbArgs.begin(),
                  {lldb.get(), "--repl-language", "mojo", "--repl"});
  return llvm::sys::ExecuteAndWait(lldb.get(), lldbArgs);
}

void M::registerREPLSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("repl", repl);
}
