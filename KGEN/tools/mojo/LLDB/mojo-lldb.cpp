//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-lldb.h"
#include "../Common/LLDB.h"
#include "llvm/Option/ArgList.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "REPL/REPLOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct LLDBOptTable : public llvm::opt::PrecomputedOptTable {
  LLDBOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Launches Mojo within the LLDB debugger.
/// Exits unsuccessfully if LLDB could not be found in the user's PATH.
static int lldb(const State &state) {
  // Parse command line arguments. We forward most arguments to the underlying
  // invocation of lldb, and so don't check for invalid options.
  LLDBOptTable options;
  unsigned unused = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, unused, unused);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "LLDB/LLDBOptionsHelpText.inc"
    );
  }

  return invokeLLDB(state, args, {});
}

void M::registerLLDBSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("lldb", lldb);
}
