//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-debug.h"
#include "../Common/LLDB.h"
#include "llvm/Option/ArgList.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "Debug/DebugOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DebugOptTable : public llvm::opt::PrecomputedOptTable {
  DebugOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Launches LLDB with the Mojo plugin enabled.
/// Exits unsuccessfully if LLDB could not be found in the SDK.
static int debug(const State &state) {
  // Parse command line arguments. We forward most arguments to the underlying
  // invocation of lldb, and so don't check for invalid options.
  DebugOptTable options;
  unsigned unused = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, unused, unused);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Debug/DebugOptionsHelpText.inc"
    );
  }

  return invokeLLDB(state, args, {});
}

void M::registerDebugSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("debug", debug);
}
