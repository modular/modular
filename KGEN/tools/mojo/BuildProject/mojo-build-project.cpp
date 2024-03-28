//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build-project.h"

#include "Support/Driver/DriverSupport.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "BuildProject/BuildProjectOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct BuildProjectOptTable : public llvm::opt::PrecomputedOptTable {
  BuildProjectOptTable()
      : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// For now, an empty stub command that does nothing. Eventually, this will
/// interface with a separate `mojo-build-server` executable.
static int buildProject(const State &state) {
  // Parse command line arguments.
  BuildProjectOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  // If `--help` appears anywhere within the arguments, print help text.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "BuildProject/BuildProjectOptionsHelpText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  return EXIT_SUCCESS;
}

void M::registerBuildProjectSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build-project", buildProject);
}
