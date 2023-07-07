//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Demangle/mojo-demangle.h"
#include "Doc/mojo-doc.h"
#include "Package/mojo-package.h"
#include "REPL/mojo-repl.h"
#include "Run/mojo-run.h"

#include "Config/Version.h"
#include "Support/Driver/DriverSupport.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "DriverOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DriverOptTable : public llvm::opt::PrecomputedOptTable {
  DriverOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// `main` entry point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Install LLVM signal handlers and convert `argc` and `argv` for Windows
  // hosts.
  llvm::InitLLVM initLLVM(argc, argv);

  // Store command line arguments and record the program name.
  SmallVector<const char *, 256> argvStorage(argv, argv + argc);
  const char *programName = argvStorage.front();
  ArrayRef<const char *> arguments = ArrayRef(argvStorage).slice(1);

  // Register subcommands and their options.
  SubcommandRegistry registry;
  registerDemangleSubcommand(registry);
  registerDocSubcommand(registry);
  registerPackageSubcommand(registry);
  registerREPLSubcommand(registry);
  registerRunSubcommand(registry);

  // If the user hasn't provided any arguments, treat this as the `repl`
  // subcommand.
  if (arguments.empty())
    return registry.getCallback("repl").get()(State(programName, arguments));

  // Otherwise, parse the first argument: it's either a subcommand, or one of a
  // handful of top-level driver options that we allow in this first position.
  DriverOptTable options;
  llvm::opt::InputArgList args(arguments.begin(), arguments.end());
  unsigned index = 0;
  std::unique_ptr<llvm::opt::Arg> firstArg = options.ParseOneArg(args, index);
  switch (firstArg->getOption().getID()) {
  case options::OPT_version: {
    // Print the version and exit.
    ModularVersion version = getModularVersion();
    llvm::outs() << "mojo " << version.major << '.' << version.minor << '.'
                 << version.patch << "\n";
    return 0;
  }
  case options::OPT_help:
    return State(programName, ArrayRef(arguments).slice(1))
        .printHelp(
#include "DriverOptionsHelpText.inc"
        );
  case options::OPT_INPUT:
    // This isn't an option; we'll interpret it as a subcommand.
    break;
  default: {
    // Otherwise, we don't know what this is. Report an error.
    return State(programName, ArrayRef(arguments).slice(1))
        .reportError(llvm::formatv("unrecognized option '{0}'",
                                   firstArg->getAsString(args)));
  }
  }

  // Store the program name and subcommand arguments in the driver state object.
  State state(programName, arguments.slice(index));

  // Find the callback for the subcommand name the user provided, or exit with
  // an error if no match is found.
  ErrorOr<SubcommandRegistry::Callback> callback =
      registry.getCallback(firstArg->getAsString(args));
  if (callback.isError())
    return state.reportError(callback.getError());

  // If we found a matching subcommand, invoke its callback.
  return callback.get()(state);
}
