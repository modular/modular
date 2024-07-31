//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Build/mojo-build.h"
#include "Debug/mojo-debug.h"
#include "Demangle/mojo-demangle.h"
#include "Doc/mojo-doc.h"
#include "Format/mojo-format.h"
#include "Package/mojo-package.h"
#include "REPL/mojo-repl.h"
#include "Run/mojo-run.h"
#include "Test/mojo-test.h"

#include "Config/Version.h"
#include "KGEN/Support/Configuration.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/LogicalResult.h"
#include "Support/Process.h"

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
  llvm::setBugReportMsg(
      "Please submit a bug report to https://github.com/modularml/mojo/issues "
      "and include the crash backtrace along with all the relevant source "
      "codes.\n");

  // Store command line arguments and record the program name.
  SmallVector<const char *, 256> argvStorage(argv, argv + argc);
  const char *programName = argvStorage.front();
  ArrayRef<const char *> arguments = ArrayRef(argvStorage).slice(1);

  // Register subcommands and their options.
  SubcommandRegistry registry;
  registerBuildSubcommand(registry);
  registerDemangleSubcommand(registry);
  registerDocSubcommand(registry);
  registerFormatSubcommand(registry);
  registerPackageSubcommand(registry);
  registerREPLSubcommand(registry);
  registerDebugSubcommand(registry);
  registerRunSubcommand(registry);
  registerTestSubcommand(registry);

  // If the user hasn't provided any arguments, treat this as the `repl`
  // subcommand.
  if (arguments.empty())
    return registry.getCallback("repl").get()(
        State(programName, "repl", arguments));

  // Otherwise, parse the first argument; it could be:
  // - One of a handful of top-level driver options that we allow in this first
  //   position.
  // - One of the registered subcommands.
  // - A positional ("input") argument, or an option, that we don't recognize.
  DriverOptTable options;
  llvm::opt::InputArgList args(arguments.begin(), arguments.end());
  unsigned index = 0;
  std::unique_ptr<llvm::opt::Arg> firstArg = options.ParseOneArg(args, index);
  switch (firstArg->getOption().getID()) {
  case options::OPT_version: {
    // Print the version and exit.
    ModularVersion version = getModularVersion();
    llvm::outs() << llvm::formatv("mojo {0}.{1}.{2}{3} ({4})\n", version.major,
                                  version.minor, version.patch, version.label,
                                  version.revision);
    return 0;
  }
  case options::OPT_help:
    // Print the top level driver help text and exit.
    return State(programName, ArrayRef(arguments).slice(1))
        .printHelp(
#include "DriverOptionsHelpText.inc"
        );
  case options::OPT_INPUT: {
    // This could be a subcommand, or it could be an input file for the `run`
    // subcommand.
    std::string arg = firstArg->getAsString(args);
    ErrorOr<SubcommandRegistry::Callback> callback = registry.getCallback(arg);
    // If it's a subcommand, invoke its callback.
    if (succeeded(callback))
      return callback.get()(
          State(programName, arg.c_str(), arguments.slice(index)));

    // If it looks like a Mojo source file, invoke the `run` subcommand.
    State state(programName, "run", arguments);
    StringRef argRef(arg);
    if (argRef.ends_with(".mojo") || argRef.ends_with(".🔥"))
      return registry.getCallback("run").get()(state);

    // Otherwise, we don't know what this is; return an error.
    return state.reportError(callback.getError());
  }
  default:
    // This is some sort of option, so we'll pass it along to the `run` command
    // to parse. This allows for invocations such as `mojo -Ifoo Foo.mojo`.
    return registry.getCallback("run").get()(
        State(programName, "run", arguments));
  }
}
