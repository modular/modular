//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-driver.h"
#include "Demangle/mojo-demangle.h"
#include "Doc/mojo-doc.h"
#include "REPL/mojo-repl.h"

#include "Config/Version.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"

using namespace M;

#define MOJO_DRIVER_OPTIONS_PATH "DriverOptions.inc"
#include "OptTable.inc"

namespace {
struct DriverOptTable : public llvm::opt::PrecomputedOptTable {
  DriverOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

int State::reportError(Twine errorMessage) const {
  llvm::errs() << programName << ": error: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

//===----------------------------------------------------------------------===//
// SubcommandRegistry
//===----------------------------------------------------------------------===//

void SubcommandRegistry::addCallback(StringRef subcommand,
                                     SubcommandRegistry::Callback callback) {
  std::string cmd = subcommand.str();
  assert(callbacks.count(cmd) == 0 && "subcommand already registered");
  assert(callback && "callback cannot be empty");
  callbacks.insert({cmd, callback});
}

llvm::Expected<SubcommandRegistry::Callback>
SubcommandRegistry::getCallback(StringRef subcommand) {
  auto it = callbacks.find(subcommand.str());
  if (it != callbacks.end())
    return it->second;

  // The user provided a subcommand name we don't recognize; return an error
  // message.
  std::string message = ("no such command '" + subcommand + "'").str();

  // If there are any close matches, point those out in the message.
  std::string nearest;
  unsigned minDistance = std::numeric_limits<unsigned>::max();
  for (const auto &kv : callbacks) {
    unsigned distance = subcommand.edit_distance(kv.first());
    if (distance < minDistance) {
      minDistance = distance;
      nearest = kv.first();
    }
  }
  if (minDistance <= 2)
    message += ". Did you mean '" + nearest + "'?";

  return llvm::make_error<llvm::StringError>(message,
                                             llvm::inconvertibleErrorCode());
}

//===----------------------------------------------------------------------===//
// `main` entry point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Install LLVM signal handlers and convert `argc` and `argv` for Windows
  // hosts.
  llvm::InitLLVM initLLVM(argc, argv);

  // Store command line arguments and  record the program name.
  SmallVector<const char *, 256> argvStorage(argv, argv + argc);
  const char *programName = argvStorage.front();
  ArrayRef<const char *> arguments = ArrayRef(argvStorage).slice(1);

  const std::string usage =
      llvm::formatv("{0} <command> [options]", programName);

  if (argc <= 1) {
    // The user hasn't provided any arguments; print usage and exit.
    return State(programName, arguments)
        .reportError(llvm::formatv("no command provided\n\n"
                                   "usage: {0}\n\n"
                                   "For more information, try '{1} --help'.",
                                   usage, programName));
  }

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
  case options::OPT_help: {
    // Print the top-level driver help and exit.
    options.printHelp(llvm::outs(), usage.c_str(),
                      "The Mojo🔥 command-line interface.");
    return 0;
  }
  case options::OPT_INPUT: {
    // This isn't an option; we'll interpret it as a subcommand.
    break;
  }
  default: {
    // Otherwise, we don't know what this is. Report an error.
    return State(programName, ArrayRef(arguments).slice(1))
        .reportError(llvm::formatv("unrecognized option '{0}'",
                                   firstArg->getAsString(args)));
  }
  }

  // Register subcommands and their options.
  SubcommandRegistry registry;
  registerDemangleSubCommand(registry);
  registerDocSubCommand(registry);
  registerREPLSubCommand(registry);

  // Store the program name and subcommand arguments in the driver state object.
  State state(programName, arguments.slice(index));

  // Find the callback for the subcommand name the user provided, or exit with
  // an error if no match is found.
  llvm::Expected<SubcommandRegistry::Callback> callback =
      registry.getCallback(firstArg->getAsString(args));
  if (!callback) {
    llvm::handleAllErrors(callback.takeError(),
                          [&](const llvm::ErrorInfoBase &err) {
                            state.reportError(err.message());
                          });
    return 1;
  }

  // If we found a matching subcommand, invoke its callback.
  return callback.get()(state);
}
