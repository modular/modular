//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-driver.h"
#include "mojo-demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace M;

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

int State::reportError(Twine errorMessage) const {
  llvm::errs() << programName << ": " << errorMessage << "\n";
  return EXIT_FAILURE;
}

//===----------------------------------------------------------------------===//
// SubCommandRegistry
//===----------------------------------------------------------------------===//

void SubCommandRegistry::addCallback(llvm::cl::SubCommand *subCommand,
                                     SubCommandRegistry::Callback callback) {
  assert(callbacks.count(subCommand) == 0 && "subcommand already registered");
  assert(callback && "callback cannot be empty");
  callbacks.insert({subCommand, callback});
}

SubCommandRegistry::Callback
SubCommandRegistry::getCallback(llvm::cl::SubCommand *subCommand) {
  auto it = callbacks.find(subCommand);
  assert(it != callbacks.end() && "subcommand is not registered");
  return it->second;
}

//===----------------------------------------------------------------------===//
// `main` entry point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Install LLVM signal handlers and convert `argc` and `argv` for Windows
  // hosts.
  llvm::InitLLVM initLLVM(argc, argv);

  // Store the program name in the driver state object.
  State state(argv[0]);

  // Register subcommands and their options.
  SubCommandRegistry registry;
  registerDemangleSubCommand(registry);

  // Parse the command line arguments. This exits the process if invalid
  // arguments are provided, or if `--help` is specified on the command line.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "The Mojo🔥 command-line interface.");

  // Dispatch to subcommand entry points.
  for (llvm::cl::SubCommand *subcommand :
       llvm::cl::getRegisteredSubcommands()) {
    if (*subcommand) {
      // The top-level subcommand is a type of subcommand, but it isn't a valid
      // choice for the user.
      if (subcommand == &llvm::cl::SubCommand::getTopLevel())
        break;

      // Otherwise, invoke the subcommand's callback function.
      return registry.getCallback(subcommand)(state);
    }
  }

  // If the user invoked the tool with no subcommand arguments, print help
  // and exit with a non-zero status code.
  llvm::cl::PrintHelpMessage(false, true);
  return 1;
}
