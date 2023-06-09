//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DRIVER_H
#define MOJO_DRIVER_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/CommandLine.h"
#include <unordered_map>

namespace M {

/// Additional driver state that is passed to each of the subcommand functions.
class State {
public:
  /// Initializes the driver state with the given program name.
  State(const char *programName) : programName(programName) {}

  /// Write the given error message to stderr and return a non-zero exit code.
  int reportError(Twine errorMessage) const;

private:
  /// The name of the executable that the user invoked.
  /// This is used for error reporting.
  const char *programName;
};

/// A mapping from each subcommand to a callback function that encapsulates the
/// logic of executing that subcommand.
class SubCommandRegistry {
public:
  /// A callback is a function that takes the driver `State` and returns an
  /// integer exit code.
  using Callback = std::function<int(const State &)>;

  /// Registers a subcommand and its callback.
  void addCallback(llvm::cl::SubCommand *subCommand, Callback callback);
  /// Fetches the callback function for a given subcommand.
  Callback getCallback(llvm::cl::SubCommand *subCommand);

private:
  /// The backing store for subcommands and their callback functions.
  DenseMap<llvm::cl::SubCommand *, Callback> callbacks;
};
} // namespace M

#endif // MOJO_DRIVER_H
