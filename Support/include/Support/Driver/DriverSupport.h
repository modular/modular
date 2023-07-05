//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DRIVER_DRIVERSUPPORT_H
#define SUPPORT_DRIVER_DRIVERSUPPORT_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"

namespace M {

/// Additional driver state that is passed to each of the subcommand functions.
struct State {
  /// Initializes the driver state with the given program name and arguments.
  State(const char *programName, ArrayRef<const char *> arguments)
      : programName(programName), arguments(arguments) {}

  /// Write the given error message to stderr and return a non-zero exit code.
  int reportError(Twine errorMessage) const;

  /// Prints the given `helpText` for the current command and returns a
  /// successful exit code.
  int printHelp(Twine helpText) const;

  /// The name of the executable that the user invoked.
  /// This is used for error reporting.
  const char *programName;

  /// The command line arguments that the user provided, excluding the program
  /// and subcommand names.
  const ArrayRef<const char *> arguments;
};

/// A mapping from each subcommand to a callback function that encapsulates the
/// logic of executing that subcommand.
class SubcommandRegistry {
public:
  /// A callback is a function that takes the driver `State` and returns an
  /// integer exit code.
  using Callback = std::function<int(const State &)>;

  /// Registers a subcommand and its callback.
  void addCallback(StringRef subcommand, Callback callback);
  /// Attempts to return the callback function associated with the given
  /// subcommand name, or an error if no match can be found.
  ErrorOr<Callback> getCallback(StringRef subcommand);

private:
  /// The backing store for subcommands and their callback functions.
  llvm::StringMap<Callback> callbacks;
};
} // namespace M

#endif // SUPPORT_DRIVER_DRIVERSUPPORT_H
