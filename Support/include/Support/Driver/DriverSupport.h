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

#include <filesystem>
#include <memory>
#include <string_view>

namespace llvm {
class MemoryBuffer;

namespace opt {
class InputArgList;
class OptSpecifier;
} // namespace opt
} // namespace llvm

namespace M {

/// Open the Mojo source file at the given path for reading and return its
/// buffer, or if an error occurs, return an error message.
///
/// Note that this function considers it an error if the given `path` does not
/// end in a Mojo file extension (`.mojo` or `.🔥`), even if the `path` refers
/// to `stdin` ("-").
ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> openMojoInputFile(StringRef path);

/// Resolve the absolute Mojo source file or package at the given path, or if an
/// error occurs, return an error message.
///
/// Note that this function considers it an error if the given `path` does not
/// end in a Mojo file extension (`.mojo`, `.🔥`, `.mojopkg`, `.📦`), even if
/// the `path` refers to `stdin` ("-").
ErrorOr<std::filesystem::path> resolveMojoInputFileOrPackage(StringRef path);

/// Additional driver state that is passed to each of the subcommand functions.
struct State {
  /// Initializes the driver state with the given program name and arguments.
  State(const char *programName, ArrayRef<const char *> arguments)
      : programName(programName), subcommand(nullptr), arguments(arguments) {}

  /// Initializes the driver state with the given program name, subcommand
  /// name, and arguments.
  State(const char *programName, const char *subcommand,
        ArrayRef<const char *> arguments)
      : programName(programName), subcommand(subcommand), arguments(arguments) {
  }

  /// Write the given error message to stderr and return a non-zero exit code.
  int reportError(Twine errorMessage) const;

  /// Print the given `helpText` to stdout and return a successful exit code.
#if __cplusplus >= 202002
  int printHelp(std::u8string_view helpText) const;
#endif
  int printHelp(std::string_view helpText) const;

  /// If `args` has any unknown arguments (as indicated by the
  /// `unknownOptionID`, which is defined independently in each driver command),
  /// report an error for each of them and return an unsuccessful exit code.
  /// Otherwise, return a successful exit code.
  int rejectUnknownArguments(llvm::opt::InputArgList &args,
                             llvm::opt::OptSpecifier unknownOptionID) const;

  /// The name of the executable that the user invoked.
  /// This is used for error reporting.
  const char *programName;

  /// The name of the subcommand that the user selected, or null if the user
  /// invoked the top-level driver.
  const char *subcommand;

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
