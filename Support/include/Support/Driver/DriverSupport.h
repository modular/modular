//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DRIVER_DRIVERSUPPORT_H
#define SUPPORT_DRIVER_DRIVERSUPPORT_H

#include "Support/Driver/DiagnosticFormat.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llvm {
class MemoryBuffer;

namespace opt { // NOLINT, upstream namespace
class Arg;
class ArgList;
class InputArgList;
class OptTable;
} // namespace opt
} // namespace llvm

namespace M {

/// Open the Mojo source file at the given path for reading and return its
/// buffer, or if an error occurs, return an error message.
///
/// Note that this function considers it an error if the given `path` does not
/// end in a Mojo file extension (`.mojo`), even if the `path` refers
/// to `stdin` ("-").
ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> openMojoInputFile(StringRef path);

/// Resolve the absolute Mojo source file or package at the given path, or if an
/// error occurs, return an error message.
///
/// Note that this function considers it an error if the given `path` does not
/// end in a Mojo file extension (`.mojo`, `.mojoc`, or `.mojopkg`), even if
/// the `path` refers to `stdin` ("-").
ErrorOr<std::filesystem::path> resolveMojoInputFileOrPackage(StringRef path);

/// Returns the values of the positional arguments in `args`, in order. Each
/// option table gives them its own `INPUT` ID, which shared code cannot name,
/// so they are found by their option class.
std::vector<std::string> getInputValues(const llvm::opt::ArgList &args);

/// Returns the first positional argument in `args`, or null if there is none.
const llvm::opt::Arg *getFirstInputArg(const llvm::opt::ArgList &args);

/// The `--help` and `--help-hidden` texts that driver-tblgen generates for a
/// command.
struct HelpTexts {
#if __cplusplus >= 202002
  HelpTexts(std::u8string_view help, std::u8string_view helpHidden);
#endif
  HelpTexts(std::string_view help, std::string_view helpHidden)
      : help(help), helpHidden(helpHidden) {}

  std::string_view help;
  std::string_view helpHidden;
};

/// Additional driver state that is passed to each of the subcommand functions.
struct State {
  /// Initializes the driver state with the given program name and arguments.
  State(const char *programName, llvm::ArrayRef<const char *> arguments)
      : programName(programName), subcommand(nullptr), arguments(arguments) {}

  /// Initializes the driver state with the given program name, subcommand
  /// name, and arguments.
  State(const char *programName, const char *subcommand,
        llvm::ArrayRef<const char *> arguments)
      : programName(programName), subcommand(subcommand), arguments(arguments) {
  }

  /// The state object only holds pointers and references; it is cheap to copy.
  State(const State &) = default;

  /// Write the given error message to stderr and return a non-zero exit code.
  int reportError(const Twine &message) const;

  /// Write the given message to stderr, according to format config.
  void reportWarning(const Twine &message) const;

  bool areWarningsDisabled() const { return disableWarnings; }
  bool areWarningsAsErrors() const { return warningsAsErrors; }

  /// Print the given `helpText` to stdout and return a successful exit code.
#if __cplusplus >= 202002
  int printHelp(std::u8string_view helpText) const;
#endif
  int printHelp(std::string_view helpText) const;

  /// Parses `arguments` with `optTable` into `args`. With `stopAtFirstInput`,
  /// the arguments after the first positional one are left unparsed, for the
  /// program that the command runs.
  ///
  /// If `--help` or `--help-hidden` was given, prints the matching text from
  /// `helpTexts`, through `printHelpText` when it is set, and returns its exit
  /// code. Otherwise returns nullopt.
  std::optional<int>
  parseArguments(const llvm::opt::OptTable &optTable,
                 const HelpTexts &helpTexts, llvm::opt::InputArgList &args,
                 bool stopAtFirstInput = false,
                 llvm::function_ref<int(std::string_view)> printHelpText = {});

  /// Reports the first of the marshalling `errors`, or else the option that
  /// `parseArguments` found without its value and each unknown argument in
  /// `args`, and returns an unsuccessful exit code if there were any.
  /// Otherwise returns a successful exit code.
  int rejectInvalidArguments(const llvm::opt::InputArgList &args,
                             llvm::ArrayRef<std::string> errors = {}) const;

  /// In debug modes, asserts that all `args` have been "claimed." This is a
  /// no-op in release modes.
  ///
  /// Checking whether an argument was passed in on the command line, or
  /// checking its value, results in that argument being "claimed." Unclaimed
  /// arguments are those that `mojo` never checks, and so have no bearing on
  /// the behavior of `mojo`. This represents a Modular employee programming
  /// error.
  void assertNoUnusedArguments(const llvm::opt::InputArgList &args) const;

  /// The name of the executable that the user invoked.
  /// This is used for error reporting.
  const char *programName;

  /// The name of the subcommand that the user selected, or null if the user
  /// invoked the top-level driver.
  const char *subcommand;

  /// The command line arguments that the user provided, excluding the program
  /// and subcommand names.
  const llvm::ArrayRef<const char *> arguments;

  /// The output format for driver errors, as reported by `reportError`.
  DiagnosticFormat diagnosticFormat = DiagnosticFormat::Text;

  /// Whether to report warnings or silently ignore them.
  bool disableWarnings = false;

  /// Whether to treat warnings as errors.
  bool warningsAsErrors = false;

  /// The option that ended the arguments without its value, or null.
  const char *missingValueOption = nullptr;
};

/// A mapping from each subcommand to a callback function that encapsulates the
/// logic of executing that subcommand.
class SubcommandRegistry {
public:
  /// A callback is a function that takes the driver `State` and returns an
  /// integer exit code.
  using Callback = std::function<int(const State &)>;

  /// Registers a subcommand and its callback.
  void addCallback(StringRef subcommand, Callback callback,
                   std::optional<StringRef> deprecatedFor = std::nullopt);
  /// Attempts to return the callback function associated with the given
  /// subcommand name, or an error if no match can be found.
  ErrorOr<Callback> getCallback(StringRef subcommand);

private:
  /// The backing store for subcommands and their callback functions.
  llvm::StringMap<std::pair<Callback, std::optional<StringRef>>> callbacks;
};
} // namespace M

#endif // SUPPORT_DRIVER_DRIVERSUPPORT_H
