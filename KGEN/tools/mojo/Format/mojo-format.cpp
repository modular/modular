//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-format.h"

#include "../../common/Telemetry.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Init/Init.h"
#include "KGEN/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

#include <filesystem>

using namespace M;

#define DRIVER_OPTIONS_PATH "Format/FormatOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct FormatOptTable : public llvm::opt::PrecomputedOptTable {
  FormatOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Format a set of Mojo source files. Returns an integer representing a
/// successful exit code if formatting succeeded, otherwise returns a failure
/// code.
static int format(const State &state) {
  // Parse command line arguments.
  FormatOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Format/FormatOptionsHelpText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Process the input files.
  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (!args.hasArg(options::OPT_INPUT))
    return state.reportError("no inputs provided");

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext("mojo", Init::Options());
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Check that the inputs are all valid Mojo/Python files, or directories.
  std::error_code ec;
  bool hasStdin = false;
  for (const std::string &input : inputs) {
    // Allow "-" to represent stdin.
    if (input == "-") {
      if (inputs.size() > 1)
        return state.reportError("cannot mix '-' with other inputs");
      hasStdin = true;
      break;
    }

    std::filesystem::path inputPath(input);
    if (!std::filesystem::exists(inputPath, ec)) {
      return state.reportError(
          llvm::formatv("input '{0}' does not exist", input));
    }

    if (std::filesystem::is_directory(inputPath, ec))
      continue;
    if (ec)
      return state.reportError(ec.message());

    if (!llvm::is_contained(ArrayRef<StringRef>{".mojo", ".🔥", ".py"},
                            inputPath.extension().string())) {
      return state.reportError(
          llvm::formatv("invalid input '{0}', expected a source .mojo/.🔥/.py "
                        "file, or a directory",
                        input));
    }
  }

  StringRef lineLengthArg = args.getLastArgValue(options::OPT_line_length);
  if (!lineLengthArg.empty()) {
    int lineLength = 0;
    if (lineLengthArg.getAsInteger(10, lineLength)) {
      return state.reportError(llvm::formatv(
          "expected integer value for --line-length, but got '{0}'",
          lineLengthArg));
    }
  }

  // Check for additional options.
  bool isQuiet = args.hasArg(options::OPT_quiet);

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Read the mojo configuration.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    return state.reportError(Twine("failed to parse 'modular.cfg': ") +
                             configOr.getError());
  }
  KGEN::MojoConfig config = std::move(*configOr);

  // Resolve the path to mblack.
  StringRef mblack = config.getMBlackPath();
  if (!std::filesystem::exists(mblack.str(), ec) || ec ||
      !llvm::sys::fs::can_execute(mblack)) {
    return state.reportError("unable to resolve Mojo formatter in PATH");
  }

  // Forward the curated options to mblack.
  SmallVector<StringRef> mblackArgs = {mblack, "--fast", "--preview"};
  if (!lineLengthArg.empty()) {
    mblackArgs.push_back("--line-length");
    mblackArgs.push_back(lineLengthArg);
  }
  // If we're formatting stdin, we need to tell mblack to expect a Mojo file.
  if (hasStdin)
    llvm::append_range(mblackArgs, ArrayRef<StringRef>{"-t", "mojo"});
  if (isQuiet)
    mblackArgs.push_back("-q");
  llvm::append_range(mblackArgs, inputs);
  return llvm::sys::ExecuteAndWait(mblack, mblackArgs);
}

void M::registerFormatSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("format", format);
}
