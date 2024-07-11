//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build-project.h"
#include "../../common/Telemetry.h"

#include "AsyncRT/Init/Init.h"
#include "KGEN/MojoBuild/BSPClient.h"
#include "KGEN/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/MDialect/MDialect.h"

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

#include <optional>
#include <string>

#ifndef _WIN32_
#include <unistd.h>
#endif

using namespace M;

#define DRIVER_OPTIONS_PATH "BuildProject/BuildProjectOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct BuildProjectOptTable : public llvm::opt::PrecomputedOptTable {
  BuildProjectOptTable()
      : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// parseBuildProjectArgs
//===----------------------------------------------------------------------===//

namespace {
/// This struct provides an in-memory representation of the arguments passed to
/// the `build-project` subcommand for structured access.
struct BuildProjectArgs {
  /// The path to the build server's workspace. This is the root directory of
  /// the Mojo project being built.
  std::filesystem::path workspacePath;
};
} // namespace

/// Parse all command line arguments other than `--help`, and collect them into
/// the given `buildProjectArgs` struct.
static ErrorOrSuccess
parseBuildProjectArgs(const State &state, const llvm::opt::InputArgList &args,
                      BuildProjectArgs &buildProjectArgs) {
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return Error("too many inputs, expected exactly one");
  buildProjectArgs.workspacePath =
      args.getLastArgValue(options::OPT_INPUT, ".").str();
  std::error_code ec;
  buildProjectArgs.workspacePath =
      std::filesystem::weakly_canonical(buildProjectArgs.workspacePath, ec);
  if (ec)
    return Error("input path could not be made absolute: " + ec.message());

  return success();
}

/// Returns the path to the `mojo-build-server` executable, or an error if none
/// could be found.
static ErrorOr<std::string> getMojoBuildServerPath(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef path = config.getBuildServerPath();
  if (!std::filesystem::exists(path.str(), ec) || ec)
    return Error(llvm::formatv(
        "unable to resolve the mojo-build-server path at '{0}'", path));
  return path.str();
}

/// Launches a `mojo-build-server` executable and sends it messages to build
/// targets in the given workspace (the current working directory, by default).
static int buildProject(const State &subcommandState) {
  //===--------------------------------------------------------------------===//
  // Command line argument parsing & process initialization
  //===--------------------------------------------------------------------===//

  State state = subcommandState;
  BuildProjectOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  // Initialize crash reporting, etc.
  ErrorOr<ContextRef> ctxOrErr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(LLCL::RuntimeOptions()));
  if (ctxOrErr.isError())
    return state.reportError(ctxOrErr.getError());
  ContextRef ctx = std::move(*ctxOrErr);

  // Initialize telemetry.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread =
      logToolInvocationEventAsync(telemetryCtx, state.programName, args);

  // If `--help` appears anywhere within the arguments, print help text.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "BuildProject/BuildProjectOptionsHelpText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Determine the path to the project being built.
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return state.reportError("too many inputs, expected exactly one");
  std::filesystem::path rootUri(
      args.getLastArgValue(options::OPT_INPUT, ".").str());
  std::error_code ec;
  rootUri = std::filesystem::weakly_canonical(rootUri, ec);
  if (ec)
    return state.reportError("input path could not be made absolute: " +
                             ec.message());

  // Parse all command line arguments.
  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  BuildProjectArgs buildProjectArgs;
  if (auto err = parseBuildProjectArgs(state, args, buildProjectArgs))
    return state.reportError(err.getError());

  state.assertNoUnusedArguments(args);

  //===--------------------------------------------------------------------===//
  // Build server communication
  //===--------------------------------------------------------------------===//

  // Find the path to the mojo-build-server executable.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    return state.reportError(Twine("failed to read 'modular.cfg': ") +
                             configOr.getError());
  }
  KGEN::MojoConfig config = std::move(*configOr);

  auto serverPathOr = getMojoBuildServerPath(config);
  if (serverPathOr.isError())
    return state.reportError(serverPathOr.getError());
  std::string serverPath = *serverPathOr;

  // Create temporary files to marshall the input to, and output from, the
  // client (which in this case is this executable, the `mojo build-project`
  // command).
  auto inOr = TempFile::create("mojo-build-project-in-%%%%%%.json");
  auto outOr = TempFile::create("mojo-build-project-out-%%%%%%.json");
  if (inOr.isError())
    return state.reportError(inOr.getError());
  if (outOr.isError())
    return state.reportError(outOr.getError());
  TempFile in = std::move(*inOr);
  TempFile out = std::move(*outOr);

#ifndef _WIN32_
  std::FILE *inFile = fdopen(dup(in.getFD()), "r");
  int outFD = dup(out.getFD());
#else
  std::FILE *inFile = fdopen(_dup(in.getFD()), "r");
  int outFD = _dup(out.getFD());
#endif

  mlir::lsp::Logger::setLogLevel(mlir::lsp::Logger::Level::Debug);
  Build::BSPClient client(std::move(in), inFile, std::move(out), outFD,
                          "mojo-build-project", buildProjectArgs.workspacePath,
                          serverPath);
  ErrorOrSuccess result = client.run();
  if (result.isError())
    return state.reportError(result.getError());

  return 0;
}

void M::registerBuildProjectSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build-project", buildProject);
}
