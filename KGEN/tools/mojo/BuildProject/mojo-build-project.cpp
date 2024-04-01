//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build-project.h"
#include "../../common/Telemetry.h"

#include "KGEN/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/Init/Init.h"
#include "Support/MDialect/MDialect.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

#include <optional>
#include <string>

using namespace M;

#define DRIVER_OPTIONS_PATH "BuildProject/BuildProjectOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct BuildProjectOptTable : public llvm::opt::PrecomputedOptTable {
  BuildProjectOptTable()
      : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

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

/// For now, this simply launches a `mojo-build-server` executable and sends it
/// initialization and exit messages. Eventually, this will send messages
/// that result in the compilation of a Mojo project.
static int buildProject(const State &state) {
  //===--------------------------------------------------------------------===//
  // Command line argument parsing & process initialization
  //===--------------------------------------------------------------------===//

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
  initializeTelemetry(telemetryCtx, state.programName, args);

  // If `--help` appears anywhere within the arguments, print help text.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "BuildProject/BuildProjectOptionsHelpText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Assert that we've parsed all command line arguments.
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

  // For now, as a proof of concept, send delimited messages to the build
  // server.
  auto inOr = writeTempFile(
      "mojo-build-project-%%%%%%.json", [](llvm::raw_ostream &in) {
        in << "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"build/initialize\","
           << "\"params\":{\"displayName\":\"mojo-build-project\"}}\n";
        in << "// -----\n";
        in << "{\"jsonrpc\":\"2.0\",\"method\":\"exit\"}\n";
      });
  if (inOr.isError())
    return state.reportError(inOr.getError());
  TempFile in = std::move(*inOr);

  return llvm::sys::ExecuteAndWait(
      serverPath, {serverPath},
      /*Env=*/std::nullopt,
      /*Redirects=*/
      {in.getPath().c_str(), std::nullopt, std::nullopt});
}

void M::registerBuildProjectSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build-project", buildProject);
}
