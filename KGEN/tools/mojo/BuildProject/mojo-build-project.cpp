//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build-project.h"

#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
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
static ErrorOr<std::string> getMojoBuildServerPath(const char *programName) {
  std::string executable = llvm::sys::fs::getMainExecutable(
      programName, (void *)getMojoBuildServerPath);
  if (executable.empty())
    return Error("could not determine `mojo` path");

  // For now, `mojo-build-server` is found relative to `mojo` -- it exists in
  // the same directory. In the future, its path will be stored in the Modular
  // config file.
  std::string path = std::filesystem::path(executable).parent_path().string();
  return toModularErrorOr(
      llvm::sys::findProgramByName("mojo-build-server",
                                   /*Paths=*/ArrayRef<StringRef>(path)));
}

/// For now, this simply launches a `mojo-build-server` executable and sends it
/// initialization and exit messages. Eventually, this will send messages
/// that result in the compilation of a Mojo project.
static int buildProject(const State &state) {
  //===--------------------------------------------------------------------===//
  // Command line argument parsing
  //===--------------------------------------------------------------------===//
  BuildProjectOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

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

  auto serverPathOrErr = getMojoBuildServerPath(state.programName);
  if (serverPathOrErr.isError())
    return state.reportError(serverPathOrErr.getError());
  std::string serverPath = *serverPathOrErr;

  // For now, as a proof of concept, send delimited messages to the build
  // server.
  auto inOrErr = writeTempFile(
      "mojo-build-project-%%%%%%.json", [](llvm::raw_ostream &in) {
        in << "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"build/initialize\","
           << "\"params\":{\"displayName\":\"mojo-build-project\"}}\n";
        in << "// -----\n";
        in << "{\"jsonrpc\":\"2.0\",\"method\":\"exit\"}\n";
      });
  if (inOrErr.isError())
    return state.reportError(inOrErr.getError());
  TempFile in = std::move(*inOrErr);

  return llvm::sys::ExecuteAndWait(
      serverPath, {serverPath},
      /*Env=*/std::nullopt,
      /*Redirects=*/
      {in.getPath().c_str(), std::nullopt, std::nullopt});
}

void M::registerBuildProjectSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build-project", buildProject);
}
