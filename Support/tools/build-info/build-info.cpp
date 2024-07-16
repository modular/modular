//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// TODO: Reduce duplication with system-info tool

#include "Support/BuildInfo.h"
#include "Support/CommandLine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

using namespace M;
using namespace llvm;

namespace {

struct BuildInfoCLIOptions {

  llvm::cl::opt<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-")};

  llvm::cl::list<BuildProperty> queryProperty{
      "query", M::cl::desc("Available Queries:"),
      M::cl::values(clEnumValN(BuildProperty::ModularVersion, "modular-version",
                               "Modular version string"),
                    clEnumValN(BuildProperty::GitRevision, "git-revision",
                               "Modular Git revision"),
                    clEnumValN(BuildProperty::BuildType, "build-type",
                               "Build type used to build Modular"),
                    clEnumValN(BuildProperty::KernelsBuildType,
                               "kernels-build-type",
                               "Build type used to build kernels"),
                    clEnumValN(BuildProperty::AsyncRTMaxProfilingLevel,
                               "asyncrt-max-profiling-level",
                               "Maximum profiling level built into AsyncRT"),
                    clEnumValN(BuildProperty::LLVMTargets, "llvm-targets",
                               "The LLVM targets Modular is built with"),
                    clEnumValN(BuildProperty::PreferredMemoryAlignment,
                               "preferred-memory-alignment",
                               "Memory alignment Modular is built to prefer")),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};
};
} // namespace

static int reportError(const Twine &errorMessage) {
  llvm::errs() << "build-info: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  BuildInfoCLIOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Build Info Tool");

  auto outFilePathStr = cli.outputFilename.getValue();

  std::error_code ec;
  if (outFilePathStr != "-" && !std::filesystem::exists(outFilePathStr, ec) &&
      !ec) {
    auto outFilePath = std::filesystem::path(outFilePathStr);
    if (outFilePath.has_parent_path())
      std::filesystem::create_directories(outFilePath.parent_path(), ec);
  }
  // If anything failed, report the failure.
  if (ec)
    exit(reportError("std::filesystem: " + ec.message() + ": " +
                     outFilePathStr));

  std::error_code error;
  auto outputFile = std::make_unique<llvm::ToolOutputFile>(
      outFilePathStr, error, llvm::sys::fs::OF_None);
  if (error)
    exit(reportError("Cannot open output file: '" + outFilePathStr +
                     "': " + error.message()));

  auto &os = outputFile ? outputFile->os() : llvm::outs();

  BuildInfo buildInfo = getBuildInfo();

  if (cli.queryProperty.empty()) {
    buildInfo.print(os);
    os.flush();
    outputFile->keep();
    return EXIT_SUCCESS;
  }

  for (auto query : cli.queryProperty)
    buildInfo.print(query, os);

  os.flush();
  outputFile->keep();

  return EXIT_SUCCESS;
}
