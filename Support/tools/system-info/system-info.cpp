//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "Support/Host.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>

using namespace M;
using namespace llvm;

namespace {

struct SystemInfoCLIOptions {

  llvm::cl::opt<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-")};

  llvm::cl::list<HostProperty> QueryProperty{
      "query", M::cl::desc("Available Queries:"),
      M::cl::values(
          clEnumValN(HostProperty::TargetTriple, "target-triple",
                     "Host target triple"),
          clEnumValN(HostProperty::OS, "os", "Host operating system"),
          clEnumValN(HostProperty::Arch, "arch", "Host CPU architecture"),
          clEnumValN(HostProperty::CPUModel, "cpu-model",
                     "Host CPU model name"),
          clEnumValN(HostProperty::Features, "features",
                     "Host CPU features printed as comma-separated values"),
          clEnumValN(HostProperty::CoreCount, "core-count",
                     "Host number of cores"),
          clEnumValN(HostProperty::L1CacheSize, "l1-cache-size",
                     "Host L1 DCache size"),
          clEnumValN(HostProperty::L2CacheSize, "l2-cache-size",
                     "Host L2 DCache size"),
          clEnumValN(HostProperty::L3CacheSize, "l3-cache-size",
                     "Host L3 DCache size"),
          clEnumValN(HostProperty::L4CacheSize, "l4-cache-size",
                     "Host L4 DCache size"),
          clEnumValN(HostProperty::Affinities, "affinities",
                     "Preferred CPU ids for numPhysicalCores threads if both "
                     "CPUSystemInfo and thread affinities are supported.")),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};
};
} // namespace

static int reportError(Twine errorMessage) {
  llvm::errs() << "system-info: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  SystemInfoCLIOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular System Info Tool");

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

  auto hostMachineOr = getHostMachineInfo();
  if (hostMachineOr.isError())
    return reportError(hostMachineOr.getError());

  HostMachineInfo hostInfo = hostMachineOr.takeValue();

  if (cli.QueryProperty.empty()) {
    hostInfo.print(os);
    os.flush();
    outputFile->keep();
    return EXIT_SUCCESS;
  }

  for (auto query : cli.QueryProperty)
    hostInfo.print(query, os);

  os.flush();
  outputFile->keep();

  return EXIT_SUCCESS;
}
