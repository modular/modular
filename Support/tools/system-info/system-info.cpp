//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "Support/Host.h"
#include "Support/SIMD.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <filesystem>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>

using namespace M;
using namespace llvm;

namespace {
enum class QuerySystemProperty {
  TargetTriple,
  OS,
  Arch,
  Features,
  CoreCount,
  SIMDBitWidth,
  L1CacheSize,
  L2CacheSize,
  L3CacheSize,
  L4CacheSize
};

struct SystemInfoCLIOptions {

  llvm::cl::opt<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-")};

  llvm::cl::list<QuerySystemProperty> QueryProperty{
      "query", M::cl::desc("Available Queries:"),
      M::cl::values(
          clEnumValN(QuerySystemProperty::TargetTriple, "target-triple",
                     "Host target triple"),
          clEnumValN(QuerySystemProperty::OS, "os", "Host operating system"),
          clEnumValN(QuerySystemProperty::Arch, "arch",
                     "Host CPU architecture"),
          clEnumValN(QuerySystemProperty::Features, "features",
                     "Host CPU features printed as comma-separated values"),
          clEnumValN(QuerySystemProperty::CoreCount, "core-count",
                     "Host number of cores"),
          clEnumValN(QuerySystemProperty::SIMDBitWidth, "simd-bitwidth",
                     "Host SIMD bitwidth"),
          clEnumValN(QuerySystemProperty::L1CacheSize, "l1-cache-size",
                     "Host L1 DCache size"),
          clEnumValN(QuerySystemProperty::L2CacheSize, "l2-cache-size",
                     "Host L2 DCache size"),
          clEnumValN(QuerySystemProperty::L3CacheSize, "l3-cache-size",
                     "Host L3 DCache size"),
          clEnumValN(QuerySystemProperty::L4CacheSize, "l4-cache-size",
                     "Host L4 DCache size")),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};
};
} // namespace

static void dumpTargetTriple(raw_ostream &os) {
  os << sys::getDefaultTargetTriple();
}

static void dumpOS(raw_ostream &os) {
  os << llvm::Triple::getOSTypeName(
      llvm::Triple(sys::getDefaultTargetTriple()).getOS());
}

static void dumpArch(raw_ostream &os) { os << sys::getHostCPUName(); }

static void dumpFeatures(raw_ostream &os) {
  StringMap<bool> features;
  if (sys::getHostCPUFeatures(features)) {
    llvm::interleaveComma(
        llvm::make_filter_range(
            features, [](const auto &feature) { return feature.getValue(); }),
        os, [&](const auto &feature) { os << feature.getKey(); });
  }
}

static void dumpCoreCount(raw_ostream &os) { os << get_physical_cores(); }

static void dumpSIMDBitWidth(raw_ostream &os) { os << kPreferredSIMDBitWidth; }

static void dumpCacheSize(raw_ostream &os, size_t cacheLevel) {
  auto val = getHostCPUCacheSize(cacheLevel);
  if (val.isError()) {
    os << "Error: " << val.getError();
    return;
  }
  os << *val;
}

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

  if (cli.QueryProperty.empty()) {
    os << "target-triple: ";
    dumpTargetTriple(os);
    os << "\narch: ";
    dumpArch(os);
    os << "\nfeatures: ";
    dumpFeatures(os);
    os << "\ncore-count: ";
    dumpCoreCount(os);
    os << "\nsimd-bitwidth: ";
    dumpSIMDBitWidth(os);
    os << "\nl1-cache-size: ";
    dumpCacheSize(os, 1);
    os << "\nl2-cache-size: ";
    dumpCacheSize(os, 2);
    os << "\nl3-cache-size: ";
    dumpCacheSize(os, 3);
    os << "\nl4-cache-size: ";
    dumpCacheSize(os, 4);
    os << "\n";
  }

  for (auto query : cli.QueryProperty) {
    switch (query) {
    case QuerySystemProperty::TargetTriple:
      dumpTargetTriple(os);
      break;
    case QuerySystemProperty::OS:
      dumpOS(os);
      break;
    case QuerySystemProperty::Arch:
      dumpArch(os);
      break;
    case QuerySystemProperty::Features:
      dumpFeatures(os);
      break;
    case QuerySystemProperty::CoreCount:
      dumpCoreCount(os);
      break;
    case QuerySystemProperty::SIMDBitWidth:
      dumpSIMDBitWidth(os);
      break;
    case QuerySystemProperty::L1CacheSize:
      dumpCacheSize(os, 1);
      break;
    case QuerySystemProperty::L2CacheSize:
      dumpCacheSize(os, 2);
      break;
    case QuerySystemProperty::L3CacheSize:
      dumpCacheSize(os, 3);
      break;
    case QuerySystemProperty::L4CacheSize:
      dumpCacheSize(os, 4);
    }
    os << "\n";
  }

  os.flush();
  outputFile->keep();

  return EXIT_SUCCESS;
}
