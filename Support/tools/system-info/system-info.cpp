//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "Support/SIMD.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace llvm;

namespace {
enum class QueryProperty {
  TargetTriple,
  Arch,
  Features,
  CoreCount,
  SIMDBitWidth
};

struct SystemInfoCLIOptions {
  M::cl::opt<QueryProperty> QueryProperty{
      "query", M::cl::desc("Available Queries:"),
      M::cl::values(
          clEnumValN(QueryProperty::TargetTriple, "target-triple",
                     "Host target triple"),
          clEnumValN(QueryProperty::Arch, "arch", "Host CPU architecture"),
          clEnumValN(QueryProperty::Features, "features",
                     "Host CPU features printed as comma-separated values"),
          clEnumValN(QueryProperty::CoreCount, "core-count",
                     "Host number of cores"),
          clEnumValN(QueryProperty::SIMDBitWidth, "simd-width",
                     "Host SIMD bitwidth")),
      llvm::cl::Required};
};
} // namespace

int main(int argc, char **argv) {
  SystemInfoCLIOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular System Info Tool");

  raw_ostream &os(outs());
  switch (cli.QueryProperty) {
  case QueryProperty::TargetTriple:
    os << sys::getDefaultTargetTriple() << "\n";
    break;
  case QueryProperty::Arch:
    os << sys::getHostCPUName() << "\n";
    break;
  case QueryProperty::Features: {
    StringMap<bool> features;
    if (!sys::getHostCPUFeatures(features))
      break;
    llvm::interleaveComma(
        llvm::make_filter_range(
            features, [](const auto &feature) { return feature.getValue(); }),
        os, [&](const auto &feature) { os << feature.getKey(); });
    break;
  }
  case QueryProperty::CoreCount:
    os << sys::getHostNumPhysicalCores() << "\n";
    break;
  case QueryProperty::SIMDBitWidth:
    os << kPreferredSIMDBitWidth << "\n";
    break;
  }
  return EXIT_SUCCESS;
}
