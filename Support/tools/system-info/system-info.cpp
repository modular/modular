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
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace llvm;

namespace {
enum class QuerySystemProperty {
  TargetTriple,
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
  M::cl::opt<QuerySystemProperty> QueryProperty{
      "query", M::cl::desc("Available Queries:"),
      M::cl::values(
          clEnumValN(QuerySystemProperty::TargetTriple, "target-triple",
                     "Host target triple"),
          clEnumValN(QuerySystemProperty::Arch, "arch",
                     "Host CPU architecture"),
          clEnumValN(QuerySystemProperty::Features, "features",
                     "Host CPU features printed as comma-separated values"),
          clEnumValN(QuerySystemProperty::CoreCount, "core-count",
                     "Host number of cores"),
          clEnumValN(QuerySystemProperty::SIMDBitWidth, "simd-width",
                     "Host SIMD bitwidth"),
          clEnumValN(QuerySystemProperty::L1CacheSize, "l1-cache-size",
                     "Host L1 DCache size"),
          clEnumValN(QuerySystemProperty::L2CacheSize, "l2-cache-size",
                     "Host L2 DCache size"),
          clEnumValN(QuerySystemProperty::L3CacheSize, "l3-cache-size",
                     "Host L3 DCache size"),
          clEnumValN(QuerySystemProperty::L4CacheSize, "l4-cache-size",
                     "Host L4 DCache size")),
      llvm::cl::Required};
};
} // namespace

int main(int argc, char **argv) {
  SystemInfoCLIOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular System Info Tool");

  raw_ostream &os(outs());
  switch (cli.QueryProperty) {
  case QuerySystemProperty::TargetTriple:
    os << sys::getDefaultTargetTriple();
    break;
  case QuerySystemProperty::Arch:
    os << sys::getHostCPUName();
    break;
  case QuerySystemProperty::Features: {
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
      llvm::interleaveComma(
          llvm::make_filter_range(
              features, [](const auto &feature) { return feature.getValue(); }),
          os, [&](const auto &feature) { os << feature.getKey(); });
    }
    break;
  }
  case QuerySystemProperty::CoreCount:
    os << sys::getHostNumPhysicalCores();
    break;
  case QuerySystemProperty::SIMDBitWidth:
    os << kPreferredSIMDBitWidth;
    break;
  case QuerySystemProperty::L1CacheSize: {
    auto val = getHostCPUCacheSize(1);
    if (val.isError()) {
      os << "Error: " << val.getError();
      return EXIT_SUCCESS;
    }
    os << *val;
    break;
  }
  case QuerySystemProperty::L2CacheSize: {
    auto val = getHostCPUCacheSize(1);
    if (val.isError()) {
      os << "Error: " << val.getError();
      return EXIT_SUCCESS;
    }
    os << *val;
    break;
  }
  case QuerySystemProperty::L3CacheSize: {
    auto val = getHostCPUCacheSize(3);
    if (val.isError()) {
      os << "Error: " << val.getError();
      return EXIT_SUCCESS;
    }
    os << *val;
    break;
  }
  case QuerySystemProperty::L4CacheSize: {
    auto val = getHostCPUCacheSize(4);
    if (val.isError()) {
      os << "Error: " << val.getError();
      return EXIT_SUCCESS;
    }
    os << *val;
    break;
  }
  }
  os << "\n";
  return EXIT_SUCCESS;
}
