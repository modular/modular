//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/SIMD.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <errno.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif // __APPLE__

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

template <int CacheLevel>
static M::ErrorOr<size_t> cacheSize() {
#if defined(__APPLE__)
  size_t result;
  size_t len = sizeof(result);
  switch (CacheLevel) {
  case 1:
    if (sysctlbyname("hw.l1dcachesize", &result, &len, nullptr, 0))
      return M::Error("unable to query the hw.l1dcachesize");
    return result;
  case 2:
    if (sysctlbyname("hw.l2cachesize", &result, &len, nullptr, 0))
      return M::Error("unable to query the hw.l3dcachesize");
    return result;
  default:
    return 0;
  }
#elif defined(__linux__)
  std::string path = "/sys/devices/system/cpu/cpu0/cache/index" +
                     std::to_string(CacheLevel - 1) + "/size";
  auto file = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (std::error_code error = file.getError())
    return M::Error("unable to open '" + path + "': " + error.message());

  std::string buffer = (*file)->getBuffer().str();
  const char *line = buffer.c_str();

  char *suffix;
  size_t quantity = std::strtoull(line, &suffix, 0);
  if (errno)
    return M::Error(Twine("Unable to parse the ") + buffer + " from the '" +
                    path + "' file: " + strerror(errno));

  size_t multiplier = 1;
  switch (*suffix) {
  case 'g':
    [[fallthrough]];
  case 'G':
    multiplier = 1024;
    [[fallthrough]];
  case 'm':
    [[fallthrough]];
  case 'M':
    multiplier *= 1024;
    [[fallthrough]];
  case 'k':
    [[fallthrough]];
  case 'K':
    multiplier *= 1024;
  default:
    break;
  }
  return quantity * multiplier;
#elif defined(_WIN32)
  // TODO: Figure out later, but this needs to use the
  // GetLogicalProcessorInformation API.
  return 0;
#else
  llvm_unreachable("unsupported platform");
#endif
}

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
    if (auto val = cacheSize<1>(); succeeded(val))
      os << *val;
    break;
  }
  case QuerySystemProperty::L2CacheSize: {
    if (auto val = cacheSize<2>(); succeeded(val))
      os << *val;
    break;
  }
  case QuerySystemProperty::L3CacheSize: {
    if (auto val = cacheSize<3>(); succeeded(val))
      os << *val;
    break;
  }
  case QuerySystemProperty::L4CacheSize: {
    if (auto val = cacheSize<4>(); succeeded(val))
      os << *val;
    break;
  }
  }
  os << "\n";
  return EXIT_SUCCESS;
}
