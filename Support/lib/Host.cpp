//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Host.h"
#include "llvm/ADT/Twine.h"

#include <fstream>
#include <ios>
#include <string>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif // __APPLE__

using namespace M;

ErrorOr<size_t> M::getHostCPUCacheSize(size_t cacheLevel) {
#if defined(__APPLE__)
  size_t result;
  size_t len = sizeof(result);
  switch (cacheLevel) {
  case 1:
    if (sysctlbyname("hw.l1dcachesize", &result, &len, nullptr, 0))
      return M::Error("unable to query the hw.l1dcachesize");
    return result;
  case 2:
    if (sysctlbyname("hw.l2cachesize", &result, &len, nullptr, 0))
      return M::Error("unable to query the hw.l2dcachesize");
    return result;
  default:
    return 0;
  }
#elif defined(__linux__)
  std::string path = "/sys/devices/system/cpu/cpu0/cache/index" +
                     std::to_string(cacheLevel - 1) + "/size";

  std::ifstream fs(path, std::ios::in);
  if (!fs)
    return M::Error(Twine("Unable to read '") + path + "' file");

  std::string contents;
  std::getline(fs, contents);

  size_t pos;
  size_t quantity = std::stoul(contents, &pos);
  std::string unit = contents.substr(pos);

  size_t multiplier = 1;
  switch (unit[0]) {
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
  return Error("unsupported platform");
#endif
}
