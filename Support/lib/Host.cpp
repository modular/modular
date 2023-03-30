//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Host.h"
#include "Support/SIMD.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <fstream>
#include <ios>
#include <string>

#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/task.h>
#include <sys/sysctl.h>
#endif // __APPLE__

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif // _MSC_VER

using namespace M;
using namespace llvm;

M::ErrorOr<size_t> M::getHostCPUCacheSize(size_t cacheLevel) {
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
                     std::to_string(cacheLevel) + "/size";

  std::ifstream fs(path, std::ios::in);

  // There are some times, for example inside of a Docker container on mac,
  // where the file is not there as expected. For now, hardcoding returning
  // 0 in that case, with maybe a bit of smarts in the future where the host
  // can specify what the cache is.
  if (!fs)
    return 0;

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
    break;
  default:
    break;
  }
  return quantity * multiplier;
#elif defined(_MSC_VER)

  // We can only get info for L1, L2 & L3 cache.
  if (cacheLevel >= 4)
    return 0;

  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> processorInfos;
  DWORD bufferLength = 0;

  DWORD returnCode =
      GetLogicalProcessorInformation(processorInfos.data(), &bufferLength);

  if (!returnCode) {

    // This is the only error where there is a reason for retry.
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {

      processorInfos.resize(bufferLength);

    } else {
      std::error_code ec = llvm::mapWindowsError(GetLastError());
      return Error(ec.message());
    }

    // Try once again with the new buffer length and pre allocated buffer.
    returnCode =
        GetLogicalProcessorInformation(processorInfos.data(), &bufferLength);

    // We can recheck for insufficient buffer length and keep on doing this
    // but it should be pretty rare to fail twice with that reason. So we will
    // bail out.
    if (!returnCode) {
      std::error_code ec = llvm::mapWindowsError(GetLastError());
      return Error(ec.message());
    }
  }

  for (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION &processorInfo :
       processorInfos) {

    if (processorInfo.Relationship == RelationCache) {
      const CACHE_DESCRIPTOR &cache = processorInfo.Cache;
      if (cache.Level == cacheLevel)
        return cache.Size;
    }
  }

  return Error("Information not available");
#else
  return Error("unsupported platform");
#endif
}

static void dumpFeatures(raw_ostream &os,
                         const std::vector<std::string> &features) {
  llvm::interleaveComma(features, os);
}

void M::HostMachineInfo::print(llvm::raw_ostream &os) const {
  os << "target-triple: ";
  os << triple;
  os << "\nos: ";
  os << osName;
  os << "\narch: ";
  os << cpuArch;
  os << "\nfeatures: ";
  dumpFeatures(os, cpuFeatures);
  os << "\ncore-count: ";
  os << numPhysicalCores;
  os << "\nsimd-bitwidth: ";
  os << simdBitWidth;
  os << "\nl1-cache-size: ";
  os << l1CacheSize;
  os << "\nl2-cache-size: ";
  os << l2CacheSize;
  os << "\nl3-cache-size: ";
  os << l3CacheSize;
  os << "\nl4-cache-size: ";
  os << l4CacheSize;
  os << "\n";
}

void M::HostMachineInfo::print(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("target-triple", triple);
  json.attribute("os", osName);
  json.attribute("arch", cpuArch);
  json.attribute("features", cpuFeatures);
  json.attribute("core-count", numPhysicalCores);
  json.attribute("simd-bitwidth", simdBitWidth);
  json.attribute("l1-cache-size", l1CacheSize);
  json.attribute("l2-cache-size", l2CacheSize);
  json.attribute("l3-cache-size", l3CacheSize);
  json.attribute("l4-cache-size", l4CacheSize);
  json.objectEnd();
}

void HostMachineInfo::print(HostProperty property,
                            llvm::raw_ostream &os) const {
  switch (property) {
  case HostProperty::TargetTriple:
    os << triple;
    break;
  case HostProperty::OS:
    os << osName;
    break;
  case HostProperty::Arch:
    os << cpuArch;
    break;
  case HostProperty::Features:
    dumpFeatures(os, cpuFeatures);
    break;
  case HostProperty::CoreCount:
    os << numPhysicalCores;
    break;
  case HostProperty::SIMDBitWidth:
    os << simdBitWidth;
    break;
  case HostProperty::L1CacheSize:
    os << l1CacheSize;
    break;
  case HostProperty::L2CacheSize:
    os << l2CacheSize;
    break;
  case HostProperty::L3CacheSize:
    os << l3CacheSize;
    break;
  case HostProperty::L4CacheSize:
    os << l4CacheSize;
  }
  os << "\n";
}

M::ErrorOr<HostMachineInfo> M::getHostMachineInfo() {
  HostMachineInfo machineInfo;

  machineInfo.triple = sys::getDefaultTargetTriple();
  machineInfo.osName =
      Triple::getOSTypeName(Triple(machineInfo.triple).getOS());
  machineInfo.cpuArch = sys::getHostCPUName();

  StringMap<bool> features;
  sys::getHostCPUFeatures(features);

  for (const auto &feature : features)
    if (feature.getValue())
      machineInfo.cpuFeatures.push_back(feature.getKey().str());
  llvm::sort(machineInfo.cpuFeatures);

  machineInfo.numPhysicalCores = get_physical_cores();
  machineInfo.simdBitWidth = kPreferredSIMDBitWidth;

  UNWRAP_ERROR(l1CacheSize, getHostCPUCacheSize(1));
  machineInfo.l1CacheSize = l1CacheSize;

  UNWRAP_ERROR(l2CacheSize, getHostCPUCacheSize(2));
  machineInfo.l2CacheSize = l2CacheSize;

  UNWRAP_ERROR(l3CacheSize, getHostCPUCacheSize(3));
  machineInfo.l3CacheSize = l3CacheSize;

  UNWRAP_ERROR(l4CacheSize, getHostCPUCacheSize(4));
  machineInfo.l4CacheSize = l4CacheSize;
  return std::move(machineInfo);
}

size_t M::getProcessPhysicalMemUsage() {
#if defined(__linux__)
  // On linux we'll use the (approximate) process resident number of pages.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/self/statm");
  if (std::error_code ec = errOrBuf.getError())
    return 0;
  StringRef buffer = (*errOrBuf)->getBuffer();
  // Buffer will be "size resident shared text lib data dt", all as num pages.
  SmallVector<StringRef, 7> strs;
  buffer.split(strs, " ");
  if (strs.size() != 7)
    return 0;
  size_t value;
  if (strs[1].getAsInteger(10, value))
    return 0;
  // Convert from pages to bytes.
  return value * llvm::sys::Process::getPageSizeEstimate();
#elif defined(__APPLE__)
  struct task_basic_info info;
  unsigned count = TASK_BASIC_INFO_COUNT;
  kern_return_t result =
      task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count);
  if (result != KERN_SUCCESS)
    return 0;
  return info.resident_size;
#else
  return 0;
#endif
}
