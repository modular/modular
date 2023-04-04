//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Host.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/SIMD.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
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

#define DEBUG_TYPE "host"

using namespace M;

//===----------------------------------------------------------------------===//
// CPUSystemInfo
//===----------------------------------------------------------------------===//

#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
/// Given contents of /proc/cpuinfo and the thread affinity mask describing
/// which CPUs should be considered available, returns CPUSystemInfo.
ErrorOr<CPUSystemInfo> M::Detail::getLinuxX86CPUSystemInfoImpl(
    const cpu_set_t &availableCpus, std::unique_ptr<llvm::MemoryBuffer> buf) {
  // Describes a virtual core.
  struct Entry {
    // Dense socket index.
    int physicalId = -1;
    // Non-dense index of physical core in socket.
    int coreId = -1;
    // Dense system-wide processor identifier, what we'll refer to as cpuID.
    int processor = -1;

    bool operator<(const Entry &that) const {
      return std::tie(physicalId, coreId, processor) <
             std::tie(that.physicalId, that.coreId, that.processor);
    }
  };

  SmallVector<StringRef> strs;
  buf->getBuffer().split(strs, "\n", /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);

  // Collect entries
  SmallVector<Entry> entries;
  Entry currEntry;
  for (StringRef line : strs) {
    std::pair<StringRef, StringRef> Data = line.split(':');
    StringRef name = Data.first.trim();
    StringRef val = Data.second.trim();
    // These fields are available if the kernel is configured with CONFIG_SMP.
    if (name == "processor") {
      if (val.getAsInteger(10, currEntry.processor))
        return Error("ill-formed /proc/cpuinfo output");
    } else if (name == "physical id") {
      if (val.getAsInteger(10, currEntry.physicalId))
        return Error("ill-formed /proc/cpuinfo output");
    } else if (name == "core id") {
      if (val.getAsInteger(10, currEntry.coreId))
        return Error("ill-formed /proc/cpuinfo output");
      if (currEntry.physicalId < 0 || currEntry.coreId < 0 ||
          currEntry.processor < 0)
        return Error("ill-formed /proc/cpuinfo output");
      if (CPU_ISSET(currEntry.processor, &availableCpus)) {
        // Only include if processor already in affinity set.
        entries.push_back(currEntry);
        currEntry = Entry();
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "getLinuxX86CPUSystemInfo: Ignoring processor "
                   << std::to_string(currEntry.processor)
                   << " since excluded from main thread's affinity set\n");
      }
    }
  }
  llvm::sort(entries);

  // Build system info.
  CPUSystemInfo systemInfo;
  Entry prevEntry;
  for (auto entry : entries) {
    if (entry.physicalId != prevEntry.physicalId)
      systemInfo.sockets.emplace_back();
    CPUSystemInfo::Socket &socket = systemInfo.sockets.back();
    if (entry.physicalId != prevEntry.physicalId ||
        entry.coreId != prevEntry.coreId)
      socket.physicalCores.emplace_back();
    CPUSystemInfo::PhysicalCore &physicalCore = socket.physicalCores.back();
    physicalCore.virtualCores.emplace_back(entry.processor);
    prevEntry = entry;
  }
  return systemInfo;
}

/// On X86 Linux systems /proc/cpuinfo allows us to distinguish
/// virtual cores, physical cores and sockets.
///
/// Adapted from third-party/llvm-project/llvm/lib/Support/Unix/Threading.inc.
ErrorOr<CPUSystemInfo> M::Detail::getLinuxX86CPUSystemInfo() {
  // Only consider cpuIDs which are already in the affinity set of the
  // calling thread. This way we'll respect any restrictions already set.
  cpu_set_t callersAffinity;
  if (int rc = sched_getaffinity(0, sizeof(callersAffinity), &callersAffinity))
    return Error("can't retrieve schedule affinity for main thread: " +
                 std::to_string(rc));

  // Read /proc/cpuinfo as a stream (until EOF reached). It cannot be
  // mmapped because it appears to have 0 size.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> errOrBuf =
      llvm::MemoryBuffer::getFileAsStream("/proc/cpuinfo");
  if (std::error_code ec = errOrBuf.getError())
    return Error("can't open /proc/cpuinfo: " + ec.message());

  return getLinuxX86CPUSystemInfoImpl(callersAffinity, std::move(*errOrBuf));
}
#endif

ErrorOr<CPUSystemInfo> CPUSystemInfo::get() {
#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
  return Detail::getLinuxX86CPUSystemInfo();
#endif
  return Error("CPUSystemInfo is not supported by this build");
}

void CPUSystemInfo::print(raw_ostream &os) const {
  os << "CPUSystemInfo(";
  llvm::interleave(
      sockets, os,
      [&](const Socket &s) {
        os << "Socket(";
        llvm::interleave(
            s.physicalCores, os,
            [&](const PhysicalCore &pc) {
              os << "{";
              llvm::interleave(
                  pc.virtualCores, os,
                  [&](const VirtualCore &vc) { os << vc.cpuID; }, ", ");
              os << "}";
            },
            ", ");
        os << ")";
      },
      ", ");
  os << ")";
}

std::vector<size_t> CPUSystemInfo::getPreferredCpuIDs(size_t numThreads) const {
  std::vector<size_t> cpuIDs;
  size_t virtualCoreIndex = 0;
  while (true) {
    size_t origNumCpuIDs = cpuIDs.size();
    for (const auto &socket : sockets) {
      for (const auto &physicalCore : socket.physicalCores) {
        if (virtualCoreIndex < physicalCore.virtualCores.size()) {
          cpuIDs.emplace_back(
              physicalCore.virtualCores[virtualCoreIndex].cpuID);
          if (cpuIDs.size() >= numThreads)
            // Found enough.
            return cpuIDs;
        }
      }
    }
    if (cpuIDs.size() == origNumCpuIDs)
      // No more virtual cores to add. We'll need to start re-using them.
      virtualCoreIndex = 0;
    else
      // Need to use additional virtual cores on the same physical core (if
      // any).
      ++virtualCoreIndex;
  }
}

//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

#if defined(HAVE_LINUX_SET_AFFINITY)
ErrorOrSuccess M::Detail::setCallersThreadAffinityLinux(size_t cpuID) {
  assert(cpuID < CPU_SETSIZE);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpuID, &cpuset);
  int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (rc != 0)
    return Error("unable to set thread CPU affinity: " + std::to_string(rc));
  return success();
}

ErrorOrSuccess
M::Detail::runWithThreadAffinityLinux(size_t cpuID,
                                      llvm::unique_function<void()> &workFn) {
  assert(cpuID < CPU_SETSIZE);
  cpu_set_t origset;
  int rc = sched_getaffinity(0, sizeof(origset), &origset);
  if (rc != 0)
    return Error("unable to get thread CPU affinity: " + std::to_string(rc));
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpuID, &cpuset);
  rc = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (rc != 0)
    return Error("unable to set thread CPU affinity: " + std::to_string(rc));
  // We're -fno-exceptions so no need for exception handling here.
  workFn();
  rc = sched_setaffinity(0, sizeof(cpuset), &origset);
  if (rc != 0) {
    // We've run the workFn, so can't report failure.
    LLVM_DEBUG(llvm::dbgs() << "runWithThreadAffinityLinux: unable to restore "
                               "thread CPU affinity: " +
                                   std::to_string(rc)
                            << "\n");
  }
  return success();
}
#endif

bool M::haveThreadAffinity() {
#if defined(HAVE_LINUX_SET_AFFINITY)
  return true;
#endif
  return false;
}

ErrorOrSuccess M::setThreadAffinity(size_t cpuID) {
#if defined(HAVE_LINUX_SET_AFFINITY)
  return Detail::setCallersThreadAffinityLinux(cpuID);
#endif
  return Error("setThreadAffinity is not supported by this build");
}

ErrorOrSuccess M::runWithThreadAffinity(size_t cpuID,
                                        llvm::unique_function<void()> workFn) {
#if defined(HAVE_LINUX_SET_AFFINITY)
  return Detail::runWithThreadAffinityLinux(cpuID, workFn);
#endif
  return Error("runWithThreadAffinity is not supported by this build");
}

//===----------------------------------------------------------------------===//
// Cache sizes
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// HostMachineInfo
//===----------------------------------------------------------------------===//

static void dumpFeatures(raw_ostream &os,
                         const std::vector<std::string> &features) {
  llvm::interleaveComma(features, os);
}

static void
dumpAffinities(raw_ostream &os,
               const std::optional<std::vector<size_t>> &affinities) {
  if (affinities) {
    os << "[";
    llvm::interleave(*affinities, os, ", ");
    os << "]";
  } else {
    os << "none";
  }
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
  os << "\npreferred-mem-alignment: ";
  os << preferredMemoryAlignment;
  os << "\nl1-cache-size: ";
  os << l1CacheSize;
  os << "\nl2-cache-size: ";
  os << l2CacheSize;
  os << "\nl3-cache-size: ";
  os << l3CacheSize;
  os << "\nl4-cache-size: ";
  os << l4CacheSize;
  os << "\naffinities: ";
  dumpAffinities(os, affinities);
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
  json.attribute("preferred-mem-alignment", preferredMemoryAlignment);
  json.attribute("l1-cache-size", l1CacheSize);
  json.attribute("l2-cache-size", l2CacheSize);
  json.attribute("l3-cache-size", l3CacheSize);
  json.attribute("l4-cache-size", l4CacheSize);
  if (affinities) {
    json.attribute("affinites", *affinities);
  }
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
  case HostProperty::PreferredMemoryAlignment:
    os << preferredMemoryAlignment;
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
    break;
  case HostProperty::Affinities:
    dumpAffinities(os, affinities);
    break;
  }
  os << "\n";
}

void HostMachineInfo::printStaticInfo(raw_ostream &os) const {
  print(HostProperty::TargetTriple, os);
  print(HostProperty::OS, os);
  print(HostProperty::Arch, os);
  print(HostProperty::Features, os);
  print(HostProperty::SIMDBitWidth, os);
  print(HostProperty::L1CacheSize, os);
  print(HostProperty::L2CacheSize, os);
  print(HostProperty::L3CacheSize, os);
  print(HostProperty::L4CacheSize, os);
}

M::ErrorOr<HostMachineInfo> M::getHostMachineInfo() {
  HostMachineInfo machineInfo;

  machineInfo.triple = llvm::sys::getDefaultTargetTriple();
  machineInfo.osName =
      llvm::Triple::getOSTypeName(llvm::Triple(machineInfo.triple).getOS());
  machineInfo.cpuArch = llvm::sys::getHostCPUName();

  llvm::StringMap<bool> features;
  llvm::sys::getHostCPUFeatures(features);

  for (const auto &feature : features)
    if (feature.getValue())
      machineInfo.cpuFeatures.push_back(feature.getKey().str());
  llvm::sort(machineInfo.cpuFeatures);

  machineInfo.numPhysicalCores = llvm::get_physical_cores();
  machineInfo.simdBitWidth = kPreferredSIMDBitWidth;
  machineInfo.preferredMemoryAlignment = kPreferredMemoryAlignment;

  UNWRAP_ERROR(l1CacheSize, getHostCPUCacheSize(1));
  machineInfo.l1CacheSize = l1CacheSize;

  UNWRAP_ERROR(l2CacheSize, getHostCPUCacheSize(2));
  machineInfo.l2CacheSize = l2CacheSize;

  UNWRAP_ERROR(l3CacheSize, getHostCPUCacheSize(3));
  machineInfo.l3CacheSize = l3CacheSize;

  UNWRAP_ERROR(l4CacheSize, getHostCPUCacheSize(4));
  machineInfo.l4CacheSize = l4CacheSize;

  if (haveThreadAffinity()) {
    ErrorOr<CPUSystemInfo> errOrSysInfo = CPUSystemInfo::get();
    if (!errOrSysInfo.isError()) {
      machineInfo.affinities =
          errOrSysInfo->getPreferredCpuIDs(machineInfo.numPhysicalCores);
    }
    // else: ignore error, leave field empty to denote affinities are not avail.
  }

  return std::move(machineInfo);
}

//===----------------------------------------------------------------------===//
// Memory usage
//===----------------------------------------------------------------------===//

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
