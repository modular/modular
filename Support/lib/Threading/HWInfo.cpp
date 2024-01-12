//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <filesystem>

#include "Support/ErrorOr.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif // _MSC_VER

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif // __APPLE__

#define DEBUG_TYPE "threading-hw-info"

using namespace M;

//===----------------------------------------------------------------------===//
// CPUSystemInfo
//===----------------------------------------------------------------------===//

namespace {

#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
std::unique_ptr<llvm::MemoryBuffer> fileBuffer(StringRef path) {
  auto errOrBuf = llvm::MemoryBuffer::getFileAsStream(path);
  if (std::error_code ec = errOrBuf.getError()) {
    LLVM_DEBUG(llvm::dbgs()
               << "getLinuxCPULimits: Could not open " << path << "\n");
    return nullptr;
  }
  return std::move(errOrBuf.get());
}
#endif

} // namespace

#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
ErrorOr<std::string>
Detail::parseV1CPUCgroupFile(const llvm::MemoryBuffer &buf) {
  std::string cgroup;
  SmallVector<StringRef> strs;
  buf.getBuffer().split(strs, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef line : strs) {
    SmallVector<StringRef> frags;
    line.split(frags, ":");
    StringRef sys = frags[1].trim();
    StringRef grp = frags[2].trim();
    // Cpuset settings are already reflected in the CPU affinity mask, but
    // further changes must be monitored to trigger CPUSystemInfo regeneration.
    if (sys == "cpu,cpuacct") {
      cgroup = grp;
      break;
    }
  }
  if (cgroup.empty())
    return Error("could not parse v1 CPU cgroup");
  return cgroup;
}

ErrorOr<Detail::CPULimits>
Detail::parseV1CPULimits(const llvm::MemoryBuffer &quotaBuf,
                         const llvm::MemoryBuffer &periodBuf) {
  Detail::CPULimits limits;
  if (quotaBuf.getBuffer().trim().getAsInteger(10, limits.quota_us))
    return Error("can't parse CPU quota as an int");
  if (periodBuf.getBuffer().trim().getAsInteger(10, limits.period_us))
    return Error("can't parse CPU period as an int");
  return limits;
}

ErrorOr<std::string>
Detail::parseV2CPUCgroupFile(const llvm::MemoryBuffer &buf,
                             const std::function<bool(StringRef)> &exists) {
  StringRef cgroup;
  SmallVector<StringRef> strs;
  buf.getBuffer().split(strs, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef line : strs) {
    if (line.consume_front("0::")) {
      cgroup = line;
      break;
    }
  }
  if (cgroup.empty())
    return Error("could not parse v2 CPU cgroup");

  StringRef curr = cgroup.trim();
  const size_t n = curr.count('/');
  // Check each level of the v2 filesystem to find the right file.
  for (size_t i = 0; i <= n; ++i) {
    const auto path = ("/sys/fs/cgroup/" + curr + "/cpu.max").str();
    if (!exists(path)) {
      auto pos = curr.rfind('/');
      if (pos != StringRef::npos) {
        curr = curr.slice(0, pos);
        continue;
      } else
        return Error("could not resolve CPU max file");
    }
    break;
  }
  return curr.str();
}

ErrorOr<Detail::CPULimits>
Detail::parseV2CPULimits(const llvm::MemoryBuffer &maxBuf) {
  Detail::CPULimits limits;
  SmallVector<StringRef> strs;
  maxBuf.getBuffer().split(strs, " ", /*MaxSplit=*/2, /*KeepEmpty=*/false);
  if (strs.empty())
    return Error("can't parse empty CPU max and period");
  if (strs[0] != "max" && strs[0].trim().getAsInteger(10, limits.quota_us))
    return Error("can't parse CPU max as an int");
  if (strs.size() == 2 && strs[1].trim().getAsInteger(10, limits.period_us))
    return Error("can't parse CPU period as an int");
  return limits;
}

/// Looks up various cgroup CPU limits for the current process.
Detail::CPULimits Detail::getLinuxCPULimits() {
  // Detect cgroup version.
  auto errOrControllers =
      llvm::MemoryBuffer::getFileAsStream("/sys/fs/cgroup/cgroup.controllers");
  bool isV1 = true;
  if (errOrControllers.getError().value() == 0) {
    SmallVector<StringRef> strs;
    (*errOrControllers)
        ->getBuffer()
        .split(strs, " ", /*MaxSplit=*/2, /*KeepEmpty=*/false);
    for (const auto str : strs) {
      // When using the hybrid layout, the cpu controller might not be mounted
      // as v2; in which case we fallback to locating the v1 filesystem.
      if (str == "cpu") {
        isV1 = false;
        break;
      }
    }
  }

  // Read and parse /proc/self/cgroup
  auto cgroupBuf = fileBuffer("/proc/self/cgroup");
  if (!cgroupBuf)
    return {};
  Detail::CPULimits limits;
  if (isV1) {
    const auto errOrCgroup = parseV1CPUCgroupFile(*cgroupBuf);
    if (errOrCgroup.isError()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getLinuxCPULimits: " << errOrCgroup.getError() << "\n");
      return {};
    }
    const std::string &cgroup = *errOrCgroup;
    // Quota and period files found in membership-specific filesystems.
    const auto quotaBuf =
        fileBuffer("/sys/fs/cgroup/cpu/" + cgroup + "/cpu.cfs_quota_us");
    if (!quotaBuf)
      return {};
    const auto periodBuf =
        fileBuffer("/sys/fs/cgroup/cpu/" + cgroup + "/cpu.cfs_period_us");
    if (!periodBuf)
      return {};
    const auto errOrLimits = parseV1CPULimits(*quotaBuf, *periodBuf);
    if (errOrLimits.isError()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getLinuxCPULimits: " << errOrLimits.getError() << "\n");
      return {};
    }
    limits = *errOrLimits;
  } else {
    const auto errOrCgroup =
        parseV2CPUCgroupFile(*cgroupBuf, [](StringRef path) {
          return std::filesystem::exists(path.str());
        });
    if (errOrCgroup.isError()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getLinuxCPULimits: " << errOrCgroup.getError() << "\n");
      return {};
    }
    const std::string &cgroup = *errOrCgroup;
    // Lives at some level of the cgroup v2 unified filesystem as a combined
    // file.
    const auto maxBuf = fileBuffer("/sys/fs/cgroup/" + cgroup + "/cpu.max");
    if (!maxBuf)
      return {};
    const auto errOrLimits = parseV2CPULimits(*maxBuf);
    if (errOrLimits.isError()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getLinuxCPULimits: " << errOrLimits.getError() << "\n");
      return {};
    }
    limits = *errOrLimits;
  }
  // The bounds and explanations for these values can be found at:
  // https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.rst
  if (limits.quota_us != -1 && limits.quota_us < 1000) {
    LLVM_DEBUG(llvm::dbgs()
               << "getLinuxCPULimits: Expected cpu quota above 1ms\n");
    return {};
  }
  if (limits.period_us < 1000 || limits.period_us > 1000000) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "getLinuxCPULimits: Expected cpu period between 1ms and 1s\n");
    return {};
  }
  return limits;
}

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

#if defined(_MSC_VER)
ErrorOr<size_t> Detail::getNumPhysicalCoresWindows() {
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};
  DWORD bufferSize = 0;
  DWORD result = GetLogicalProcessorInformation(buffer.data(), &bufferSize);
  if (result == FALSE) {
    DWORD lastError = GetLastError();
    if (lastError == ERROR_INSUFFICIENT_BUFFER) {
      DWORD numInfo = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      buffer.resize(numInfo);
    } else {
      std::error_code ec = llvm::mapWindowsError(lastError);
      return M::Error(ec.message());
    }
  }
  result = GetLogicalProcessorInformation(buffer.data(), &bufferSize);
  if (result == FALSE) {
    std::error_code ec = llvm::mapWindowsError(GetLastError());
    return M::Error(ec.message());
  }

  DWORD processorCoreCount = 0;
  for (const auto &processorInfo : buffer) {
    if (processorInfo.Relationship == RelationProcessorCore)
      ++processorCoreCount;
  }
  return processorCoreCount;
}
#endif

ErrorOr<CPUSystemInfo> CPUSystemInfo::get() {
#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
  return Detail::getLinuxX86CPUSystemInfo();
#endif
  return Error("CPUSystemInfo is not supported by this build");
}

size_t M::getNumPhysicalCores() {
  auto threadStrat = llvm::hardware_concurrency();
  threadStrat.UseHyperThreads = false;
  return threadStrat.compute_thread_count();
}

size_t M::getNumLogicalCores() {
  return llvm::hardware_concurrency().compute_thread_count();
}

/// Returns the number of physical performance cores across all CPU sockets. If
/// not known, will return the total number of physical cores.
/// TODO: add implementations for Windows and Linux
size_t M::getNumPerformanceCores() {
#if defined(__APPLE__)
  // Attempt to read the sysctl "hw.perflevel0.physicalcpu", which contains the
  // number of local performance cores. This is described here [1]. This can be
  // used to create a runtime if it is available to avoid contention and lagging
  // on the efficiency cores. We rely on the operating system to keep busy
  // threads running on the performance cores rather than any explicit affinity.
  //
  // Per sysctl(2), this type is expected to be int32_t.
  //
  // [1]
  // https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_system_capabilities
  int32_t pcores;
  size_t len = sizeof(pcores);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &len, nullptr, 0) ==
          0 &&
      pcores > 0)
    return static_cast<size_t>(pcores);
  return M::getNumPhysicalCores();
#endif
  return M::getNumPhysicalCores();
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
          if (cpuIDs.size() >= numThreads) {
            // Found enough.
            return cpuIDs;
          }
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
