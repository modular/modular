//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Threading/HWInfo.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif // _MSC_VER

#define DEBUG_TYPE "threading-hw-info"

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

static M::ErrorOr<size_t> getNumPhysicalCoresImpl() {
#ifdef _MSC_VER
  return Detail::getNumPhysicalCoresWindows();
#endif
  return llvm::get_physical_cores();
}

M::ErrorOr<size_t> M::getNumPhysicalCores() {
  static ErrorOr<size_t> numPhysicalCoresOr = getNumPhysicalCoresImpl();
  if (numPhysicalCoresOr.isError())
    return M::Error(numPhysicalCoresOr.getError());
  return *numPhysicalCoresOr;
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
