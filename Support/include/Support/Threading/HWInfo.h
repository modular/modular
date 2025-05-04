//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_THREADING_HWINFO_H
#define SUPPORT_THREADING_HWINFO_H

#include "Support/ErrorOr.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include <vector>

#if defined(__linux__) && (defined(__i386__) || defined(__x86_64__))
#define HAVE_LINUX_X86_SYSTEM_INFO
#endif

#if defined(__linux__)
#define HAVE_LINUX_SET_AFFINITY
#endif

namespace M {

/// The distinguished CPU ID denoting 'no affinity to be set'.
constexpr size_t kNoAffinity = ~0;

//===----------------------------------------------------------------------===//
// CPUSystemInfo
//===----------------------------------------------------------------------===//

/// Describes the sockets, physical cores, and virtual cores in a CPU system
/// when supported by the host os. However does not capture sharing of caches
/// and plethora of other details. See https://www.open-mpi.org/projects/hwloc/
struct CPUSystemInfo {
  /// A 'virtual' core, generally without dedicated cache or ALU resources.
  /// Systems with hyperthreading can have multiple virtual cores per
  /// 'physical' core.
  struct VirtualCore {
    size_t cpuID;

    VirtualCore(size_t cpuID) : cpuID(cpuID) {}
  };

  /// A 'physical' core, generally with its own dedicated cache levels.
  struct PhysicalCore {
    llvm::SmallVector<VirtualCore, 2> virtualCores;
  };

  /// A 'socket', generally with its own NUMA memory area and dedicated cache
  /// levels.
  struct Socket {
    llvm::SmallVector<PhysicalCore, 16> physicalCores;
  };

  llvm::SmallVector<Socket, 1> sockets;

  /// Returns system info if it can be determined. Fidelity with respect to
  /// actual hardware may vary depending on host OS. Returns an error if system
  /// info cannot be determined.
  static ErrorOr<CPUSystemInfo> get();

  /// Returns numThreads cpuIDs drawn from this system info, following these
  /// heuristics (from strongest to weakest):
  ///  - prefer threads on distinct virtual cores
  ///  - prefer virtual cores on distinct physical cores
  ///  - prefer physical cores on the same socket
  /// If numThreads exceeds the number of virtual cores in the system then
  /// cpu IDs will be repeated in the result.
  std::vector<size_t> getPreferredCpuIDs(size_t numThreads) const;

  void print(raw_ostream &os) const;
};

inline raw_ostream &operator<<(raw_ostream &os, const CPUSystemInfo &info) {
  info.print(os);
  return os;
}

/// Returns the number of physical cores across all CPU sockets
size_t getNumPhysicalCores();

/// Returns the number of hardware threads, including hyperthreads across all
/// CPU sockets
size_t getNumLogicalCores();

/// Returns the number of physical performance cores across all CPU sockets. If
/// not known, will return the total number of physical cores.
size_t getNumPerformanceCores();

/// Returns the set of local MAC addresses.
std::vector<std::string> localMACs();

/// Describes CPU limits in an OS-agnostic way.
struct CPULimits {
  /// Unfortunately, millicores are a canonical way of representing the limit,
  /// even though it has far more subtlely than this.
  std::optional<size_t> millicores;

  /// Returns local limits, if available.
  static ErrorOr<CPULimits> get();
};

//===----------------------------------------------------------------------===//
// OS and architecture-specific utilities, visible for testing only
//===----------------------------------------------------------------------===//

namespace Detail {
#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
/// Specifies CPU quota per period of CPU time allotted by the Linux CFS.
struct linuxCPULimits {
  int quota_us = -1;
  int period_us = 100000;
};

/// Returns the cgroup v1 CPU membership from |buf|.
ErrorOr<std::string> parseV1CPUCgroupFile(const llvm::MemoryBuffer &buf);
ErrorOr<linuxCPULimits> parseV1CPULimits(const llvm::MemoryBuffer &quotaBuf,
                                         const llvm::MemoryBuffer &periodBuf);

/// Returns the effective cgroup v2 CPU membership from |buf|. This is
/// determined by searching /sys/fs/cgroup/ until a cpu.max file is found.
ErrorOr<std::string>
parseV2CPUCgroupFile(const llvm::MemoryBuffer &buf,
                     const std::function<bool(StringRef)> &exists);
ErrorOr<linuxCPULimits> parseV2CPULimits(const llvm::MemoryBuffer &maxBuf);

linuxCPULimits getLinuxCPULimits();

ErrorOr<CPUSystemInfo>
getLinuxX86CPUSystemInfoImpl(const cpu_set_t &availableCpus,
                             std::unique_ptr<llvm::MemoryBuffer> buf);
ErrorOr<CPUSystemInfo> getLinuxX86CPUSystemInfo();
#endif // defined(HAVE_LINUX_X86_SYSTEM_INFO)

#if defined(HAVE_LINUX_SET_AFFINITY)
ErrorOrSuccess setThreadAffinityLinux(size_t cpuID);
ErrorOrSuccess runWithThreadAffinityLinux(size_t cpuID,
                                          llvm::function_ref<void()> &workFn);
#endif // defined(HAVE_LINUX_SET_AFFINITY)
} // namespace Detail
} // namespace M

#endif // SUPPORT_THREADING_HWINFO_H
