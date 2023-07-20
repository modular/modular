//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Utilities for interrogating and interacting with the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HOST_H
#define SUPPORT_HOST_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace llvm {
// Forward declare.
class MemoryBuffer;
} // namespace llvm

namespace llvm::json {
// Forward declare.
class OStream;
} // namespace llvm::json

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#define HOST_IS_APPLE_SILICON_PROCESSOR
#endif

#if defined(__linux__) && (defined(__i386__) || defined(__x86_64__))
#define HAVE_LINUX_X86_SYSTEM_INFO
#endif

#if defined(__linux__)
#define HAVE_LINUX_SET_AFFINITY
#endif

namespace M {

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
    SmallVector<VirtualCore, 2> virtualCores;
  };

  /// A 'socket', generally with its own NUMA memory area and dedicated cache
  /// levels.
  struct Socket {
    SmallVector<PhysicalCore, 16> physicalCores;
  };

  SmallVector<Socket, 1> sockets;

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

//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

/// Returns true if thread affinity is available on this target.
bool haveThreadAffinity();

/// Attempts to sets the caller's thread affinity to the given CPU id. Returns
/// error if affinity is not supported on this target or the operation fails.
ErrorOrSuccess setThreadAffinity(size_t cpuID);

/// Attempts to runs workFn with caller's thread affinity set to the given CPU
/// id. Returns error if thread affinity is not supported on this target
/// or the operation fails.
ErrorOrSuccess runWithThreadAffinity(size_t cpuID,
                                     llvm::function_ref<void()> workFn);

//===----------------------------------------------------------------------===//
// CPU Model Info
//===----------------------------------------------------------------------===//

/// Get the actual model name of the host's CPU, e.g. "Intel(R) Xeon(R)
/// Platinum 8275CL CPU @ 3.00GHz".
ErrorOr<std::string> getHostCPUModelName();

//===----------------------------------------------------------------------===//
// Cache sizes
//===----------------------------------------------------------------------===//

/// Get the D$ or unified cache size in bytes at a 1-based cache level index.
/// An error is returned if there is an OS error in finding the cache level.  If
/// the cache level does not exist, 0 is returned.
ErrorOr<size_t> getHostCPUCacheSize(size_t cacheLevel);

/// Get the number of physical cores of in processor.
M::ErrorOr<size_t> getNumPhysicalCores();

//===----------------------------------------------------------------------===//
// SIMD Width
//===----------------------------------------------------------------------===//

/// Gets the SIMD width from the processor features. The features are comma
/// separated.
size_t simdWidthFromFeatures(StringRef features);
size_t simdWidthFromFeatures(ArrayRef<std::string> features);

//===----------------------------------------------------------------------===//
// HostMachineInfo
//===----------------------------------------------------------------------===//

enum class HostProperty {
  TargetTriple,
  OS,
  Arch,
  CPUModel,
  Features,
  SIMDBitWidth,
  CoreCount,
  L1CacheSize,
  L2CacheSize,
  L3CacheSize,
  L4CacheSize,
  Affinities
};

/// Information of host machine.
struct HostMachineInfo {
  std::string triple;
  std::string osName;
  std::string cpuArch;
  std::string cpuModelName;
  // This is the SIMD bit-width of the host system.
  size_t simdBitWidth = 0;
  std::vector<std::string> cpuFeatures;
  size_t numPhysicalCores = 0;
  // These represent either data or unified cache size -- they do not include
  // instruction-only caches, but may include cache size that are shared
  // between instruction and data.
  size_t l1CacheSize = 0;
  size_t l2CacheSize = 0;
  size_t l3CacheSize = 0;
  size_t l4CacheSize = 0;
  // Preferred CPU ids for numPhysicalCores threads if both CPUSystemInfo
  // and thread affinities are supported. Otherwise empty.
  std::optional<std::vector<size_t>> affinities;

  /// Returns a HostMachineInfo representing assumptions about the host machine
  /// encoded by serializedTargetInfo, which should be the result of
  /// M::serializeTargetInfoAttr in MAttrs.h. Only some fields of the result are
  /// filled in:
  ///  - triple
  ///  - cpuArch
  ///  - cpuFeatures
  /// The remainder are empty/zero.
  static ErrorOr<HostMachineInfo>
  deserializeTargetInfoFromJSON(StringRef serializedTargetInfo);

  void print(llvm::raw_ostream &os) const;
  void print(llvm::json::OStream &json) const;
  void print(HostProperty property, llvm::raw_ostream &os) const;

  /// Print information excluding the ones that are likely to change with
  /// threading configuration, such as number of cores and affinities.
  void printStaticInfo(llvm::raw_ostream &os) const;

  /// Returns error if this host machine does not satisfy the assumptions
  /// in required. Only the following fields are checked:
  ///  - triple: if required non-empty, actual must be string equal.
  ///  - cpuArch: if required non-empty, actual must be string equal.
  ///  - cpuFeatures: actual must be superset of required.
  ///
  /// NOTE: We may need to do some canonicalization on triples and cpuArch
  ///       to remove unnecessary detail.
  ErrorOrSuccess
  checkSatisfiesRequirements(const HostMachineInfo &required) const;
};

/// Get information about the host machine.
ErrorOr<HostMachineInfo> getHostMachineInfo();

//===----------------------------------------------------------------------===//
// Memory usage
//===----------------------------------------------------------------------===//

/// Returns the current process' physical memory usage, or 0 if value is
/// not available. Generally determined from the OS's reported resident
/// page value, and may not very reliable.
size_t getProcessPhysicalMemUsage();

//===----------------------------------------------------------------------===//
// OS and architecture-specific utilities, visible for testing only
//===----------------------------------------------------------------------===//

namespace Detail {
#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
ErrorOr<CPUSystemInfo>
getLinuxX86CPUSystemInfoImpl(const cpu_set_t &availableCpus,
                             std::unique_ptr<llvm::MemoryBuffer> buf);
ErrorOr<CPUSystemInfo> getLinuxX86CPUSystemInfo();
#endif

#if defined(HAVE_LINUX_SET_AFFINITY)
ErrorOrSuccess setThreadAffinityLinux(size_t cpuID);
ErrorOrSuccess runWithThreadAffinityLinux(size_t cpuID,
                                          llvm::function_ref<void()> &workFn);
#endif

#if defined(_MSC_VER)
M::ErrorOr<size_t> getNumPhysicalCoresWindows();
#endif
} // namespace Detail

} // namespace M

#endif // SUPPORT_HOST_H
