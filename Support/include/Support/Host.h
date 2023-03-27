//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HOST_H
#define SUPPORT_HOST_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <string>

namespace llvm::json {
class OStream;
}

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#define HOST_IS_APPLE_SILICON_PROCESSOR
#endif

namespace M {

enum class HostProperty {
  TargetTriple,
  OS,
  Arch,
  Features,
  CoreCount,
  SIMDBitWidth,
  L1CacheSize,
  L2CacheSize,
  L3CacheSize,
  L4CacheSize
};

ErrorOr<size_t> getHostCPUCacheSize(size_t cacheLevel);

/// Information of host machine.
struct HostMachineInfo {
  std::string triple;
  std::string osName;
  std::string cpuArch;
  std::vector<std::string> cpuFeatures;
  size_t numPhysicalCores;
  size_t simdBitWidth;
  size_t l1CacheSize;
  size_t l2CacheSize;
  size_t l3CacheSize;
  size_t l4CacheSize;

  void print(llvm::raw_ostream &os) const;
  void print(llvm::json::OStream &json) const;
  void print(HostProperty property, llvm::raw_ostream &os) const;
};

/// Get information about the host machine.
ErrorOr<HostMachineInfo> getHostMachineInfo();

/// Returns the current process' physical memory usage, or 0 if value is
/// not available. Generally determined from the OS's reported resident
/// page value, and may not very reliable.
size_t getProcessPhysicalMemUsage();

} // namespace M

#endif // SUPPORT_HOST_H
