//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helper for determining which CPU IDs to use for thread affinity given a
// requested number of threads.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/ThreadAffinity.h"
#include "Support/Threading/ThreadAffinity.h"

#include "Support/MArchTarget/Host.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

#ifdef _MSC_VER
#include "llvm/Support/WindowsError.h"
#include <windows.h>
#endif

#define DEBUG_TYPE "llcl"

M::ErrorOr<std::vector<size_t>>
M::LLCL::getThreadAffinityCpuIds(bool withAffinity, size_t numThreads,
                                 size_t maxThreads) {
  int performanceCores = M::getNumPerformanceCores();
  int physicalCores = M::getNumPhysicalCores();
  int logicalCores = M::getNumLogicalCores();
  ErrorOr<CPULimits> limitsOr = CPULimits::get();
  bool usingLimits = !limitsOr.isError() && limitsOr->millicores;

  if (numThreads == 0) {
    // There are some rules to defaulting the number of threads.
    //
    // First, if cores are imbalanced then allow the operating system to
    // schedule us and don't attempt to set any kind of affinity.
    //
    // Second, if affinity is set then only use physical cores.
    //
    // Finally, if affinity is not set then use logical cores.
    if (performanceCores != physicalCores) {
      // If there is an imbalance in the system, we set the number of cores to
      // be the number of performance cores and allow the operating system to
      // schedule us.
      withAffinity = false;
      numThreads = performanceCores;
    } else if (withAffinity) {
      numThreads = physicalCores;
    } else {
      numThreads = logicalCores;
    }
    LLVM_DEBUG(llvm::dbgs() << "getThreadAffinityCpuIds: Defaulting number of "
                            << "threads to physical cores across all "
                            << "sockets " << numThreads << "\n");
  }
  if (usingLimits &&
      numThreads > std::max(1UL, (*limitsOr->millicores) / 1000)) {
    // If we are limited by the cgroup in some way, then we need to cap our
    // untilization. Note that the computation of the affinity set is likely to
    // be affected here, meaning that it will be unbounded, but we don't need
    // to set that explicitly.
    size_t limit = std::max(1UL, (*limitsOr->millicores) / 1000);
    LLVM_DEBUG(llvm::dbgs()
               << "getThreadAffinityCpuIds: Reducing number of threads from "
               << numThreads << " to " << limit << ".\n");
    numThreads = limit;
  }
  if (numThreads > maxThreads) {
    LLVM_DEBUG(llvm::dbgs()
               << "getThreadAffinityCpuIds: Reducing number of threads from "
               << numThreads << " to " << maxThreads << ".\n");
    numThreads = maxThreads;
  }

  auto cpuIDs = std::vector<size_t>(numThreads, kNoAffinity);
  if (withAffinity && haveThreadAffinity()) {
    ErrorOr<CPUSystemInfo> errOrSystemInfo = CPUSystemInfo::get();
    if (const char *err = errOrSystemInfo.getError()) {
      // We will be using the defaults, already set above.
      LLVM_DEBUG(
          llvm::dbgs()
          << "getThreadAffinityCpuIds: Unable to determine CPUSystemInfo: "
          << err << "\n");
    } else {
      // We will be using the preferred CPU IDs, set below.
      LLVM_DEBUG(llvm::dbgs() << "getThreadAffinityCpuIds: System info is "
                              << *errOrSystemInfo << "\n");
      cpuIDs = errOrSystemInfo->getPreferredCpuIDs(numThreads);
      LLVM_DEBUG(llvm::dbgs() << "getThreadAffinityCpuIds: Using thread "
                                 "affinity for CPUs {";
                 llvm::interleave(cpuIDs, llvm::dbgs(), ", ");
                 llvm::dbgs() << "}\n";);
    }
  }
  return cpuIDs;
}

void M::LLCL::runWithThreadAffinity(size_t cpuID,
                                    llvm::function_ref<void()> workFn) {
  if (cpuID == kNoAffinity) {
    workFn();
  } else {
    ErrorOrSuccess errOr = M::runWithThreadAffinity(cpuID, workFn);
    if (const char *err = errOr.getError()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "unable to run with thread affinity: " << err << "\n");
      workFn();
    }
  }
}

void M::LLCL::setThreadAffinity(size_t cpuID) {
  if (cpuID != kNoAffinity) {
    ErrorOrSuccess errOr = M::setThreadAffinity(cpuID);
    if (const char *err = errOr.getError()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "unable to set thread affinity: " << err << "\n");
    }
  }
}
