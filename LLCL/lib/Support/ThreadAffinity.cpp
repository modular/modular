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

namespace {

void adjustForCpuLimits(std::vector<size_t> &cpuIDs) {
#if defined(HAVE_LINUX_X86_SYSTEM_INFO)
  M::LLCL::Detail::adjustForLinuxCpuLimits(M::Detail::getLinuxCPULimits(),
                                           cpuIDs);
#endif
}

} // namespace

M::ErrorOr<std::vector<size_t>>
M::LLCL::getThreadAffinityCpuIds(size_t numThreads, size_t maxWorkers) {
  if (numThreads == 0) {
    numThreads = M::getNumPhysicalCores();
    LLVM_DEBUG(llvm::dbgs() << "getThreadAffinityCpuIds: Defaulting number of "
                            << "threads to physical cores across all "
                            << "sockets " << numThreads << "\n");
  }
  if constexpr (kUseThreadAffinity) {
    if (haveThreadAffinity()) {
      ErrorOr<CPUSystemInfo> errOrSystemInfo = CPUSystemInfo::get();
      if (const char *err = errOrSystemInfo.getError()) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "getThreadAffinityCpuIds: Unable to determine CPUSystemInfo: "
            << err << "\n");
        // Fallthrough for fallback case.
      } else {
        LLVM_DEBUG(llvm::dbgs() << "getThreadAffinityCpuIds: System info is "
                                << *errOrSystemInfo << "\n");
        if (numThreads > maxWorkers) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "getThreadAffinityCpuIds: Reducing number of threads from "
              << numThreads << " to " << maxWorkers << ".\n");
          numThreads = maxWorkers;
        }
        std::vector<size_t> cpuIDs =
            errOrSystemInfo->getPreferredCpuIDs(numThreads);
        LLVM_DEBUG(llvm::dbgs() << "getThreadAffinityCpuIds: Using thread "
                                   "affinity for CPUs {";
                   llvm::interleave(cpuIDs, llvm::dbgs(), ", ");
                   llvm::dbgs() << "}\n";);
        adjustForCpuLimits(cpuIDs);
        return cpuIDs;
      }
    }
  }

  if (numThreads > maxWorkers) {
    LLVM_DEBUG(llvm::dbgs()
               << "getThreadAffinityCpuIds: Reducing number of threads from "
               << numThreads << " to " << maxWorkers << ".\n");
    numThreads = maxWorkers;
  }
  auto cpuIDs = std::vector<size_t>(numThreads, kNoAffinity);
  adjustForCpuLimits(cpuIDs);
  return cpuIDs;
}

void M::LLCL::runWithThreadAffinity(size_t cpuID,
                                    llvm::function_ref<void()> workFn) {
  if constexpr (kUseThreadAffinity) {
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
  } else {
    workFn();
  }
}

void M::LLCL::setThreadAffinity(size_t cpuID) {
  if constexpr (kUseThreadAffinity) {
    if (cpuID != kNoAffinity) {
      ErrorOrSuccess errOr = M::setThreadAffinity(cpuID);
      if (const char *err = errOr.getError()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "unable to set thread affinity: " << err << "\n");
      }
    }
  }
}

#if defined(HAVE_LINUX_X86_SYSTEM_INFO)

void M::LLCL::Detail::adjustForLinuxCpuLimits(
    const M::Detail::CPULimits &limits, std::vector<size_t> &cpuIDs) {
  if (limits.quota_us != -1) {
    // Limit thread count to the below to prevent inadvertent CFS scheduler
    // throttling when CPU limits are in use. Also disables thread affinity.
    const size_t maxProcessors = limits.maxProcessors();
    if (maxProcessors < cpuIDs.size()) {
      cpuIDs.resize(maxProcessors);
      cpuIDs.assign(maxProcessors, kNoAffinity);
    }
  }
}

#endif // defined(HAVE_LINUX_X86_SYSTEM_INFO)
