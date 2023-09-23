//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Threading/ThreadAffinity.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "thread-affinity"

using namespace M;

//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

#if defined(HAVE_LINUX_SET_AFFINITY)
ErrorOrSuccess M::Detail::setThreadAffinityLinux(size_t cpuID) {
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
                                      llvm::function_ref<void()> &workFn) {
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
  return Detail::setThreadAffinityLinux(cpuID);
#endif
  return Error("setThreadAffinity is not supported by this build");
}

ErrorOrSuccess M::runWithThreadAffinity(size_t cpuID,
                                        llvm::function_ref<void()> workFn) {
#if defined(HAVE_LINUX_SET_AFFINITY)
  return Detail::runWithThreadAffinityLinux(cpuID, workFn);
#endif
  return Error("runWithThreadAffinity is not supported by this build");
}
