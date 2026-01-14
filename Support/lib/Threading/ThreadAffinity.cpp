//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Threading/ThreadAffinity.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/DebugLog.h"
#include <cstddef>

#define DEBUG_TYPE "thread-affinity"

#ifdef __linux__
#define HAVE_LINUX_SET_AFFINITY 1
#else
#define HAVE_LINUX_SET_AFFINITY 0
#endif

using namespace M;

//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

#if HAVE_LINUX_SET_AFFINITY
static ErrorOrSuccess setThreadAffinityLinux(size_t cpuID) {
  assert(cpuID < CPU_SETSIZE);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpuID, &cpuset);
  if (int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset))
    return Error("unable to set thread CPU affinity: " + std::to_string(rc));
  return success();
}

ErrorOrSuccess static runWithThreadAffinityLinux(
    size_t cpuID, llvm::function_ref<void()> &workFn) {
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
    LDBG() << "runWithThreadAffinityLinux: unable to restore "
              "thread CPU affinity: "
           << rc;
  }
  return success();
}
#endif

bool M::haveThreadAffinity() {
#if HAVE_LINUX_SET_AFFINITY
  return true;
#else
  return false;
#endif // HAVE_LINUX_SET_AFFINITY
}

ErrorOrSuccess M::setThreadAffinity(size_t cpuID) {
#if HAVE_LINUX_SET_AFFINITY
  return setThreadAffinityLinux(cpuID);
#else
  return Error("setThreadAffinity is not supported by this build");
#endif // HAVE_LINUX_SET_AFFINITY
}

ErrorOrSuccess M::runWithThreadAffinity(size_t cpuID,
                                        llvm::function_ref<void()> workFn) {
#if HAVE_LINUX_SET_AFFINITY
  return runWithThreadAffinityLinux(cpuID, workFn);
#else
  return Error("runWithThreadAffinity is not supported by this build");
#endif // HAVE_LINUX_SET_AFFINITY
}
