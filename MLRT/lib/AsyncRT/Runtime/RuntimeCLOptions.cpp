//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/RuntimeCLOptions.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"

using namespace M::AsyncRT;

std::unique_ptr<Runtime> RuntimeOptions::createRuntime() const {
  RuntimeOptions runtimeOptions; //{*this};
  // runtimeOptions.workQueueType = getWorkQueueType();
  switch (getWorkQueueType()) {
  case RuntimeOptions::WorkQueueType::kDefault:
    assert(0 && "should be resolved");
    LLVM_FALLTHROUGH;
  case RuntimeOptions::WorkQueueType::kSingleThread:
    runtimeOptions.singleThreaded = true;
    break;
  case RuntimeOptions::WorkQueueType::kThreadPool:
    runtimeOptions.numThreads = numThreads;
    runtimeOptions.maxThreads = maxThreads;
    runtimeOptions.withAffinity = withAffinity;
    runtimeOptions.threadBusyWaitTime = threadBusyWaitTime;
    break;
  }
  runtimeOptions.profileFilename = getProfileFilename();
  runtimeOptions.profilerDebuginfo = profilerDebuginfo;
  return AsyncRT::createUniqueRuntime(runtimeOptions);
}
