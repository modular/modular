//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/RuntimeManager.h"
#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"

#include "llvm/Support/ErrorHandling.h"

#include <mutex>

namespace M::AsyncRT {

RuntimeRef getOrCreateRuntime(RuntimeSource source,
                              const RuntimeOptions &options) {
  std::lock_guard<std::mutex> lock(getGlobalRuntimeMutex());
  Runtime *existingRuntime = getGlobalRuntimePointer();
  if (existingRuntime) {
    if (getStoredGlobalRuntimeCreationOptions() != options)
      llvm::report_fatal_error(
          "AsyncRT::getOrCreateRuntime called requesting different options to "
          "those used to create the existing Runtime.");
    return RuntimeRef::copy(existingRuntime);
  }

  assert(Runtime::getCurrentRuntimeOrNull() == nullptr &&
         "creating a runtime from a thread already associated with an outer "
         "runtime");
  CompactRuntimePtr runtimePtr = CompactRuntimePtr::reserve();
  std::unique_ptr<Allocator> allocator =
      getAllocator(options.getAllocatorOptions());
  std::unique_ptr<WorkQueue> workQueue =
      options.singleThreaded
          ? createSingleThreadWorkQueue(runtimePtr)
          : createThreadPoolWorkQueue(
                runtimePtr, options.numThreads, options.maxThreads,
                options.mainWillDonate, options.withAffinity,
                std::chrono::microseconds(options.threadBusyWaitTime),
                options.poolName);
  RuntimeRef newRuntime = RuntimeRef::take(
      new Runtime(runtimePtr, std::move(allocator), std::move(workQueue),
                  source, options.profileFilename,
                  options.runtimeProfilingTypeMask, options.profilerDebuginfo));

  getStoredGlobalRuntimeCreationOptions() = options;
  setGlobalRuntimePointer(newRuntime.getPointer());
  return newRuntime.copy();
}

} // namespace M::AsyncRT
