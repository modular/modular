//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core LLCL Runtime.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CompactRuntimePtr.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Chain.h"
#include "Support/Context.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace M;
using namespace M::AsyncRT;

void WorkQueue::vtableAnchor() {}
void Allocator::vtableAnchor() {}

/// Create "Chain" AsyncValue, making sure that "Chain" type is registered
/// before the construction. "Chain" is core to LLCL implementation, so it
/// needs to be registered unconditionally from LLCL.
static AsyncValueRef<Chain> createReadyChain(Runtime &runtime) {
  return AsyncValueRef<Chain>::createReady(runtime);
}

//===----------------------------------------------------------------------===//
// CompactRuntimePtr
//===----------------------------------------------------------------------===//

CompactRuntimePtr::CompactRuntimePtr(Runtime *runtime)
    : CompactRuntimePtr(runtime ? runtime->getCompactPtr()
                                : CompactRuntimePtr()) {}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

Runtime::Runtime(CompactRuntimePtr runtimePtr, Context *context,
                 std::unique_ptr<Allocator> allocator,
                 std::unique_ptr<WorkQueue> workQueue,
                 StringRef profileFilename, uint64_t runtimeProfilingTypeMask,
                 RuntimeOptions::ProfilerDebuginfo profilerDebuginfo)
    : context(context),
      signature(TypeID::getSignature() ^ CompactRuntimePtr::getSignature()),
      allocator(std::move(allocator)), workQueue(std::move(workQueue)),
      profilerDebuginfo(profilerDebuginfo), runtimeIndex(runtimePtr.index),
      readyChain(createReadyChain(*this)) {
  // Establish association of runtime to runtime index.
  Detail::RuntimeTable::getSingleton().setRuntime(runtimePtr.index, this);

  // NOTE: Users can't pass in profileFilename AND activate the time
  // profiler in the caller.
  if (!profileFilename.empty())
    profiler.emplace(/*timeTraceGranularity=*/0, "Main", profileFilename,
                     runtimeProfilingTypeMask);
}

Runtime::~Runtime() {
  // Explicitly shutdown the workQueue while the Runtime is still alive.
  // Shutting down the workqueue will execute unfinished tasks, and those tasks
  // can add new tasks to the runtime, so we need to make sure to tie all this
  // off before invalidating the workQueue pointer.
  workQueue->shutdown();

  // Remove association of runtime to runtime index.
  Detail::RuntimeTable::getSingleton().clearRuntime(runtimeIndex);

  // We're done with profiling.
  if (profiler) {
    if (auto E = profiler->write("-"))
      llvm::report_fatal_error("unable to write time trace profile");
  }
}

std::unique_ptr<Allocator>
AsyncRT::getAllocator(const RuntimeOptions &options) {
  if (options.useAfterFreeAllocator) {
#ifdef HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
    return createUseAfterFreeAllocator();
#else
    llvm_unreachable("cannot use the user-after-free allocator");
#endif
  }
  if (options.tcmallocAllocator) {
#ifdef USE_TCMALLOC
    return createTCMallocAllocator();
#else
    llvm_unreachable("cannot use the tcmalloc allocator because the code was "
                     "not compiled with the tcmalloc library");
#endif
  }

  return createMallocAllocator();
}

static std::unique_ptr<Runtime>
createRuntimeImpl(Context *context, const RuntimeOptions &options) {
  CompactRuntimePtr runtimePtr = CompactRuntimePtr::reserve();
  std::unique_ptr<Allocator> allocator = getAllocator(options);
  if (options.leakCheckedAllocator)
    allocator = createLeakCheckAllocator(std::move(allocator));
  if (options.profilingAllocator)
    allocator = createProfilingAllocator(std::move(allocator));
  std::unique_ptr<WorkQueue> workQueue =
      options.singleThreaded
          ? createSingleThreadWorkQueue(runtimePtr)
          : createThreadPoolWorkQueue(
                runtimePtr, options.numThreads, options.maxThreads,
                options.mainWillDonate, options.withAffinity,
                std::chrono::microseconds(options.threadBusyWaitTime),
                options.poolName, options.paranoid);
  return std::make_unique<Runtime>(
      runtimePtr, context, std::move(allocator), std::move(workQueue),
      options.profileFilename, options.runtimeProfilingTypeMask,
      options.profilerDebuginfo);
}

std::unique_ptr<Runtime>
AsyncRT::createUniqueRuntime(const RuntimeOptions &options) {
  assert(Runtime::getCurrentRuntimeOrNull() == nullptr &&
         "creating a runtime from a thread already associated with an outer "
         "runtime");
  return createRuntimeImpl(nullptr, options);
}

std::unique_ptr<Runtime>
AsyncRT::createNestedRuntime(const RuntimeOptions &options) {
  if (auto runtime = Runtime::getCurrentRuntimeOrNull())
    return createRuntimeImpl(runtime->context, options);
  else
    return createRuntimeImpl(nullptr, options);
}
