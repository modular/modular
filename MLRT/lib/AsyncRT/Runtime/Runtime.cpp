//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core AsyncRT Runtime.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "MLRT/AsyncRT/Runtime/Allocator.h"
#include "MLRT/AsyncRT/Runtime/AsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/CompactRuntimePtr.h"
#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/AsyncRT/Support/Chain.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <chrono>

using namespace M;
using namespace M::AsyncRT;

void WorkQueue::vtableAnchor() {}
void Allocator::vtableAnchor() {}

/// Create "Chain" AsyncValue, making sure that "Chain" type is registered
/// before the construction. "Chain" is core to AsyncRT implementation, so it
/// needs to be registered unconditionally from AsyncRT.
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

Runtime::Runtime(CompactRuntimePtr runtimePtr,
                 std::unique_ptr<Allocator> allocator,
                 std::unique_ptr<WorkQueue> workQueue, RuntimeSource source,
                 StringRef profileFilename, uint64_t runtimeProfilingTypeMask,
                 RuntimeOptions::ProfilerDebuginfo profilerDebuginfo)
    : signature(TypeID::getSignature() ^ CompactRuntimePtr::getSignature()),
      allocator(std::move(allocator)), workQueue(std::move(workQueue)),
      profilerDebuginfo(profilerDebuginfo), runtimeIndex(runtimePtr.index),
      source(source), readyChain(createReadyChain(*this)) {
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

  // Clear global pointer if it pointed to this runtime (same pattern as
  // Context).
  clearGlobalRuntimePointerIfEquals(this);

  // We're done with profiling.
  if (profiler) {
    if (auto e = profiler->write("-"))
      llvm::report_fatal_error("unable to write time trace profile");
  }
}

std::unique_ptr<Allocator>
AsyncRT::getAllocator(const AllocatorOptions &options) {
  // Create base allocator: UseAfterFree, TCMalloc, or Malloc
  // These are mutually exclusive and must be enabled at compile time.
  std::unique_ptr<Allocator> allocator;
  if (options.useAfterFreeAllocator) {
#if HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
    allocator = createUseAfterFreeAllocator();
#else
    llvm_unreachable("cannot use the user-after-free allocator");
#endif
  } else if (options.tcmallocAllocator) {
    allocator = createTCMallocAllocator();
  } else {
    allocator = createMallocAllocator();
  }
  // Optionally wrap in one or more debug allocators.
  if (options.leakCheckedAllocator)
    allocator = createLeakCheckAllocator(std::move(allocator));
  if (options.profilingAllocator)
    allocator = createProfilingAllocator(std::move(allocator));
  return allocator;
}
