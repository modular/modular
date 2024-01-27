//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core LLCL Runtime.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Chain.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace M;
using namespace M::LLCL;

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

Runtime::Runtime(CompactRuntimePtr runtimePtr,
                 std::unique_ptr<Allocator> allocator,
                 std::unique_ptr<WorkQueue> workQueue,
                 StringRef profileFilename)
    : signature(TypeID::getSignature() ^ CompactRuntimePtr::getSignature()),
      allocator(std::move(allocator)), workQueue(std::move(workQueue)),
      profileFilename(profileFilename), runtimeIndex(runtimePtr.index),
      readyChain(createReadyChain(*this)) {
  // Establish association of runtime to runtime index.
  Detail::RuntimeTable::getSingleton().setRuntime(runtimePtr.index, this);

  // NOTE: Users can't pass in profileFilename AND activate the time
  // profiler in the caller.
  if (!profileFilename.empty())
    profiler.emplace(/*timeTraceGranularity=*/0, "Main");
}

Runtime::~Runtime() {
  // Explicitly shutdown the workQueue while the Runtime is still alive.
  // Shutting down the workqueue will execute unfinished tasks, and those tasks
  // can add new tasks to the runtime, so we need to make sure to tie all this
  // off before invalidating the workQueue pointer.
  workQueue->shutdown();

  // Clear cancellation value if present.
  restartFromCancellation();

  // Remove association of runtime to runtime index.
  Detail::RuntimeTable::getSingleton().clearRuntime(runtimeIndex);

  // We're done with profiling.
  if (profiler) {
    if (auto E = profiler->write(profileFilename, "-"))
      llvm::report_fatal_error("unable to write time trace profile");
  }
}

/// Cancel the current MEF Execution. This transitions this Runtime to the
/// canceled state, which causes all asynchronously executing threads to be
/// canceled when they check the cancellation state (e.g. in MEFExecutor).
void Runtime::cancelExecution(EncodedDiagnostic message) {
  AnyAsyncValueRef messageVal =
      AnyAsyncValueRef::createError(*this, std::move(message));

  AsyncValue *expectedValue = nullptr;
  // Use memory_order_release for the success case so that error_value is
  // visible to other threads when they load with memory_order_acquire. For the
  // failure case, we do not care about expectedValue, so we can use
  // memory_order_relaxed.
  if (cancelValue.compare_exchange_strong(
          expectedValue, messageVal.getPointer(), std::memory_order_release,
          std::memory_order_relaxed))
    (void)messageVal.releasePointer();
}

/// restartFromCancellation() transitions Runtime from the canceled state to
/// the normal execution state.
void Runtime::restartFromCancellation() {
  // Use memory_order_acq_rel so that previous writes on this thread are visible
  // to other threads and previous writes from other threads (e.g. the return
  // 'value') are visible to this thread.
  AsyncValue *value = cancelValue.exchange(nullptr, std::memory_order_acq_rel);
  // Deallocate the value.
  AnyAsyncValueRef::take(value);
}

static std::unique_ptr<Runtime>
createRuntimeImpl(const RuntimeOptions &options) {
  CompactRuntimePtr runtimePtr = CompactRuntimePtr::reserve();
#if defined(HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR)
  std::unique_ptr<Allocator> allocator =
      options.useAfterFreeAllocator ? createUseAfterFreeAllocator()
      : options.tcmallocAllocator   ? createTCMallocAllocator()
                                    : createMallocAllocator();
#else
  std::unique_ptr<Allocator> allocator = options.tcmallocAllocator
                                             ? createTCMallocAllocator()
                                             : createMallocAllocator();
#endif
  if (options.leakCheckedAllocator)
    allocator = createLeakCheckAllocator(std::move(allocator));
  if (options.profilingAllocator)
    allocator = createProfilingAllocator(std::move(allocator));
  std::unique_ptr<WorkQueue> workQueue =
      options.singleThreaded
          ? createSingleThreadWorkQueue(runtimePtr)
          : createThreadPoolWorkQueue(
                runtimePtr, options.numThreads, options.mainWillDonate,
                options.threadBusyWaitTime, options.poolName, options.paranoid);
  return std::make_unique<Runtime>(runtimePtr, std::move(allocator),
                                   std::move(workQueue),
                                   options.profileFilename);
}

std::unique_ptr<Runtime>
LLCL::createUniqueRuntime(const RuntimeOptions &options) {
  assert(Runtime::getCurrentRuntimeOrNull() == nullptr &&
         "creating a runtime from a thread already associated with an outer "
         "runtime");
  return createRuntimeImpl(options);
}

std::unique_ptr<Runtime>
LLCL::createNestedRuntime(const RuntimeOptions &options) {
  return createRuntimeImpl(options);
}

ConditionallyOwnedPointer<Runtime>
LLCL::createRuntimeIfNeeded(const RuntimeOptions &options) {
  return ConditionallyOwnedPointer<Runtime>::takeIfNeeded(
      Runtime::getCurrentRuntimeOrNull(),
      [&options]() { return createRuntimeImpl(options).release(); });
}
