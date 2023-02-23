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
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/Profiling.h"
#include "Support/TimeProfiler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace M::LLCL;

void WorkQueue::vtableAnchor() {}
void Allocator::vtableAnchor() {}

/// Create "Chain" AsynchValue, making sure that "Chain" type is registered
/// before the construction. "Chain" is core to LLCL implemention, so it
/// needs to be registered unconditonally from LLCL.
static AsyncValueRef<Chain> createReadyChain(Runtime &runtime) {
  AsyncValue::registerType<Chain>();
  return AsyncValueRef<Chain>::createReady(runtime);
}

//===----------------------------------------------------------------------===//
// CompactRuntimePtr
//===----------------------------------------------------------------------===//

/// The `CompactRuntimePtr` type provides a pointer compressed version of
/// `Runtime*` that fits in 8 bits.  This allows every AsyncValue to carry a
/// backpointer to the Runtime that allocated them, and allows deallocating the
/// memory for the AsyncValue through the Runtime's allocator.
///
/// This is implemented with a static array of Runtime pointers that are given
/// unique IDs.
static std::atomic<uint8_t> nextRuntimeIndex{0};
static Runtime *allRuntimes[CompactRuntimePtr::kInvalidIndex];

CompactRuntimePtr::CompactRuntimePtr(Runtime *runtime)
    : CompactRuntimePtr(runtime ? runtime->getCompactPtr()
                                : CompactRuntimePtr()) {}

Runtime *CompactRuntimePtr::get() const {
  assert(index != kInvalidIndex);
  return allRuntimes[index];
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

Runtime::Runtime(std::unique_ptr<Allocator> allocator,
                 std::unique_ptr<WorkQueue> workQueue,
                 StringRef profileFilename)
    : signature(TypeID::getSignature()), allocator(std::move(allocator)),
      workQueue(std::move(workQueue)), profileFilename(profileFilename),
      runtimeIndex(nextRuntimeIndex.fetch_add(1)),
      readyChain(createReadyChain(*this)) {
  // We provide a dense numbering of runtime instances right now, but we could
  // make this fancier to allow deallocating and reusing indexes if needbe.
  assert(runtimeIndex < CompactRuntimePtr::kInvalidIndex &&
         "Created too many Runtimes");
  allRuntimes[runtimeIndex] = this;

  // Register the C scalar types as async value types.
  AsyncValue::registerTypes<bool, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t, float, double>();

  // NOTE: Users can't pass in profileFilename AND activate the time profiler in
  // the caller.
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
  allRuntimes[runtimeIndex] = nullptr;

  // If we are the latest runtime index to be allocated, we can deallocate our
  // ID (allowing it to be reused).  This is best-effort but not guaranteed.
  uint8_t expected = runtimeIndex + 1;
  (void)nextRuntimeIndex.compare_exchange_strong(expected, runtimeIndex);

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
