//===- Runtime.cpp --------------------------------------------------------===//
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
#include "llvm/ADT/ArrayRef.h"
using namespace LLCL;

void WorkQueue::vtableAnchor() {}
void Allocator::vtableAnchor() {}

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
    : CompactRuntimePtr(runtime->getCompactPtr()) {}

Runtime *CompactRuntimePtr::get() const {
  assert(index != kInvalidIndex);
  return allRuntimes[index];
}

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

Runtime::Runtime(std::unique_ptr<Allocator> allocator,
                 std::unique_ptr<WorkQueue> workQueue)
    : allocator(std::move(allocator)), workQueue(std::move(workQueue)),
      runtimeIndex(nextRuntimeIndex.fetch_add(1)),
      readyChain(AsyncValueRef<Chain>::createReady(*this)) {
  // We provide a dense numbering of runtime instances right now, but we could
  // make this fancier to allow deallocating and reusing indexes if needbe.
  assert(runtimeIndex < CompactRuntimePtr::kInvalidIndex &&
         "Created too many Runtimes");
  allRuntimes[runtimeIndex] = this;
}

Runtime::~Runtime() {
  // Explicitly call the destructor here while the workQueue is still alive.
  // This is because the work queue's destructor will clear out its internal
  // task list by doing all the work required. This will result in segfaults if
  // the work queue itself has been freed already because inflight tasks that
  // run during the destructor try to add work to the Runtime object whose
  // pointer is now gone.
  workQueue->~WorkQueue();
  WorkQueue *wq = workQueue.release();
  operator delete(wq);

  // Clear cancellation value if present.
  restartFromCancellation();
  allRuntimes[runtimeIndex] = nullptr;
}

/// Cancel the current BEF Execution. This transitions this Runtime to the
/// canceled state, which causes all asynchronously executing threads to be
/// canceled when they check the cancellation state (e.g. in BEFExecutor).
void Runtime::cancelExecution(EncodedDiagnostic message) {
  RCRef<AsyncValue> messageVal = AsyncValue::createError(std::move(message));

  AsyncValue *expectedValue = nullptr;
  // Use memory_order_release for the success case so that error_value is
  // visible to other threads when they load with memory_order_acquire. For the
  // failure case, we do not care about expectedValue, so we can use
  // memory_order_relaxed.
  if (cancelValue.compare_exchange_strong(
          expectedValue, messageVal.getPointer(), std::memory_order_release,
          std::memory_order_relaxed))
    (void)messageVal.release();
}

/// restartFromCancellation() transitions Runtime from the canceled state to
/// the normal execution state.
void Runtime::restartFromCancellation() {
  // Use memory_order_acq_rel so that previous writes on this thread are visible
  // to other threads and previous writes from other threads (e.g. the return
  // 'value') are visible to this thread.
  AsyncValue *value = cancelValue.exchange(nullptr, std::memory_order_acq_rel);
  // Deallocate the value.
  RCRef<AsyncValue>::take(value);
}
