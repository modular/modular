//===- SingleThreadWorkQueue.cpp ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/ConcurrentQueue.h"
#include "Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
using namespace LLCL;

namespace {

/// This class implements a work queue that uses the client thread to execute
/// all work. It spawns no additional threads and has no internal
/// synchronization.  The only thread used is the client thread when it gets
/// donated.
class SingleThreadWorkQueue : public WorkQueue {
public:
  SingleThreadWorkQueue() {}
  ~SingleThreadWorkQueue() {
    // Complete any work that's still in-flight.
    doWork([]() -> bool { return false; });
  }

  void await(llvm::ArrayRef<AnyAsyncValueRef> values) override;
  int getParallelismLevel() const override { return 1; }

protected:
  /// Enqueue a block of work. This does not use synchronization since this
  void addTaskInternal(TaskFunctionBase *work) override {
    workItems.enqueue(work);
  }

private:
  template <typename Callback>
  void doWork(Callback &&stopPredicate);

  ConcurrentQueue workItems;
};
} // end anonymous namespace

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values) {
  // We are done when values_remaining drops to zero.
  int numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value->andThen([&numRemaining]() { --numRemaining; });

  if (numRemaining == 0)
    return;

  // Run work items until numRemaining drops to zero.
  doWork([&]() -> bool { return numRemaining == 0; });
}

/// Execute blocks of work.  If `stopPredicate` is non-null, then we stop
/// early if it returns true.
template <typename T>
void SingleThreadWorkQueue::doWork(T &&stopPredicate) {
  while (auto item = workItems.dequeue()) {
    if (!stopPredicate()) {
      item->call();
    }
  }
}

std::unique_ptr<WorkQueue> LLCL::createSingleThreadWorkQueue() {
  return std::make_unique<SingleThreadWorkQueue>();
}
