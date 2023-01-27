//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Support/ConcurrentQueue.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
using namespace M::LLCL;

namespace {

/// This class implements a work queue that uses the client thread to execute
/// all work. It spawns no additional threads and has no internal
/// synchronization.  The only thread used is the client thread when it gets
/// donated.
class SingleThreadWorkQueue : public WorkQueue {
public:
  SingleThreadWorkQueue() = default;

  void shutdown() override {
    // Complete any work that's still in-flight.
    doWork([]() -> bool { return false; });
  }

  ~SingleThreadWorkQueue() override {}

  /// Enqueue a block of work. This does not use synchronization since this
  void addTask(TaskFunction work) override {
    workItems.enqueue(std::move(work));
  }

  void await(llvm::ArrayRef<AnyAsyncValueRef> values) override;
  size_t getParallelismLevel() const override { return 1; }

private:
  template <typename Callback>
  void doWork(Callback &&stopPredicate);

  ConcurrentQueue<TaskFunction> workItems;
};
} // namespace

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values) {
  // We are done when values_remaining drops to zero.
  int numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value.andThenSync([&numRemaining]() { --numRemaining; });

  if (numRemaining == 0)
    return;

  // Run work items until numRemaining drops to zero.
  doWork([&]() -> bool { return numRemaining == 0; });
}

/// Execute blocks of work.  If `stopPredicate` is non-null, then we stop
/// early if it returns true.
template <typename T>
void SingleThreadWorkQueue::doWork(T &&stopPredicate) {
  while (auto callable = workItems.dequeue()) {
    callable();
    if (stopPredicate())
      break;
  }
}

std::unique_ptr<WorkQueue> M::LLCL::createSingleThreadWorkQueue() {
  return std::make_unique<SingleThreadWorkQueue>();
}
