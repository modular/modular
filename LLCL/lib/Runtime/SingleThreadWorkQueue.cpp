//===- SingleThreadWorkQueue.cpp ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
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
    doWork({});
  }

  void addTask(TaskFunction work) override;
  void await(llvm::ArrayRef<AnyAsyncValueRef> values) override;
  int getParallelismLevel() const override { return 1; }

private:
  void doWork(llvm::unique_function<bool()> stopPredicate);
  std::vector<TaskFunction> workItems;
};
} // end anonymous namespace

/// Enqueue a block of work. This does not use synchronization since this
void SingleThreadWorkQueue::addTask(TaskFunction work) {
  workItems.push_back(std::move(work));
}

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
void SingleThreadWorkQueue::doWork(
    llvm::unique_function<bool()> stopPredicate) {

  std::vector<TaskFunction> localWorkItems;
  while (!workItems.empty()) {
    // Work items can add new items to the vector, and we generally want to run
    // things in order, so make sure we explicitly pop the item off before a new
    // one is added.
    std::swap(localWorkItems, workItems);
    for (auto &item : localWorkItems) {
      // Check the stop predicate.
      if (!stopPredicate || !stopPredicate()) {
        item();
        continue;
      }

      // If the stop predicate said to halt, then we need to take the left-over
      // items from localWorkItems and put them back into our workItems.
      size_t itemIdx = &item - localWorkItems.data();
      workItems.insert(
          workItems.begin(),
          std::make_move_iterator(localWorkItems.begin() + itemIdx),
          std::make_move_iterator(localWorkItems.end()));
      return;
    }
    localWorkItems.clear();
  }
}

std::unique_ptr<WorkQueue> LLCL::createSingleThreadWorkQueue() {
  return std::make_unique<SingleThreadWorkQueue>();
}
