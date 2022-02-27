//===- SingleThreadWorkQueue.cpp - Simple WorkQueue implementation --------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SingleThreadWorkQueue.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/RCRef.h"
#include "Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
using namespace LLCL;

namespace {

/// This class implements a work queue that uses the client thread to execute
/// all work. It spawns no additional threads and has no internal
/// synchronization.  The only thread used is the client thread when it gets
/// donated.
class SingleThreadedWorkQueue : public WorkQueue {
public:
  SingleThreadedWorkQueue() {}
  ~SingleThreadedWorkQueue() {
    assert(workItems.empty() &&
           "WorkQueue shouldn't be destroyed if work remains!");
  }

  void addTask(TaskFunction work) override;
  void await(llvm::ArrayRef<RCRef<AsyncValue>> values) override;
  void quiesce() override;

private:
  void doWork(llvm::unique_function<bool()> stopPredicate);
  std::vector<TaskFunction> workItems;
};
} // end anonymous namespace

/// Enqueue a block of work. This does not use synchronization since this
void SingleThreadedWorkQueue::addTask(TaskFunction work) {
  workItems.push_back(std::move(work));
}

void SingleThreadedWorkQueue::await(llvm::ArrayRef<RCRef<AsyncValue>> values) {
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

/// Block until the system is quiescent (no pending work and no inflight work).
/// Because we are single threaded, we *have* to use the client thread to run
/// work - there is no one else to do it.
void SingleThreadedWorkQueue::quiesce() { doWork({}); }

/// Execute blocks of work.  If `stopPredicate` is non-null, then we stop
/// early if it returns true.
void SingleThreadedWorkQueue::doWork(
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
  return std::make_unique<SingleThreadedWorkQueue>();
}
