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

// Execute a single profiled work item.
static void doWork(ProfiledTaskFunction &&profiledTask) {
  std::move(profiledTask.waiting).record();
  profiledTask.running.restart();
  profiledTask.work();
  std::move(profiledTask.running).record();
}

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
    runUntil([]() -> bool { return false; });
  }

  ~SingleThreadWorkQueue() override = default;

  void addTask(TaskFunction &&work,
               WorkProfilerEntry &&profilerEntry) override {
    assert(work);
    WorkProfilerEntry waitingEntry =
        profilerEntry.withNameSuffix(".waiting"); // restarts clock
    workItems.enqueue(ProfiledTaskFunction(
        std::move(work), std::move(waitingEntry), std::move(profilerEntry)));
  }

  void addLocalTask(TaskFunction work) override {
    addTask(std::move(work), WorkProfilerEntry("llcl.waiter"));
  }

  void await(llvm::ArrayRef<AnyAsyncValueRef> values,
             bool runNewTasks) override;
  size_t getParallelismLevel() const override { return 1; }

private:
  /// Execute blocks of work until stopPredicate is true.
  template <typename StopPredicateFn>
  void runUntil(StopPredicateFn &&stopPredicate);

  ConcurrentQueue<ProfiledTaskFunction> workItems;
};
} // namespace

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values,
                                  bool runNewTasks) {
  // We are done when values_remaining drops to zero.
  size_t numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value.andThenSync([&numRemaining]() { --numRemaining; });

  if (numRemaining == 0)
    return;

  // Run work items until numRemaining drops to zero.
  runUntil([&]() -> bool { return numRemaining == 0; });

  assert(numRemaining == 0 &&
         "Some AsyncValues are not ready yet no further "
         "tasks are available to run. Are all input AsyncValues ready?");
}

template <typename StopPredicateFn>
void SingleThreadWorkQueue::runUntil(StopPredicateFn &&stopPredicate) {
  while (auto profiledTask = workItems.dequeue()) {
    doWork(std::move(profiledTask));
    if (stopPredicate())
      break;
  }
}

std::unique_ptr<WorkQueue> M::LLCL::createSingleThreadWorkQueue() {
  return std::make_unique<SingleThreadWorkQueue>();
}
