//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Support/ConcurrentQueue.h"
#include "LLCL/Support/ThreadAffinity.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"

#define DEBUG_TYPE "llcl"

using namespace M;
using namespace LLCL;

namespace {

/// This class implements a work queue that uses the client thread to execute
/// all work. It spawns no additional threads and has no internal
/// synchronization.  The only thread used is the client thread when it gets
/// donated.
class SingleThreadWorkQueue : public WorkQueue {
public:
  SingleThreadWorkQueue(size_t cpuID) : cpuID(cpuID) {}

  void shutdown() override {
    // Complete any work that's still in-flight.
    runUntil([]() -> bool { return false; });
  }

  ~SingleThreadWorkQueue() override = default;

  void addTask(TaskFunction &&work,
               WorkProfilerEntry &&profilerEntry) override {
    assert(work);
    // Begin the waiting clock.
    InternalProfilerEntry waitingEntry =
        profilerEntry.copy<InternalProfilerEntry>().withNameSuffix(
            ".waiting"); // restarts clock
    workItems.enqueue(ProfiledTaskFunction(
        std::move(work), std::move(waitingEntry), std::move(profilerEntry)));
  }

  void addLocalTask(TaskFunction work) override {
    addTask(std::move(work), AllWorkItemsProfilerEntry::create("llcl.waiter")
                                 .copy<WorkProfilerEntry>());
  }

  void await(llvm::ArrayRef<AnyAsyncValueRef> values, bool mayDonate) override;
  size_t getParallelismLevel() const override { return 1; }

private:
  /// Execute blocks of work until stopPredicate is true, setting thread
  /// affinity if reqested.
  template <typename StopPredicateFn>
  void runUntil(StopPredicateFn stopPredicate);

  /// Actually run work items.
  template <typename StopPredicateFn>
  void runUntilImpl(StopPredicateFn stopPredicate);

  // Execute a single profiled work item.
  void doWork(ProfiledTaskFunction &&profiledTask) {
    // Record the waiting clock.
    std::move(profiledTask.waiting).record();
    // Start the running clock.
    profiledTask.running.restart();
    assert(running == nullptr);
    running = &profiledTask.running;
    // Do the work.
    profiledTask.work();
    running = nullptr;
    // Record the running clock (if not recorded already).
    std::move(profiledTask.running).record();
  }

  /// CPU ID to set affinity to when running the runUntil loop.
  size_t cpuID;
  /// Pending work items.
  ConcurrentQueue<ProfiledTaskFunction> workItems;
  /// The profiling entry for the currently running work item, or null if none.
  WorkProfilerEntry *running = nullptr;
};
} // namespace

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values,
                                  bool mayDonate) {
  // Note we must ignore mayDonate.

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
void SingleThreadWorkQueue::runUntil(StopPredicateFn stopPredicate) {
  LLCL::runWithThreadAffinity(cpuID, [&]() { runUntilImpl(stopPredicate); });
}

template <typename StopPredicateFn>
void SingleThreadWorkQueue::runUntilImpl(StopPredicateFn stopPredicate) {
  WorkProfilerEntry *origRunning = running;
  if (origRunning) {
    WorkProfilerEntry newRunning = origRunning->withNameSuffix(".post");
    // Switch out the original entry and an entry we'll start once we're done
    // pumping work.
    std::swap(newRunning, *origRunning);
    // Record the currently running clock (if any) before we start any more
    // work.
    std::move(newRunning).record();
    running = nullptr;
  }

  while (auto profiledTask = workItems.dequeue()) {
    doWork(std::move(profiledTask));
    if (stopPredicate())
      break;
  }

  if (origRunning) {
    // Resume tracing the original work item.
    origRunning->restart();
    running = origRunning;
  }
}

std::unique_ptr<WorkQueue> M::LLCL::createSingleThreadWorkQueue() {
  auto cpuIDOr = getThreadAffinityCpuIds(/*numThreads=*/1, /*maxWorkers=*/1);

  // TODO: This function should return the error back to caller.
  if (cpuIDOr.isError())
    llvm::report_fatal_error(cpuIDOr.getError());
  std::vector<size_t> cpuIDs = *cpuIDOr;
  assert(cpuIDs.size() == 1);
  return std::make_unique<SingleThreadWorkQueue>(cpuIDs[0]);
}
