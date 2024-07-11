//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/WorkQueue.h"

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Support/ConcurrentQueue.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"

#define DEBUG_TYPE "llcl"

using namespace M;
using namespace AsyncRT;

namespace {

/// This class implements a work queue that uses only the caller's thread to
/// execute work. It spawns no additional threads. However, the work queue
/// itself is thread safe, and addTask and await may be called from any
/// threads.
class SingleThreadWorkQueue : public WorkQueue {
public:
  SingleThreadWorkQueue(CompactRuntimePtr runtimePtr)
      : outerRuntime(CompactRuntimePtr::getCurrentRuntime()) {
    // Associate this thread with the given runtime, possibly overwriting
    // any existing runtime association.
    CompactRuntimePtr::setCurrentRuntime(runtimePtr);
  }

  void shutdown() override;

  ~SingleThreadWorkQueue() override {
    // Note we can't assert state == kShutdown since queue may be created
    // and destroyed without ever being included in a runtime.
    assert(!workItems.dequeue());

    // Restore the association of this thread with the outer runtime, if any.
    CompactRuntimePtr::setCurrentRuntime(outerRuntime);
  }

  void addTask(WorkItem &&workItem, int taskId = -1) override {
    assert(workItem);
#if MODULAR_PARANOID
    assert(state != kShutdown);
    {
      std::lock_guard<std::mutex> guard(mu);
      if (!workItem.use && !useStack.empty()) {
        // Propagate the current use into this work item.
        workItem.use = useStack.back().copy();
      }
    }
#endif
    workItems.enqueue(std::move(workItem));
  }

  void addLocalTask(WorkItem &&workItem) override {
    addTask(std::move(workItem));
  }

  void await(llvm::ArrayRef<AnyAsyncValueRef> values) override;

  bool callerIsForeign() const override { return false; }

#if MODULAR_PARANOID
  void pushDefaultUse(ResourceUse use) override {
    assert(use);
    std::lock_guard<std::mutex> guard(mu);
    useStack.emplace_back(std::move(use));
  }

  void popDefaultUse() override {
    std::lock_guard<std::mutex> guard(mu);
    assert(!useStack.empty());
    useStack.pop_back();
  }

  void taskIsDone() override {
    std::lock_guard<std::mutex> guard(mu);
    if (!useStack.empty())
      useStack.back().reset();
  }
#endif

  size_t getParallelismLevel() const override { return 1; }

private:
  /// Execute blocks of work until stopPredicate is true.
  template <typename StopPredicateFn>
  void runUntil(StopPredicateFn stopPredicate);

  // Execute a single profiled work item.
  void doWork(WorkItem &&workItem) {
#if MODULAR_PARANOID
    {
      // Propagate use
      std::lock_guard<std::mutex> guard(mu);
      useStack.emplace_back(std::move(workItem.use));
    }
#endif

    // Do the work.
    {
      TimeTraceScope scope(AllWorkItemsProfilerEntry::create("llcl.doWork"));
      workItem.task();
    }

#if MODULAR_PARANOID
    {
      // Pop current use. It may already have been reset.
      std::lock_guard<std::mutex> guard(mu);
      assert(!useStack.empty());
      useStack.pop_back();
    }
#endif
  }

#if MODULAR_PARANOID
  enum WorkQueueState : uint8_t {
    kReady = 0,
    kShuttingDown = 1,
    kShutdown = 2
  };

  /// Tracks the state of the queue during shutdown.
  std::atomic<WorkQueueState> state = kReady;
#endif

  /// Pending work items.
  ConcurrentQueue<WorkItem> workItems;
  /// The outer runtime, if any, for the thread using this work queue.
  CompactRuntimePtr outerRuntime;

#if MODULAR_PARANOID
  /// Protects useStack
  std::mutex mu;
  /// Use stack.
  SmallVector<ResourceUse> useStack;
#endif
};
} // namespace

void SingleThreadWorkQueue::shutdown() {
#if MODULAR_PARANOID
  WorkQueueState expected = kReady;
  assert(state.compare_exchange_strong(expected, kShuttingDown));
#endif

  // Complete any work that's still in-flight.
  while (auto workItem = workItems.dequeue()) {
    doWork(std::move(workItem));
  }

#if MODULAR_PARANOID
  expected = kShuttingDown;
  assert(state.compare_exchange_strong(expected, kShutdown));
#endif
}

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values) {
#if MODULAR_PARANOID
  assert(state == kReady);
#endif

  // We are done when values_remaining drops to zero.
  std::atomic<size_t> numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value.andThenSync([&numRemaining]() { numRemaining.fetch_sub(1); });

  if (numRemaining.load() == 0)
    return;

  // Run work items until numRemaining drops to zero.
  runUntil([&]() -> bool { return numRemaining.load() == 0; });

  assert(numRemaining.load() == 0 &&
         "Some AsyncValues are not ready yet no further "
         "tasks are available to run. Are all input AsyncValues ready?");
#if MODULAR_PARANOID
  assert(state != kShutdown);
#endif
}

/// Time to sleep while waiting for work in the work queue.
static const std::chrono::microseconds sleepTime(100);

template <typename StopPredicateFn>
void SingleThreadWorkQueue::runUntil(StopPredicateFn stopPredicate) {
  std::chrono::microseconds totalSlept(0);
  while (true) {
    totalSlept += sleepTime;
    assert(
        totalSlept < std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::seconds(5)) &&
        "SingleThreadWorkQueue has slept for more than 5 seconds while "
        "waiting for callbacks. Some AsyncValues are not ready yet no further "
        "tasks are available to run. Are all input AsyncValues ready?");

    while (auto workItem = workItems.dequeue()) {
      totalSlept = std::chrono::microseconds(0);
      doWork(std::move(workItem));
      if (stopPredicate())
        return;
    }
    // If no work was done, still check if we are done.
    if (stopPredicate())
      return;

    // wait for any callbacks to fire
    std::this_thread::sleep_for(sleepTime);
  }
}

std::unique_ptr<WorkQueue>
M::AsyncRT::createSingleThreadWorkQueue(CompactRuntimePtr runtimePtr) {
  return std::make_unique<SingleThreadWorkQueue>(runtimePtr);
}
