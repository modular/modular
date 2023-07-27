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

/// This class implements a work queue that uses only the caller's thread to
/// execute work. It spawns no additional threads. However, the work queue
/// itself is thread safe, and addTask and await may be called from any
/// threads.
class SingleThreadWorkQueue : public WorkQueue {
public:
  SingleThreadWorkQueue(size_t cpuID) : cpuID(cpuID) {}

  void shutdown() override {
#if MODULAR_PARANOID
    WorkQueueState expected = kReady;
    assert(state.compare_exchange_strong(expected, kShuttingDown));
#endif
    // Complete any work that's still in-flight.
    runUntil([]() -> bool { return false; });
#if MODULAR_PARANOID
    expected = kShuttingDown;
    assert(state.compare_exchange_strong(expected, kShutdown));
#endif
  }

  ~SingleThreadWorkQueue() override {
    // Note we can't assert state == kShutdown since queue may be created
    // and destroyed without ever being included in a runtime.
    assert(!workItems.dequeue());
  }

  void addTask(WorkItem &&workItem) override {
    assert(workItem);
#if MODULAR_PARANOID
    assert(state != kShutdown);
    {
      std::lock_guard<std::mutex> guard(mu);
      if (!workItem.lifetime && !activeLifetimes.empty()) {
        // Propagate the current lifetime (if any) onto this work item.
        workItem.lifetime = activeLifetimes.back().copy();
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
  void pushDefaultLifetime(LifetimeRef lifetime) override {
    assert(lifetime);
    std::lock_guard<std::mutex> guard(mu);
    activeLifetimes.emplace_back(std::move(lifetime));
  }

  void popDefaultLifetime(LifetimeRef lifetime) override {
    assert(lifetime);
    std::lock_guard<std::mutex> guard(mu);
    assert(!activeLifetimes.empty());
    assert(activeLifetimes.back().getPointer() == lifetime.getPointer());
    activeLifetimes.pop_back();
  }
#endif

  size_t getParallelismLevel() const override { return 1; }

private:
  /// Execute blocks of work until stopPredicate is true, setting thread
  /// affinity if requested.
  template <typename StopPredicateFn>
  void runUntil(StopPredicateFn stopPredicate);

  /// Actually run work items.
  template <typename StopPredicateFn>
  void runUntilImpl(StopPredicateFn stopPredicate);

  // Execute a single profiled work item.
  void doWork(WorkItem &&workItem) {
#if MODULAR_PARANOID
    {
      std::lock_guard<std::mutex> guard(mu);
      activeLifetimes.emplace_back(std::move(workItem.lifetime));
      assert(!activeLifetimes.back() ||
             activeLifetimes.back()->isActive() &&
                 "starting a work item after its lifetime has ended");
    }
#endif

    {
      TimeTraceScope scope(AllWorkItemsProfilerEntry::create("llcl.doWork"));
      // Do the work.
      workItem.task();
    }

#if MODULAR_PARANOID
    {
      std::lock_guard<std::mutex> guard(mu);
      assert(!activeLifetimes.back() ||
             activeLifetimes.back()->isActive() &&
                 "lifetime was ended while a work item was in flight");
      assert(!activeLifetimes.empty());
      activeLifetimes.pop_back();
    }
#endif
  }

  /// CPU ID to set affinity to when running the runUntil loop.
  size_t cpuID;

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

#if MODULAR_PARANOID
  /// Protects activeLifetimes.
  std::mutex mu;
  /// Lifetime stack.
  SmallVector<LifetimeRef> activeLifetimes;
#endif
};
} // namespace

void SingleThreadWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values) {
#if MODULAR_PARANOID
  assert(state == kReady);
#endif

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
#if MODULAR_PARANOID
  assert(state != kShutdown);
#endif
}

template <typename StopPredicateFn>
void SingleThreadWorkQueue::runUntil(StopPredicateFn stopPredicate) {
  LLCL::runWithThreadAffinity(cpuID, [&]() { runUntilImpl(stopPredicate); });
}

template <typename StopPredicateFn>
void SingleThreadWorkQueue::runUntilImpl(StopPredicateFn stopPredicate) {
  while (auto workItem = workItems.dequeue()) {
    doWork(std::move(workItem));
    if (stopPredicate())
      break;
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
