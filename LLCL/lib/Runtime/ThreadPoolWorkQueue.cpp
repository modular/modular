//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a multi-threaded work queue implementation.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Atomics.h"
#include "LLCL/Support/LockFreeRingBuffer.h"
#include "LLCL/Support/Profiling.h"
#include "LLCL/Support/Semaphore.h"
#include "LLCL/Support/SpinWaiter.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Threading.h"

using namespace M::LLCL;
using llvm::ArrayRef;

/// This value is set to a number for workqueue threads.  Foreign threads always
/// have index #0.
static thread_local ssize_t workerIDInTLS = 0;

//===----------------------------------------------------------------------===//
// WorkerThread
//===----------------------------------------------------------------------===//

// Execute a single work item with tracing support.
static void doWork(TaskFunction &workFn, size_t workerID) {
  TIME_PROFILER_SCOPE(1, "doWork");
  workFn();
}

namespace {
/// Provides the state needed to synchronize the workers in the thread pool.
struct SharedThreadState {
  static_assert(std::atomic<uint64_t>::is_always_lock_free,
                "suspendedThreads should always be lock free");
  /// This is the time to spin wait before falling asleep.
  const std::chrono::nanoseconds busyWaitNs;

  /// True if each thread should establish and teardown time profiling.
  const bool profilingEnabled;

  /// This flag indicates when a thread should quit working and get ready to be
  /// joined.
  std::atomic<bool> doneFlag;

  /// This keeps a bitset of suspended threads, indexed by workerID.  This will
  /// thrash around a lot when the workqueue is close to empty and threads are
  /// starting and stopping themselves, but should stay zero and read-only when
  /// there is a lot of work to do.
  ///
  /// This is aligned because the state above is immutable or (in the case of
  /// doneFlag) almost never changing. We don't want doneFlag to be on the same
  /// cache line as suspendedThreads.
  AlignedAtomic<uint64_t> suspendedThreads;

  /// When a worker is about to go to sleep, it calls this method so andThen can
  /// know to wake it up when more work materializes.
  void markSuspended(unsigned workerID) {
    // TODO: Does this need to be sequentially consistent?
    suspendedThreads.fetch_or(UINT64_C(1) << workerID,
                              std::memory_order_seq_cst);
  }

  /// If the specified workerID is suspended, take its bit out of the
  /// suspendedThreads bitset and return true.  Otherwise return false.
  bool takeSuspendedThread(unsigned workerID) {
    uint64_t workerBit = UINT64_C(1) << workerID;
    auto oldValue =
        suspendedThreads.fetch_and(~workerBit, std::memory_order_seq_cst);
    return oldValue & workerBit;
  }

  /// If there are any suspended workers, return a non-negative number.
  /// Otherwise return -1.
  int takeAnySuspendedThread() {
    // TODO: Generalize this beyond 64 workers.
    // TODO: Don't use memory_order_seq_cst
    uint64_t loadedSuspendedThreads =
        suspendedThreads.load(std::memory_order_seq_cst);
    if (loadedSuspendedThreads == 0)
      return -1;

    // Iteratively compare/xchg to extract the low bit out of suspendedThreads.
    SpinWaiter<> spinner;
    do {
      // Clear the lowest bit set in suspendedThreads with `x & (x-1)` idiom.
      uint64_t newSuspendedThreads =
          loadedSuspendedThreads & (loadedSuspendedThreads - 1);

      // Try to atomically swap in the new value.
      if (suspendedThreads.compare_exchange_weak(loadedSuspendedThreads,
                                                 newSuspendedThreads)) {
        // When we succeed, that means we were successful in clearing the
        // lowermost bit.  Map that bit back into a workerID and return it.
        return llvm::countTrailingZeros(
            loadedSuspendedThreads ^ newSuspendedThreads,
            /*SuspendedThreads is not zero: */ llvm::ZB_Undefined);
      }

      spinner.wait();
    } while (loadedSuspendedThreads != 0);

    // We saw a candidate but it fell away.
    return -1;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// WorkQueueThread
//===----------------------------------------------------------------------===//

namespace {
/// Wrapper around an std::thread which is created for one instance of each
/// worker thread.
struct WorkQueueThread {
  SharedThreadState &sharedState;
  LockFreeRingBuffer<TaskFunction> &taskList;
  size_t workerID;

  /// This is a per-worker semaphore that this blocks on when they run
  /// out of things to do.
  Semaphore sema;

  // We do not construct this for element #0.
  std::optional<std::thread> thread;

  /// Create a `WorkQueueThread` from a sync state reference and a reference to
  /// a task list. This also starts the std::thread, so the sync state and task
  /// list must be initialized by the time this is called.
  WorkQueueThread(SharedThreadState &sharedState,
                  LockFreeRingBuffer<TaskFunction> &taskList, size_t workerID)
      : sharedState(sharedState), taskList(taskList), workerID(workerID) {

    if (workerID != 0)
      thread.emplace(&WorkQueueThread::runOnThread, this);
  }

  WorkQueueThread(WorkQueueThread &&other) = default;

  /// Joins the thread. Asserts that `sharedState.done` is true because
  /// otherwise the thread will never join.
  void join() {
    assert(sharedState.doneFlag.load() &&
           "Must not destroy a WorkQueueThread object that is not pending "
           "completion.");
    if (thread.has_value())
      thread->join();
  }

  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItems(bool isAwait, EarlyStopPredicateFn earlyStopPredicate,
                LateStopPredicateFn lateStopPredicate);

  /// The main function invoked by std::thread.
  void runOnThread();
};
} // namespace

void WorkQueueThread::runOnThread() {
  if (sharedState.profilingEnabled) {
    TIME_PROFILER_WORKER_INIT;
  }

  // Set the current workerID in thread local storage so we can find it later
  // when re-entering.
  workerIDInTLS = workerID;

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  llvm::set_thread_name("LLCL Thread " + llvm::Twine(workerID));

  TIME_PROFILER_BEGIN(4, "runOnThread", "");

  // Run work items until the system is asked to shut down.
  runItems(/*isAwait*/ false,
           []() -> bool {  // Fast predicate.
             return false; // Always loop.
           },
           [&]() -> bool { // slowPredicate
             // On wakeup from suspend, check to see if we're supposed to
             // shutdown and stop executing work.
             return sharedState.doneFlag.load(std::memory_order_acquire);
           });

  TIME_PROFILER_END(4);

  if (sharedState.profilingEnabled) {
    TIME_PROFILER_WORKER_WRAPUP;
  }
}

/// This method iteratively runs work items until either of the specified
/// predicates returns true.  The "early" predicate is called for every work
/// item that is executed, and the "late" one is called when waking up from
/// a suspended state.
template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItems(bool isAwait,
                               EarlyStopPredicateFn earlyStopPredicate,
                               LateStopPredicateFn lateStopPredicate) {
  // Continuously execute work units until the stopPredicate returns true.
KeepRunning:
  while (!earlyStopPredicate()) {
    // In the normal case we happily pick up and do work.
    if (auto work = taskList.dequeue()) {
      doWork(work, workerID);
      continue;
    }

    TIME_PROFILER_BEGIN(3, "spinning", isAwait ? "await thread" : "worker");

    // If we've run out of work to do, we need to quiesce and ultimately block
    // in the kernel on the semaphore.  However, we don't want to immediately
    // give up hope, because we may be "right about to" get new work incoming.
    // We also want to make sure to use exponential backoff to avoid pummeling
    // the memory hierarchy of the threads that are doing useful work.  As
    // such, we use a BusyWaitSpinWaiter.
    BusyWaitSpinWaiter spinWaiter(sharedState.busyWaitNs);

    // Spin until we find some work to do.
    while (!spinWaiter.wait()) {
      // If we ever succeed in finding work to do, go back to running like
      // normal.
      if (auto work = taskList.dequeue()) {
        TIME_PROFILER_END(3);
        doWork(work, workerID);
        goto KeepRunning;
      }

      // If we're spinning and the early or the late stop condition happens,
      // then we're done.  Checking the late stop condition here make sure our
      // threads shut down promptly when a runtime is torn down.
      if (earlyStopPredicate() || lateStopPredicate()) {
        TIME_PROFILER_END(3);
        return;
      }
    }

    TIME_PROFILER_END(3);
    TIME_PROFILER_SCOPE(3, (isAwait ? "await thread" : "worker") +
                               std::string(" sleeping"));

    // Otherwise, we we've waited long enough, yield the thread to the OS so we
    // don't burn power and starve other tasks on the system.
    sharedState.markSuspended(workerID);

    // Double check the fast predicate after marking ourselves as suspended (
    // which only matters for await()).  Await won't signal the waiter unless
    // it sees it at the right time.
    if (earlyStopPredicate()) {
      sharedState.takeSuspendedThread(workerID);
      return;
    }

    // Ok, finally block.
    sema.wait();

    // On wakeup, check the 'slow' predicate to see if we should stop (this is
    // how worker threads know to exit).  The early predicate is checked as part
    // of the outer while loop immediately after this.
    if (lateStopPredicate())
      return;
  }
}

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue
//===----------------------------------------------------------------------===//

namespace {

/// This class provides a thread-pool that implements the WorkQueue interface.
/// It starts a dynamic number of threads and distributes work to it by means
/// of a concurrent-safe queue.
class ThreadPoolWorkQueue : public WorkQueue {
public:
  /// Initialize the thread pool and start up the worker threads. By the time
  /// the constructor finishes, all the worker threads have started and shall
  /// only be cancelled by the destructor.
  ThreadPoolWorkQueue(size_t numWorkers, std::chrono::nanoseconds busyWaitNs,
                      bool profilingEnabled);

  void shutdown() override;
  ~ThreadPoolWorkQueue() override = default;

  void addTask(TaskFunction work) override;

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  size_t getParallelismLevel() const final {
    // `numWorkers` is set to the number of worker threads that are created by
    // the work queue +1 for a foreign thread.
    // TODO: This isn't actually correct.  See PR1903:
    // https://github.com/modularml/modular/issues/1903
    return numWorkers;
  }

private:
  /// This is the set of worker threads in the WorkQueue.  Note that we reserve
  /// entry #0 for foreign threads that may get donated to this queue.  That
  /// means that we never start worker #0.
  const size_t numWorkers;
  std::vector<WorkQueueThread> workers;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  SharedThreadState sharedState;

  ///  This is the ringbuffer of work to do.
  LockFreeRingBuffer<TaskFunction> taskList;
};
} // namespace

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numWorkers,
                                         std::chrono::nanoseconds busyWaitNs,
                                         bool profilingEnabled)
    : numWorkers(numWorkers), sharedState{busyWaitNs, profilingEnabled,
                                          /*doneFlag=*/false,
                                          /*suspendedThreads=*/0} {
  workers.reserve(numWorkers);
  // Initialize each thread with its required state.  Note that  thread #0
  // does not start itself: that index is reserved for foreign threads.
  for (size_t i = 0; i < numWorkers; ++i)
    workers.emplace_back(sharedState, taskList, i);
}

void ThreadPoolWorkQueue::shutdown() {
  TIME_PROFILER_SCOPE(4, "shutdown");
  int workerID = workerIDInTLS;

  // Donate this thread to help drain the work queue if there's anything left.
  while (auto work = taskList.dequeue())
    doWork(work, workerID);

  // Now we can tell all the threads to exit.
  sharedState.doneFlag.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake up if it is waiting.
  for (auto &worker : workers)
    worker.sema.post();

  // Mark no threads as suspended, even though they may not have woken up,
  // cleared their own bit and exited yet.  This ensures that any in-flight
  // andThen calls won't try to wake these threads as we start joining and
  // tearing them down.
  sharedState.suspendedThreads.store(0);

  // Join all the threads when they shut down cleanly.
  for (auto &worker : workers)
    worker.join();
}

void ThreadPoolWorkQueue::addTask(TaskFunction work) {
  auto workerID = workerIDInTLS;
  (void)workerID;
  // Try to add this work to the RingBuffer.
  TIME_PROFILER_SCOPE(2, "addTask");
  if (taskList.enqueue(work)) {
    // If there are any suspended workers, kick one of them now that there is
    // new work to do.
    int workerToPoke = sharedState.takeAnySuspendedThread();
    if (workerToPoke != -1) {
      assert(workerToPoke < int(numWorkers));
      workers[workerToPoke].sema.post();
    }
    return;
  }

  // If we failed to add it, then the ring buffer is full: just run the work
  // item locally on the current stack.
  // NOTE: This runs the risk of stack overflow, but we don't have a choice.
  doWork(work, workerID);
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
  // If all the values are ready, then we don't have to do anything.
  if (llvm::all_of(values, [](auto &av) { return av->isReady(); }))
    return;
  TIME_PROFILER_SCOPE(2, "await");

  // Figure out which WorkerThread this is being invoked from.  This could be
  // something in the WorkQueue or could be an external foreign thread (index
  // #0).
  ssize_t workerID = workerIDInTLS;
  WorkQueueThread *thisWorker = &workers[workerID];

  // We are done when values_remaining drops to zero.
  std::atomic<ssize_t> numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.  When done,
  // we signal the semaphore for this worker to make sure to wake it up if it
  // fell asleep.
  for (auto &value : values)
    value->andThen([&numRemaining, thisWorker, this]() {
      TIME_PROFILER_SCOPE(3, "await andThen");
      // Decremenet the count of async values that we're waiting on.
      // TODO: This can probably use more relaxed memory consistency!
      if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
        return;

      // Get the thread ID of the thread running the andThen, for tracing.
      auto workerID = workerIDInTLS;
      (void)workerID;

      // When it drops to zero, we're good to go and whatever thread is waiting
      // for this will exit out of its 'runItems' loop.  That said, the thread
      // may be suspended on a semaphore.  Check for this, and if so, signal its
      // semaphore so it wakes up and notes that it is done.
      auto awaitingWorkerID = thisWorker->workerID;

      // If the worker doing the await() has suspended, make sure to wake it up
      // so it notices that it is done.
      if (sharedState.takeSuspendedThread(awaitingWorkerID)) {
        // NOTE: This wakes up exactly one sleeping thread, but (in the case of
        // foreign threads) it is possible we have multiple threads blocked on
        // it, so the semaphore could be (e.g.) at -3 or something.    If/when
        // we care about this, we can keep track of the number of foreign
        // threads we've seen and post that many times.
        thisWorker->sema.post();
      }
    });

  // Run work items until the system is asked to shut down.
  thisWorker->runItems(/*isAwait*/ true,
                       [&]() -> bool { // Early predicate.
                         // Exit early as soon as numRemaining drops to zero.
                         // TODO: Relaxed memory consistency!
                         return numRemaining.load(std::memory_order_seq_cst) ==
                                0;
                       },
                       []() -> bool { // Late Predicate
                         // No additional shutdown check after waking, the early
                         // check will suffice.
                         return false;
                       });

  assert(numRemaining.load() == 0);
}

//===----------------------------------------------------------------------===//
// createThreadPoolWorkQueue entrypoint
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue>
M::LLCL::createThreadPoolWorkQueue(size_t numThreads,
                                   std::chrono::nanoseconds busyWait,
                                   bool profilingEnabled) {
  // We expect `numThreads` to be the total numbers of threads that are
  // accessing the work queue. As there will be an external thread that will
  // access the work queue and take items from it by calling `await`, we create
  // `numThreads - 1` worker threads from the thread pool work queue.
  assert(numThreads > 0);

  // We use a 64-bit value "thread suspended" value currently so we cap at 64
  // threads.  This algorithm isn't going to scale beyond 64 threads anyway.
  numThreads = std::min(numThreads, size_t(64));
  return std::make_unique<ThreadPoolWorkQueue>(numThreads, busyWait,
                                               profilingEnabled);
}
