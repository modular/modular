//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a multi-threaded work queue implementation.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Atomics.h"
#include "LLCL/Support/LockFreeRingBuffer.h"
#include "LLCL/Support/Profiling.h"
#include "LLCL/Support/Semaphore.h"
#include "LLCL/Support/SpinWaiter.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"

#define DEBUG_TYPE "llcl_wq"

using namespace M;
using namespace M::LLCL;

/// This value is set to a number for workqueue threads.  Foreign threads always
/// have index #0.
static thread_local size_t workerIDInTLS = 0;

/// Set to true to force waiters triggered by an emplace on a 'foreign'
/// thread to run immediately rather than on an LLCL task. This avoids
/// a thread switch.
constexpr bool kRunImmediatelyOnForeignThreads = false;

/// Bit index i is true if the thread with workedID i is suspended.
using SuspendedThreadsBitvec = uint64_t;
constexpr size_t kMaxWorkers = sizeof(SuspendedThreadsBitvec) * 8;
constexpr SuspendedThreadsBitvec getSuspendedThreadIdMask(size_t workerID) {
  return UINT64_C(1) << workerID;
}

/// Time profiling entries for internal LLCL state changes.
using InternalProfilerEntry =
    TimeTraceProfilerEntry<Trace::EnableTrace(Trace::kLLCL, 2)>;

static constexpr auto printWorkerId(size_t workerID) {
  return [workerID]() {
    return Twine("(workerID:").concat(Twine(workerID)).concat(Twine(")")).str();
  };
}

//===----------------------------------------------------------------------===//
// WorkerThread
//===----------------------------------------------------------------------===//

// Execute a single profiled work item.
static void doWork(ProfiledTaskFunction &&profiledTask, size_t workerID) {
  std::move(profiledTask.waiting).record();
  profiledTask.running =
      profiledTask.running.withDetailSuffix(printWorkerId(workerID));
  profiledTask.work();
  std::move(profiledTask.running).record();
}

namespace {
/// Provides the state needed to synchronize the workers in the thread pool.
struct SharedThreadState {
  static_assert(std::atomic<SuspendedThreadsBitvec>::is_always_lock_free,
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
  AlignedAtomic<SuspendedThreadsBitvec> suspendedThreads;

  /// When a worker is about to go to sleep, it calls this method so andThenSync
  /// can know to wake it up when more work materializes.
  void markSuspended(size_t workerID) {
    // TODO: Does this need to be sequentially consistent?
    suspendedThreads.fetch_or(getSuspendedThreadIdMask(workerID),
                              std::memory_order_seq_cst);
  }

  /// If the specified workerID is suspended, take its bit out of the
  /// suspendedThreads bitset and return true.  Otherwise return false.
  bool takeSuspendedThread(size_t workerID) {
    SuspendedThreadsBitvec workerBit = getSuspendedThreadIdMask(workerID);
    auto oldValue =
        suspendedThreads.fetch_and(~workerBit, std::memory_order_seq_cst);
    return oldValue & workerBit;
  }

  /// If there are any suspended workers, return a non-negative number.
  /// Otherwise return -1.
  int takeAnySuspendedThread() {
    // TODO: Generalize this beyond 64 workers.
    // TODO: Don't use memory_order_seq_cst
    SuspendedThreadsBitvec loadedSuspendedThreads =
        suspendedThreads.load(std::memory_order_seq_cst);
    if (loadedSuspendedThreads == 0)
      return -1;

    // Iteratively compare/xchg to extract the low bit out of suspendedThreads.
    SpinWaiter<> spinner;
    do {
      // Clear the lowest bit set in suspendedThreads with `x & (x-1)` idiom.
      SuspendedThreadsBitvec newSuspendedThreads =
          loadedSuspendedThreads & (loadedSuspendedThreads - 1);

      // Try to atomically swap in the new value.
      if (suspendedThreads.compare_exchange_weak(loadedSuspendedThreads,
                                                 newSuspendedThreads)) {
        // When we succeed, that means we were successful in clearing the
        // lowermost bit.  Map that bit back into a workerID and return it.
        return llvm::countTrailingZeros(loadedSuspendedThreads ^
                                        newSuspendedThreads);
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
  /// Overall state shared by all threads.
  SharedThreadState &sharedState;
  /// 'Small' tasks which can be run on this thread as available.
  /// No synchronization is required here since tasks are added to and
  /// removed only by this thread. Tasks on this list always take precedence
  /// over those in the global task list.
  SmallVector<TaskFunction> localTaskList;
  /// The overall ThreadPoolWorkQueue's task list we can 'steal' tasks from.
  LockFreeRingBuffer<ProfiledTaskFunction> &taskList;
  /// Unique index for this thread. Thread index #0 is reserved for 'foreign'
  /// threads. Though we create an entry for this index in the
  /// ThreadPoolWorkQueue workers it is not running a runOnThread work loop.
  size_t workerID;
  /// The thread id associated with this thread.
  uint64_t threadID;

  /// This is a per-worker semaphore that this blocks on when they run
  /// out of things to do.
  Semaphore sema;

  // We do not construct this for element #0.
  std::optional<std::thread> thread;

  /// Create a `WorkQueueThread` from a sync state reference and a reference to
  /// a task list. This also starts the std::thread, so the sync state and task
  /// list must be initialized by the time this is called.
  WorkQueueThread(SharedThreadState &sharedState,
                  LockFreeRingBuffer<ProfiledTaskFunction> &taskList,
                  size_t workerID)
      : sharedState(sharedState), taskList(taskList), workerID(workerID) {
    if (workerID == 0)
      // Worker #0 is for THE distinguished 'foreign' thread.
      threadID = llvm::get_threadid();
    else
      thread.emplace(&WorkQueueThread::runOnThread, this);
  }

  WorkQueueThread(WorkQueueThread &&other) = default;

  /// Schedule this (presumably small) work item on the localTaskList to be
  /// executed on the next runOnThread loop.
  ///
  /// However if this is the 'forign' thread execute the work item immediately.
  void addOrExecuteLocalTask(TaskFunction &&work);

  /// Joins the thread. Asserts that `sharedState.done` is true because
  /// otherwise the thread will never join.
  void join() {
    assert(sharedState.doneFlag.load() &&
           "Must not destroy a WorkQueueThread object that is not pending "
           "completion.");
    if (thread.has_value())
      thread->join();
  }

  /// This implements the main worker loop, used by both runOnThread and
  /// await. The loop runs until earlyStopPredicate or lateStopPredicate
  /// return true. The "early" predicate is called for every work item that
  /// is executed, and the "late" one is called when waking up from a
  /// suspended state. Tasks are taken from the global queue only if
  /// runNewTasks is true, otherwise only local tasks are run. The given labels
  /// are used only for profiling entries when spinning or sleeping.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItems(EarlyStopPredicateFn earlyStopPredicate,
                LateStopPredicateFn lateStopPredicate, bool runNewTasks,
                StringRef spinningLabel, StringRef sleepingLabel);

  /// The main function invoked by std::thread.
  void runOnThread();
};
} // namespace

void WorkQueueThread::addOrExecuteLocalTask(TaskFunction &&work) {
  if (workerID == 0) {
    LLVM_DEBUG(llvm::dbgs() << "WorkQueueThread: running immediately (emplace "
                               "on foreign thread)\n");
    doWork(ProfiledTaskFunction(std::move(work),
                                /*waiting=*/WorkProfilerEntry(),
                                /*running=*/WorkProfilerEntry("llcl.andThen")),
           workerID);
  } else {
    localTaskList.emplace_back(std::move(work));
  }
}

void WorkQueueThread::runOnThread() {
  assert(workerID != 0 && "The WorkQueueThread representing all 'foreign' "
                          "threads should not be run");

  // Set the current workerID in thread local storage so we can find it later
  // when re-entering.
  workerIDInTLS = workerID;

  // And conversely, capture the system thread id for debugging.
  threadID = llvm::get_threadid();

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  llvm::set_thread_name("LLCL Thread " + llvm::Twine(workerID));

  // Run work items until the system is asked to shut down.
  runItems(
      []() -> bool {  // Fast predicate.
        return false; // Always loop.
      },
      [this]() -> bool { // slowPredicate
        // On wakeup from suspend, check to see if we're supposed to
        // shutdown and stop executing work.
        return sharedState.doneFlag.load(std::memory_order_acquire);
      },
      /*runNewTasks=*/true, "llcl.runOnThread.spinning",
      "llcl.runOnThread.sleeping");
}

template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItems(EarlyStopPredicateFn earlyStopPredicate,
                               LateStopPredicateFn lateStopPredicate,
                               bool runNewTasks, StringRef spinningLabel,
                               StringRef sleepingLabel) {
  // Continuously execute work units until the stopPredicate returns true.
KeepRunning:
  while (!earlyStopPredicate()) {
    // Prefer to run local work items as soon as they are available.
    // NOTE: the list may grow as we do work.
    for (size_t i = 0; i < localTaskList.size() /* not const */; ++i)
      doWork(ProfiledTaskFunction(
                 std::move(localTaskList[i]), /*waiting=*/WorkProfilerEntry(),
                 /*running=*/WorkProfilerEntry("llcl.andThen")),
             workerID);
    localTaskList.clear();

    if (runNewTasks) {
      // In the normal case we happily pick up and do work.
      if (auto labelledTask = taskList.dequeue()) {
        doWork(std::move(labelledTask), workerID);
        continue;
      }
    }

    {
      TimeTraceScope scope(
          InternalProfilerEntry(spinningLabel, printWorkerId(workerID)));

      // If we've run out of work to do, we need to quiesce and ultimately block
      // in the kernel on the semaphore.  However, we don't want to immediately
      // give up hope, because we may be "right about to" get new work incoming.
      // We also want to make sure to use exponential backoff to avoid pummeling
      // the memory hierarchy of the threads that are doing useful work.  As
      // such, we use a BusyWaitSpinWaiter.
      BusyWaitSpinWaiter spinWaiter(sharedState.busyWaitNs);

      // Spin until we find some work to do.
      while (!spinWaiter.wait()) {
        if (runNewTasks) {
          // If we ever succeed in finding work to do, go back to running like
          // normal.
          if (auto work = taskList.dequeue()) {
            doWork(std::move(work), workerID);
            goto KeepRunning;
          }
        }

        // If we're spinning and the early or the late stop condition happens,
        // then we're done.  Checking the late stop condition here make sure our
        // threads shut down promptly when a runtime is torn down.
        if (earlyStopPredicate() || lateStopPredicate()) {
          return;
        }
      }
    }

    // Otherwise, we we've waited long enough, yield the thread to the OS so we
    // don't burn power and starve other tasks on the system.
    sharedState.markSuspended(workerID);

    // Double check the fast predicate after marking ourselves as suspended
    // (which only matters for await()).  Await won't signal the waiter unless
    // it sees it at the right time.
    if (earlyStopPredicate()) {
      sharedState.takeSuspendedThread(workerID);
      return;
    }

    {
      TimeTraceScope scope(
          InternalProfilerEntry(sleepingLabel, printWorkerId(workerID)));

      // Ok, finally block.
      sema.wait();
    }

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

  void addTask(TaskFunction &&work, WorkProfilerEntry &&profilerEntry) override;

  void addOrExecuteSmallTask(TaskFunction work) override;

  void await(ArrayRef<AnyAsyncValueRef> values, bool runNewTasks) override;

  size_t getParallelismLevel() const final {
    // `numWorkers` is set to the number of worker threads that are created by
    // the work queue +1 for a foreign thread.
    // TODO(#1903): This is a poor heuristic for subdividing work.
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
  LockFreeRingBuffer<ProfiledTaskFunction> taskList;
};
} // namespace

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numWorkers,
                                         std::chrono::nanoseconds busyWaitNs,
                                         bool profilingEnabled)
    : numWorkers(numWorkers), sharedState{busyWaitNs, profilingEnabled,
                                          /*doneFlag=*/false,
                                          /*suspendedThreads=*/0} {
  assert(numWorkers <= kMaxWorkers && "Too many workers for bitvec width");
  workers.reserve(numWorkers);
  // Initialize each thread with its required state.  Note that  thread #0
  // does not start itself: that index is reserved for foreign threads.
  for (size_t i = 0; i < numWorkers; ++i)
    workers.emplace_back(sharedState, taskList, i);
}

void ThreadPoolWorkQueue::shutdown() {
  TimeTraceScope scope(InternalProfilerEntry("llcl.shutdown"));
  size_t workerID = workerIDInTLS;

  // Donate this thread to help drain the work queue if there's anything left.
  while (auto labelledTask = taskList.dequeue())
    doWork(std::move(labelledTask), workerID);

  // Now we can tell all the threads to exit.
  sharedState.doneFlag.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake up if it is waiting.
  for (auto &worker : workers)
    worker.sema.post();

  // Mark no threads as suspended, even though they may not have woken up,
  // cleared their own bit and exited yet.  This ensures that any in-flight
  // andThenSync calls won't try to wake these threads as we start joining and
  // tearing them down.
  sharedState.suspendedThreads.store(0);

  // Join all the threads when they shut down cleanly.
  for (auto &worker : workers)
    worker.join();
}

void ThreadPoolWorkQueue::addTask(TaskFunction &&work,
                                  WorkProfilerEntry &&profilerEntry) {
  size_t workerID = workerIDInTLS;
  (void)workerID;
  // Try to add this work to the RingBuffer.
  WorkProfilerEntry waitingEntry =
      profilerEntry.withNameSuffix(".waiting")
          .withDetailSuffix(printWorkerId(workerID)); // restarts clock
  ProfiledTaskFunction profiledTask(std::move(work), std::move(waitingEntry),
                                    std::move(profilerEntry));
  if (taskList.enqueue(profiledTask)) {
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
  LLVM_DEBUG(
      llvm::dbgs()
      << "ThreadPoolWorkQueue: running immediately (task queue is full)\n");
  doWork(std::move(profiledTask), workerID);
}

void ThreadPoolWorkQueue::addOrExecuteSmallTask(M::LLCL::TaskFunction work) {
  size_t workerID = workerIDInTLS;
  if (!kRunImmediatelyOnForeignThreads && workerID == 0)
    addTask(std::move(work), WorkProfilerEntry("llcl.andThen"));
  else
    workers[workerID].addOrExecuteLocalTask(std::move(work));
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values,
                                bool runNewTasks) {
  // If all the values are ready, then we don't have to do anything.
  if (llvm::all_of(values, [](auto &av) { return av.isReady(); }))
    return;

  // Figure out which WorkerThread this is being invoked from.  This could be
  // something in the WorkQueue or could be an external foreign thread (index
  // #0).
  size_t workerID = workerIDInTLS;
  WorkQueueThread *thisWorker = &workers[workerID];

  if (thisWorker->threadID != llvm::get_threadid()) {
    llvm::errs() << "ThreadPoolWorkQueue::await: calling await from thread "
                 << llvm::get_threadid() << ", where as worker " << workerID
                 << " was created for thread " << thisWorker->threadID << "\n";
    assert(false && "invoking await from unrecognized foreign thread");
  }

  // For now, make sure there's only one 'foreign' thread.

  // We are done when numRemaining drops to zero.
  std::atomic<ssize_t> numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.  When done,
  // we signal the semaphore for this worker to make sure to wake it up if it
  // fell asleep.
  for (auto &value : values)
    value.andThenSync([&numRemaining, thisWorker, this]() {
      // Decrement the count of async values that we're waiting on.
      // TODO: This can probably use more relaxed memory consistency!
      if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
        return;

      // Get the thread ID of the thread running the andThenSync, for tracing.
      size_t workerID = workerIDInTLS;
      (void)workerID;

      // When it drops to zero, we're good to go and whatever thread is waiting
      // for this will exit out of its 'runItems' loop.  That said, the thread
      // may be suspended on a semaphore.  Check for this, and if so, signal its
      // semaphore so it wakes up and notes that it is done.
      auto awaitingWorkerID = thisWorker->workerID;

      // If the worker doing the await() has suspended, make sure to wake it up
      // so it notices that it is done.
      if (sharedState.takeSuspendedThread(awaitingWorkerID)) {
        // NOTE: We may post without a corresponding wait in
        // WorkQueueThread::runItems if the earlyStopPredicate &
        // takeSuspendedThread path executes just after our takeSuspendedThread
        // above. In that case a future wait will just go around the work loop
        // again.
        //
        // NOTE: This wakes up exactly one sleeping thread, but (in the case of
        // foreign threads) it is possible we have multiple threads blocked on
        // it, so the semaphore could be (e.g.) at -3 or something. If/when
        // we care about this, we can keep track of the number of foreign
        // threads we've seen and post that many times.
        thisWorker->sema.post();
      }
    });

  // Run work items until the system is asked to shut down.
  thisWorker->runItems(
      [&numRemaining]() -> bool { // Early predicate.
        // Exit early as soon as numRemaining drops to zero.
        // TODO: Relaxed memory consistency!
        return numRemaining.load(std::memory_order_seq_cst) == 0;
      },
      []() -> bool { // Late Predicate
        // No additional shutdown check after waking, the early
        // check will suffice.
        return false;
      },
      runNewTasks,
      runNewTasks ? "llcl.await.spinning" : "llcl.awaitQuietly.spinning",
      runNewTasks ? "llcl.await.sleeping" : "llcl.awaitQuietly.sleeping");

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
  numThreads = std::min(numThreads, kMaxWorkers);
  return std::make_unique<ThreadPoolWorkQueue>(numThreads, busyWait,
                                               profilingEnabled);
}
