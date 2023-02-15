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

//
// Terminology:
//  - Worker thread: a thread we create which is running a dedicated runItems
//    loop. These threads have a 'workerID' > 0.
//  - Foreign thread: any thread which constructs the ThreadPoolWorkQueue, or
//    calls any methods upon it. The same ThreadPoolWorkQueue may see calls
//    from multiple foreign threads. All foreign threads have the 'workerID' 0.
//  - Awaiting foreign thread: a distinguished foreign thread currently running
//    a runItems loop within an await. At most one awaiting foreign thread can
//    be active at a time, however different threads can call await
//    sequentially.
//

/// Set to true to force waiters triggered by an emplace from a 'foreign'
/// thread not running an await loop to run immediately. Otherwise they will
/// be enqued as an LLCL task (unless the task queue is full).
constexpr bool kRunImmediatelyOnForeignThreads = false;

/// Bit index i is true if the thread with workedID i is suspended.
using SuspendedThreadsBitvec = uint64_t;
constexpr size_t kMaxWorkers = sizeof(SuspendedThreadsBitvec) * 8;
constexpr SuspendedThreadsBitvec getSuspendedThreadIdMask(size_t workerID) {
  return UINT64_C(1) << workerID;
}

/// Type of profiling entries for recording internal LLCL state changes.
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

/// The index of the current thread within the WorkQueueThread workers
/// vector. However all 'foreign' threads have index 0, and only
/// worker threads will have a non-zero index.
static thread_local size_t workerIDInTLS = 0;

/// Wrapper around an std::thread created for each worker thread, along
/// with one to represent all 'foreign' threads.
struct WorkQueueThread {
  /// Overall state shared by all threads.
  SharedThreadState &sharedState;
  /// 'Local' tasks which can be run on this thread as they become available.
  /// No threading synchronization is required here since tasks are added to
  /// and removed only by the unique thread (currently) tied to this object.
  /// However, we do need to protect against runItems being called recursively.
  ///
  /// Tasks on this list always take precedence over those in the global task
  /// list.
  SmallVector<TaskFunction, 6> localTaskList;
  /// The overall ThreadPoolWorkQueue's task list we can take tasks from.
  LockFreeRingBuffer<ProfiledTaskFunction> &taskList;
  /// Unique index for this thread.
  ///
  /// Thread index #0 is reserved for all 'foreign' threads. Though we create
  /// an entry for this index in the ThreadPoolWorkQueue workers it is not
  /// running a runOnThread work loop. However, the thread may call runItems
  /// while awaiting.
  size_t workerID;
  /// This is a per-worker semaphore that this blocks on when they run
  /// out of things to do.
  Semaphore sema;
  /// The system id for the 'foreign' thread which is executing runItems using
  /// this WorkQueueThread, or none for worker threads.
  std::atomic<uint64_t> threadID = 0;
  // The underlying worker thread, or none for the 'foreign' threads
  // WorkQueueThread.
  std::optional<std::thread> thread;

  /// Create a `WorkQueueThread` from a sync state reference and a reference to
  /// a task list. This also starts the std::thread, so the sync state and task
  /// list must be initialized by the time this is called.
  WorkQueueThread(SharedThreadState &sharedState,
                  LockFreeRingBuffer<ProfiledTaskFunction> &taskList,
                  size_t workerID)
      : sharedState(sharedState), taskList(taskList), workerID(workerID) {
    if (workerID > 0)
      thread.emplace(&WorkQueueThread::runOnThread, this);
  }

  /// Schedule this work item on the localTaskList to be executed on the next
  /// runItems loop.
  ///
  /// For the 'foreign' thread this item won't be executed until await is
  /// called. All other threads will pick the item up on their next work loop.
  void addLocalTask(TaskFunction &&work) {
    localTaskList.emplace_back(std::move(work));
  }

  /// Joins the thread. Asserts that `sharedState.done` is true because
  /// otherwise the thread will never join.
  void join() {
    assert(sharedState.doneFlag.load() &&
           "Must not destroy a WorkQueueThread object that is not pending "
           "completion.");
    if (thread.has_value())
      thread->join();
  }

  /// This implements the main worker loop, used by runOnThread, await and
  /// shutdown. The loop runs until earlyStopPredicate or lateStopPredicate
  /// return true. The "early" predicate is called for every work item that
  /// is executed, and the "late" one is called when waking up from a
  /// suspended state.
  ///
  /// Tasks are taken from the global queue only if runNewTasks is true,
  /// otherwise only local tasks are run.
  ///
  /// The loop will busy wait or sleep waiting for new tasks only if
  /// waitForTasks is true, otherwise the loop will exit once the work queue
  /// and local task list is empty.
  ///
  /// The given labels are used only for profiling entries when spinning or
  /// sleeping.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItems(EarlyStopPredicateFn earlyStopPredicate,
                LateStopPredicateFn lateStopPredicate, bool runNewTasks,
                bool waitForTasks, StringRef spinningLabel,
                StringRef sleepingLabel);

  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                    LateStopPredicateFn lateStopPredicate, bool runNewTasks,
                    bool waitForTasks, StringRef spinningLabel,
                    StringRef sleepingLabel);

  /// The main function invoked by std::thread.
  void runOnThread();
};
} // namespace

void WorkQueueThread::runOnThread() {
  assert(workerID != 0 && "The WorkQueueThread representing all 'foreign' "
                          "threads should not be run");

  // Set the current workerID in thread local storage so we can find it later
  // when re-entering.
  workerIDInTLS = workerID;

  // Though not needed for interlock, capture the worker's system thread id for
  // debugging.
  threadID = llvm::get_threadid();

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  llvm::set_thread_name("LLCL Thread " + llvm::Twine(workerID));

  // Run work items until the system is asked to shut down.
  runItems(
      /*earlyStopPredicate=*/
      []() -> bool {
        return false; // Always loop.
      },
      /*lateStopPredicate=*/
      [this]() -> bool {
        // On wakeup from suspend, check to see if we're supposed to
        // shutdown and stop executing work.
        return sharedState.doneFlag.load(std::memory_order_acquire);
      },
      /*runNewTasks=*/true,
      /*waitForTasks=*/true,
      /*spinningLabel=*/"llcl.runOnThread.spinning",
      /*sleepingLabel=*/"llcl.runOnThread.sleeping");
}

template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItems(EarlyStopPredicateFn earlyStopPredicate,
                               LateStopPredicateFn lateStopPredicate,
                               bool runNewTasks, bool waitForTasks,
                               StringRef spinningLabel,
                               StringRef sleepingLabel) {
  if (workerID == 0) {
    uint64_t callerThreadID = llvm::get_threadid();
    uint64_t expectedThreadID = 0;
    if (threadID.compare_exchange_strong(expectedThreadID, callerThreadID)) {
      // We're the only foreign thread running a runItems loop.
      runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
          earlyStopPredicate, lateStopPredicate, runNewTasks, waitForTasks,
          spinningLabel, sleepingLabel);
      // Release.
      threadID = 0;
    } else if (expectedThreadID == callerThreadID) {
      // This is a recursive call to await from the same foreign thread.
      runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
          earlyStopPredicate, lateStopPredicate, runNewTasks, waitForTasks,
          spinningLabel, sleepingLabel);

    } else {
      llvm::report_fatal_error(Twine("Attempting to await from foreign thread ")
                                   .concat(Twine(callerThreadID))
                                   .concat(", however thread ")
                                   .concat(Twine(expectedThreadID))
                                   .concat(" is already running an await"));
    }
  } else {
    runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
        earlyStopPredicate, lateStopPredicate, runNewTasks, waitForTasks,
        spinningLabel, sleepingLabel);
  }
}

template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                                   LateStopPredicateFn lateStopPredicate,
                                   bool runNewTasks, bool waitForTasks,
                                   StringRef spinningLabel,
                                   StringRef sleepingLabel) {
  // Continuously execute work units until the stopPredicate returns true.
KeepRunning:
  while (!earlyStopPredicate()) {
    // Prefer to run local work items as soon as they are available.
    // CAUTION: a work function may add to this list, and may even invoke
    // runItems recursively.
    while (!localTaskList.empty()) {
      ProfiledTaskFunction labelledTask(
          std::move(localTaskList.back()), /*waiting=*/WorkProfilerEntry(),
          /*running=*/WorkProfilerEntry("llcl.waiter"));
      localTaskList.erase(localTaskList.end() - 1);
      doWork(std::move(labelledTask), workerID);
    }

    if (runNewTasks) {
      // In the normal case we happily pick up and do work.
      if (auto labelledTask = taskList.dequeue()) {
        doWork(std::move(labelledTask), workerID);
        continue;
      }
    }

    if (!waitForTasks)
      return;

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

    // Otherwise, we've waited long enough, yield the thread to the OS so we
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
                      size_t taskListCapacity);

  ~ThreadPoolWorkQueue() override;

  void shutdown() override;

  void addTask(TaskFunction &&work, WorkProfilerEntry &&profilerEntry) override;

  void addLocalTask(TaskFunction work) override;

  void await(ArrayRef<AnyAsyncValueRef> values, bool runNewTasks) override;

  size_t getParallelismLevel() const final {
    // `numWorkers` is set to the number of worker threads that are created by
    // the work queue +1 for a foreign thread.
    // TODO(#1903): This is a poor heuristic for subdividing work.
    return numWorkers;
  }

private:
  /// Returns the WorkQueueThread corresponding to the caller. If the caller
  /// is a foreign thread the same WorkQueueThread will be returned.
  WorkQueueThread *getCurrentWorkQueueThread() {
    size_t workerID = workerIDInTLS;
    assert(workerID < numWorkers);
    return workers + workerID;
  }

  /// This is the set of worker threads in the WorkQueue.  Note that we reserve
  /// entry #0 for foreign threads that may get donated to this queue.  That
  /// means that we never start worker #0.
  const size_t numWorkers;
  WorkQueueThread *workers;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  SharedThreadState sharedState;

  ///  This is the ringbuffer of work to do.
  LockFreeRingBuffer<ProfiledTaskFunction> taskList;
};
} // namespace

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numWorkers,
                                         std::chrono::nanoseconds busyWaitNs,
                                         size_t taskListCapacity)
    : numWorkers(numWorkers), sharedState{busyWaitNs,
                                          /*doneFlag=*/false,
                                          /*suspendedThreads=*/0},
      taskList(taskListCapacity) {
  assert(numWorkers <= kMaxWorkers && "Too many workers for bitvec width");
  // Initialize each thread with its required state. Note that workerID #0
  // does not start itself since it represents all 'foreign' threads.
  // workers.reserve(numWorkers);
  workers = static_cast<WorkQueueThread *>(
      malloc(sizeof(WorkQueueThread) * numWorkers));
  assert(workers);
  for (size_t i = 0; i < numWorkers; ++i)
    new (workers + i) WorkQueueThread(sharedState, taskList, i);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].~WorkQueueThread();
  free(workers);
}

void ThreadPoolWorkQueue::shutdown() {
  TimeTraceScope scope(InternalProfilerEntry("llcl.shutdown"));

  // Donate this thread to help drain the work queue if there's anything left.
  getCurrentWorkQueueThread()->runItems(
      /*earlyStopPredicate=*/[]() { return false; },
      /*lateStopPredicate=*/[]() { return false; },
      /*runNewTasks=*/true, /*waitForTasks=*/false,
      /*spinningLabel=*/"llcl.shutdown.spinning",
      /*sleepingLabel=*/"llcl.shutdown.sleeping");

  // Now we can tell all the threads to exit.
  sharedState.doneFlag.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake up if it is waiting.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].sema.post();

  // Mark no threads as suspended, even though they may not have woken up,
  // cleared their own bit and exited yet.  This ensures that any in-flight
  // andThenSync calls won't try to wake these threads as we start joining and
  // tearing them down.
  sharedState.suspendedThreads.store(0);

  // Join all the threads when they shut down cleanly.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].join();
}

void ThreadPoolWorkQueue::addTask(TaskFunction &&work,
                                  WorkProfilerEntry &&profilerEntry) {
  assert(work);
  WorkQueueThread *addingWorker = getCurrentWorkQueueThread();

  // Try to add this work to the RingBuffer.
  WorkProfilerEntry waitingEntry =
      profilerEntry.withNameSuffix(".waiting")
          .withDetailSuffix(
              printWorkerId(addingWorker->workerID)); // restarts clock
  ProfiledTaskFunction profiledTask(std::move(work), std::move(waitingEntry),
                                    std::move(profilerEntry));
  if (taskList.enqueue(profiledTask)) {
    // If there are any suspended workers, kick one of them now that there is
    // new work to do.
    int workerIDToPoke = sharedState.takeAnySuspendedThread();
    if (workerIDToPoke != -1) {
      assert(static_cast<size_t>(workerIDToPoke) < numWorkers);
      workers[workerIDToPoke].sema.post();
    }
    return;
  }

  // If we failed to add it, then the ring buffer is full: just run the work
  // item locally on the current stack.
  // NOTE: This runs the risk of stack overflow, but we don't have a choice.
  LLVM_DEBUG(
      llvm::dbgs()
      << "ThreadPoolWorkQueue: running immediately (task queue is full)\n");
  doWork(std::move(profiledTask), addingWorker->workerID);
}

void ThreadPoolWorkQueue::addLocalTask(M::LLCL::TaskFunction work) {
  assert(work);
  WorkQueueThread *callerWorker = getCurrentWorkQueueThread();
  if (callerWorker->workerID == 0 &&
      callerWorker->threadID != llvm::get_threadid()) {
    // Called from a foreign worker which is not within a runItems loop, so
    // there's no local task list we can enqueue to on this thread.
    if (kRunImmediatelyOnForeignThreads) {
      // Run right now.
      LLVM_DEBUG(llvm::dbgs() << "ThreadPoolWorkQueue: running immediately "
                                 "(called from non awaiting foreign thread)\n");
      ProfiledTaskFunction profiledTask(
          std::move(work), /*waiting=*/WorkProfilerEntry(),
          /*running=*/WorkProfilerEntry("llcl.waiter"));
      doWork(std::move(profiledTask), callerWorker->workerID);
    } else {
      // Add as a task.
      addTask(std::move(work), WorkProfilerEntry("llcl.waiter"));
    }
    return;
  }

  // Called from either a worker thread or the distinguished awaiting
  // foreign thread. Safe to enqueue directly.
  callerWorker->addLocalTask(std::move(work));
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values,
                                bool runNewTasks) {
  // If all the values are ready, then we don't have to do anything.
  if (llvm::all_of(values, [](auto &av) { return av.isReady(); }))
    return;

  // Figure out which WorkerThread this is being invoked from.  This could be
  // one of our workers or a foreign thread.
  WorkQueueThread *awaitingWorker = getCurrentWorkQueueThread();

  // We are done when numRemaining drops to zero.
  std::atomic<ssize_t> numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.  When done,
  // we signal the semaphore for this worker to make sure to wake it up if it
  // fell asleep.
  for (auto &value : values)
    value.andThenSync([&numRemaining, awaitingWorker, this]() {
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
      auto awaitingWorkerID = awaitingWorker->workerID;

      // If the worker doing the await() has suspended, make sure to wake it up
      // so it notices that it is done.
      if (sharedState.takeSuspendedThread(awaitingWorkerID)) {
        // NOTE: We may post without a corresponding wait in
        // WorkQueueThread::runItems if the earlyStopPredicate &
        // takeSuspendedThread path executes just after our takeSuspendedThread
        // above. In that case a future wait will just go around the work loop
        // again.
        //
        // NOTE: This wakes up exactly one sleeping thread. Since we only allow
        // one foreign thread to be running a runItems loop at a time the
        // semaphore should have at most one waiter.
        awaitingWorker->sema.post();
      }
    });

  // Run work items until the system is asked to shut down.
  awaitingWorker->runItems(
      /*earlyStopPredicate=*/
      [&numRemaining]() -> bool {
        // Exit early as soon as numRemaining drops to zero.
        // TODO: Relaxed memory consistency!
        return numRemaining.load(std::memory_order_seq_cst) == 0;
      },
      /*lateStopPredicate=*/
      []() -> bool {
        // No additional shutdown check after waking, the early
        // check will suffice.
        return false;
      },
      /*runNewTasks=*/runNewTasks,
      /*waitForTasks=*/true,
      /*spinningLabel=*/
      runNewTasks ? "llcl.await.spinning" : "llcl.awaitQuietly.spinning",
      /*sleepingLabel=*/
      runNewTasks ? "llcl.await.sleeping" : "llcl.awaitQuietly.sleeping");

  assert(numRemaining.load() == 0);
}

//===----------------------------------------------------------------------===//
// createThreadPoolWorkQueue entrypoint
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue>
M::LLCL::createThreadPoolWorkQueue(size_t numThreads,
                                   std::chrono::nanoseconds busyWait,
                                   size_t taskListCapacity) {
  // We expect `numThreads` to be the total numbers of threads that are
  // accessing the work queue. As there will be an external thread that will
  // access the work queue and take items from it by calling `await`, we create
  // `numThreads - 1` worker threads from the thread pool work queue.
  assert(numThreads > 0);

  assert(taskListCapacity > 0);

  // We use a 64-bit value "thread suspended" value currently so we cap at 64
  // threads.  This algorithm isn't going to scale beyond 64 threads anyway.
  numThreads = std::min(numThreads, kMaxWorkers);
  return std::make_unique<ThreadPoolWorkQueue>(numThreads, busyWait,
                                               taskListCapacity);
}
