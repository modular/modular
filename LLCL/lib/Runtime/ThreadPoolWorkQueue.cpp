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
#include "LLCL/Support/ThreadAffinity.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"

#define DEBUG_TYPE "llcl"

using namespace M;
using namespace M::LLCL;

//
// Terminology:
//  - Worker thread: a thread we create which is running a dedicated runItems
//    loop.
//  - Main thread: if in mainWillDonate mode, this is the thread which created
//    the work queue. That thread may call await to donate itself to processing
//    work items alongside the worker threads while waiting for values. That
//    thread must also be the one to call shutdown.
//  - Foreign thread: any thread other than a worker or main thread. Foreign
//    threads may call addTasks and await. If not in mainWillDonate mode,
//    a foreign thread may also call shutdown. A foreign thread will never
//    donate itself to processing work items.
//

//===----------------------------------------------------------------------===//
// Compile-time config
//===----------------------------------------------------------------------===//

/// Minimum task list capacity.
constexpr size_t kMinTaskListCapacity = 128;

/// Number of task list slots per thread.
constexpr size_t kTaskListSlotsPerThread = 16;

/// Amount of time to spend spinning while waiting for work before going to
/// sleep on a semaphore.
constexpr std::chrono::nanoseconds kBusyWait = std::chrono::milliseconds(1);

//===----------------------------------------------------------------------===//
// WorkerThread
//===----------------------------------------------------------------------===//

namespace {

/// Tracks the overall shutdown progress for the work queue.
enum WorkQueueState : uint8_t { kReady = 0, kShuttingDown = 1, kShutdown = 2 };

#if MODULAR_PARANOID
/// Sleep for a random period to try to tickle data races.
static void randomSleep() {
  std::chrono::milliseconds delay{(rand() % 4) * 2000};
  if (delay.count() > 0) {
    TimeTraceScope scope(AllWorkItemsProfilerEntry::create("llcl.randomSleep"));
    std::this_thread::sleep_for(delay);
  }
}
#endif

/// Bit index i is true if the thread with workedID i is suspended.
using SuspendedThreadsBitvec = uint64_t;
constexpr size_t kMaxWorkers = sizeof(SuspendedThreadsBitvec) * 8;
constexpr SuspendedThreadsBitvec getSuspendedThreadIdMask(size_t workerID) {
  return UINT64_C(1) << workerID;
}

static constexpr auto printWorkerId(size_t workerID) {
  return [workerID]() {
    return Twine("(workerID:").concat(Twine(workerID)).concat(Twine(")")).str();
  };
}

/// Provides the state needed to synchronize the workers in the thread pool.
struct SharedThreadState {
  static_assert(std::atomic<SuspendedThreadsBitvec>::is_always_lock_free,
                "suspendedThreads should always be lock free");

  SharedThreadState(bool mainWillDonate, bool paranoid)
      : mainWillDonate(mainWillDonate)
#if MODULAR_PARANOID
        ,
        paranoid(paranoid)
#endif
  {
  }

  /// If true, the 'main' thread which constructed the work queue is going to
  /// call await to donate itself as another worker alongside the
  /// numWorkers - 1 other worker threads. That thread must eventually call
  /// shutdown.
  ///
  /// Otherwise there is no 'main' thread, just 'worker' and 'foreign' threads.
  bool mainWillDonate;

#if MODULAR_PARANOID
  /// If true, try to tickle race conditions with sleeps.
  /// Very expensive, hence guard by a runtime flag in addition to the
  /// compile-time MODULAR_PARANOID flag.
  bool paranoid;

  /// Track when the overall work queue is entering or exited the shutdown
  /// quiescence period.
  std::atomic<WorkQueueState> state = kReady;
#endif

  /// This flag indicates when a worker thread should quit working and get
  /// ready to be joined.
  std::atomic<bool> doneFlag = false;

  /// This keeps a bitset of suspended threads, indexed by workerID.  This will
  /// thrash around a lot when the workqueue is close to empty and threads are
  /// starting and stopping themselves, but should stay zero and read-only when
  /// there is a lot of work to do.
  ///
  /// This is aligned because the state above is immutable or (in the case of
  /// doneFlag) almost never changing. We don't want doneFlag to be on the same
  /// cache line as suspendedThreads.
  AlignedAtomic<SuspendedThreadsBitvec> suspendedThreads = 0;

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

  /// If there are any suspended workers, return a worker id for one of them.
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
        return llvm::countr_zero(loadedSuspendedThreads ^ newSuspendedThreads);
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
/// vector. Will be left zero for 'main' and 'foreign' threads.
static thread_local size_t workerIDInTLS = 0;

/// Wrapper around an std::thread created for each worker thread, or
/// a placeholder for the 'main' thread.
struct WorkQueueThread {
  /// Overall state shared by all threads.
  SharedThreadState &sharedState;

  /// 'Local' tasks which can be run on this thread as they become available.
  /// No threading synchronization is required here since tasks are added to
  /// and removed only by the unique thread (currently) tied to this object.
  /// However, we do need to protect against runItems being called recursively.
  ///
  /// Tasks on this list always take precedence over those in taskList and
  /// overflowTaskList.
  SmallVector<TaskFunction, 6> localTaskList;

  /// The lock-free queue of pending tasks available for any worker to
  /// process.
  ///
  /// Tasks on this list always take precedence over those in overflowTaskList.
  LockFreeRingBuffer<TaskFunction> &taskList;

  /// The mutex-protected queue of pending 'overflow' tasks available for any
  /// worker to process. Since synchronization is expensive, should only be
  /// checked before the worker thread would otherwise sleep.
  std::mutex &overflowMutex; // Protects overflowTaskList
  SmallVectorImpl<TaskFunction> &overflowTaskList;

  /// Unique index for this thread.
  size_t workerID;

  /// The CPU we'd prefer this worker to have affinity for, or ~0 if no
  /// affinity is intended for this worker.
  size_t cpuID;

  /// This is a per-worker semaphore that this blocks on when they run
  /// out of things to do.
  Semaphore sema;

  /// The system's identifier for the thread associated with this
  /// WorkQueueThread, either a 'worker' or the 'main' thread if in
  /// mainWillDonate mode.
  uint64_t threadID = 0;

  // The underlying worker thread, or none if this WorkQueueThread represents
  // the 'main' thread in mainWillDonate mode.
  std::optional<std::thread> thread;

  /// Create a WorkQueueThread representing the worker with workerID. If
  /// necessary, the underlying worker thread will be created and it will
  /// enter its runItems loop.
  WorkQueueThread(SharedThreadState &sharedState,
                  LockFreeRingBuffer<TaskFunction> &taskList,
                  std::mutex &overflowMutex,
                  SmallVectorImpl<TaskFunction> &overflowTaskList,
                  size_t workerID, size_t cpuID)
      : sharedState(sharedState), taskList(taskList),
        overflowMutex(overflowMutex), overflowTaskList(overflowTaskList),
        workerID(workerID), cpuID(cpuID) {
    if (sharedState.mainWillDonate && workerID == 0) {
      // We can leave workerIDInTLS as zero.
      // Remember the caller is to be our 'main' thread, and will call
      // await to process work items.
      threadID = llvm::get_threadid();
      assert(threadID && "get_threadid returned zero for the main thread");
    } else {
      // Start a 'worker' thread.
      thread.emplace(&WorkQueueThread::runOnThread, this);
    }
  }

  ~WorkQueueThread() { assert(localTaskList.empty()); }

  /// Schedule this work item on the localTaskList to be executed on the next
  /// runItems loop.
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

  // Execute a single work item, which may have come from either addTask
  // or addLocalTask (via an AsyncValue waiter).
  template <bool IsWaiter>
  void doWork(TaskFunction &&taskFunction) {
#if MODULAR_PARANOID
    if (sharedState.paranoid)
      randomSleep();
#endif

    {
      TimeTraceScope scope(AllWorkItemsProfilerEntry::create(
          IsWaiter ? "llcl.waiter" : "llcl.doWork"));
      // Do the work.
      taskFunction();
    }

#if MODULAR_PARANOID
    if constexpr (IsWaiter) {
      assert(sharedState.state != kShutdown &&
             "ThreadPoolWorkQueue was shutdown while a waiter closure was "
             "still in-flight");
    } else {
      assert(sharedState.state != kShutdown &&
             "ThreadPoolWorkQueue was shutdown while a task closure was still "
             "in-flight");
    }
#endif
  }

  /// This implements the main worker loop, used by runOnThread, await and
  /// shutdown. The loop runs until earlyStopPredicate or lateStopPredicate
  /// return true. The "early" predicate is called for every work item that
  /// is executed, and the "late" one is called when waking up from a
  /// suspended state.
  ///
  /// The loop will busy wait or sleep waiting for new tasks only if
  /// waitForTasks is true, otherwise the loop will exit once the work queue
  /// and local task list is empty.
  ///
  /// The given labels are used only for profiling entries when spinning or
  /// sleeping. The current running profiling entry will be paused then
  /// resumed while other tasks are executed.
  ///
  /// If the caller is a 'foreign' thread then only safe to call if
  /// ensureOwningThread has returned true.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsOnOwningThread(EarlyStopPredicateFn earlyStopPredicate,
                              LateStopPredicateFn lateStopPredicate,
                              bool waitForTasks, StringRef spinningLabel,
                              StringRef sleepingLabel);

  /// As above, but without tracking the running profiling entry.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                    LateStopPredicateFn lateStopPredicate, bool waitForTasks,
                    StringRef spinningLabel, StringRef sleepingLabel);

private:
  /// The main function invoked by std::thread.
  void runOnThread();
};
} // namespace

void WorkQueueThread::runOnThread() {
  assert(!sharedState.mainWillDonate ||
         workerID != 0 &&
             "The WorkQueueThread for the main thread should not be run");

  // Set the current workerID in thread local storage so we can find it later
  // when re-entering.
  workerIDInTLS = workerID;

  // Capture the worker's thread id so we can distinguish worker threads
  // from different work queues.
  threadID = llvm::get_threadid();
  assert(threadID && "get_threadid returned zero for a worker thread");

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  llvm::set_thread_name("LLCL Thread " + llvm::Twine(workerID));

  // On systems that support it, give the thread affinity for one CPU.
  LLCL::setThreadAffinity(cpuID);

  // Run work items until the system is asked to shut down.
  runItemsImpl(
      /*earlyStopPredicate=*/[]() { return false; }, // Always loop.
      /*lateStopPredicate=*/
      [this]() {
        // On wakeup from suspend, check to see if we're supposed to
        // shutdown and stop executing work.
        return sharedState.doneFlag.load(std::memory_order_acquire);
      },
      /*waitForTasks=*/true,
      /*spinningLabel=*/"llcl.runOnThread.spinning",
      /*sleepingLabel=*/"llcl.runOnThread.sleeping");
}

template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItemsOnOwningThread(
    EarlyStopPredicateFn earlyStopPredicate,
    LateStopPredicateFn lateStopPredicate, bool waitForTasks,
    StringRef spinningLabel, StringRef sleepingLabel) {
  if (sharedState.mainWillDonate && workerID == 0) {
    // Temporarily set the main thread's affinity while it is processing work.
    LLCL::runWithThreadAffinity(cpuID, [&]() {
      runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
          earlyStopPredicate, lateStopPredicate, waitForTasks, spinningLabel,
          sleepingLabel);
    });
  } else {
    runItemsImpl<EarlyStopPredicateFn, LateStopPredicateFn>(
        earlyStopPredicate, lateStopPredicate, waitForTasks, spinningLabel,
        sleepingLabel);
  }
}

template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
void WorkQueueThread::runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                                   LateStopPredicateFn lateStopPredicate,
                                   bool waitForTasks, StringRef spinningLabel,
                                   StringRef sleepingLabel) {
  while (true) {
  KeepRunning:
    // Prefer to run local work items as soon as they are available.
    // CAUTION: a work function may add to this list, and may even invoke
    // runItems recursively.
    while (!localTaskList.empty()) {
#if MODULAR_PARANOID
      // Try to tickle bugs by working through tasks in random order.
      size_t i = rand() % localTaskList.size();
      TaskFunction taskFunction = std::move(localTaskList[i]);
      localTaskList.erase(localTaskList.begin() + i);
#else
      TaskFunction taskFunction = std::move(localTaskList.back());
      localTaskList.pop_back();
#endif
      doWork</*IsWaiter=*/true>(std::move(taskFunction));
    }

    if (earlyStopPredicate())
      return;

    // In the normal case we happily pick up and do work.
    if (auto taskFunction = taskList.dequeue()) {
      doWork</*IsWaiter=*/false>(std::move(taskFunction));
      goto KeepRunning;
    }

    if (!waitForTasks)
      return;

    {
      auto spinning =
          InternalProfilerEntry::create(spinningLabel, printWorkerId(workerID));

      // If we've run out of work to do, we need to quiesce and ultimately block
      // in the kernel on the semaphore.  However, we don't want to immediately
      // give up hope, because we may be "right about to" get new work incoming.
      // We also want to make sure to use exponential backoff to avoid pummeling
      // the memory hierarchy of the threads that are doing useful work.  As
      // such, we use a BusyWaitSpinWaiter.
      BusyWaitSpinWaiter spinWaiter(kBusyWait);

      // Spin until we find some work to do.
      while (!spinWaiter.wait()) {
        // If we ever succeed in finding work to do, go back to running like
        // normal.
        if (auto work = taskList.dequeue()) {
          std::move(spinning).record();
          doWork</*IsWaiter=*/false>(std::move(work));
          goto KeepRunning;
        }

        // If we're spinning and the early or the late stop condition happens,
        // then we're done.  Checking the late stop condition here make sure
        // our threads shut down promptly when a runtime is torn down.
        if (earlyStopPredicate() || lateStopPredicate()) {
          std::move(spinning).record();
          return;
        }
      }
      std::move(spinning).record();
    }

    // The lock-free task queue appears to be empty. Since we're about to go
    // to sleep anyway, we can justify the expense of pumping any items out
    // of the overflow task queue into the lock-free queue.
    // Note we don't worry about preserving order for the overflow tasks since
    // there's no guarantee of fairness anyway.
    {
      std::lock_guard<std::mutex> guard(overflowMutex);
      if (!overflowTaskList.empty()) {
        while (!overflowTaskList.empty()) {
          TaskFunction taskFunction = std::move(overflowTaskList.back());
          overflowTaskList.pop_back();
          if (!taskList.enqueue(taskFunction)) {
            // Oops, went too far.
            overflowTaskList.emplace_back(std::move(taskFunction));
            break;
          }
        }
        goto KeepRunning;
      }
    }

    // We've waited long enough for new work to show up, so yield the thread to
    // the OS so we don't burn power and starve other tasks on the system.
    sharedState.markSuspended(workerID);

    // Double check the fast predicate after marking ourselves as suspended
    // (which only matters for await()).  Await won't signal the waiter unless
    // it sees it at the right time.
    if (earlyStopPredicate()) {
      sharedState.takeSuspendedThread(workerID);
      return;
    }

    {
      // Ok, finally block.
      TimeTraceScope scope(
          WorkProfilerEntry::create(sleepingLabel, printWorkerId(workerID)));
      sema.wait();
    }

    // On wakeup, check the 'slow' predicate to see if we should stop (this is
    // how worker threads know to exit).  The early predicate is checked as
    // part of the outer while loop immediately after this.
    if (lateStopPredicate())
      return;
  }
}

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue
//===----------------------------------------------------------------------===//

namespace {
/// This class provides a thread-pool that implements the WorkQueue
/// interface. It starts a dynamic number of threads and distributes work to
/// it by means of a concurrent-safe queue.
class ThreadPoolWorkQueue : public WorkQueue {
public:
  /// Initialize the thread pool and start up the worker threads, with one
  /// thread per entry in cpuIDs. By the time the constructor finishes, all
  /// the worker threads have started and shall only be cancelled by the
  /// destructor.
  ThreadPoolWorkQueue(const std::vector<size_t> &cpuIDs,
                      size_t taskListCapacity, bool mainWillDonate,
                      bool paranoid);

  ~ThreadPoolWorkQueue() override;

  void shutdown() override;

  void addTask(TaskFunction &&work) override;

  void addLocalTask(TaskFunction &&work) override;

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  bool callerIsForeign() const override;

  size_t getParallelismLevel() const final {
    // `numWorkers` is set to the number of worker threads that are created
    // by the work queue, plus one for the 'main' thread if in mainWillDonate
    // mode.
    // TODO(#1903): This is a poor heuristic for subdividing work.
    return numWorkers;
  }

private:
  /// If the caller is a worker thread or the 'main' thread for this work queue
  /// then return the WorkQueueThread which represents it. Otherwise, if the
  /// caller is a 'foreign' thread (including workers from other work queues)
  /// then return null.
  WorkQueueThread *getOwningWorkQueueThread() const {
    size_t workerID = workerIDInTLS;

    if (workerID >= numWorkers)
      // Presumably a 'worker' thread from some other work queue.
      return nullptr;

    WorkQueueThread *worker = workers + workerID;

    if (worker->threadID != llvm::get_threadid())
      // A 'foreign' thread.
      return nullptr;

    // Either the 'main' or a 'worker' thread associated with this work queue.
    return worker;
  }

  /// Returns the WorkQueueThread for workerID.
  WorkQueueThread *getWorkQueueThread(size_t workerID) const {
    assert(workerID < numWorkers);
    return workers + workerID;
  }

  /// This is the set of WorkQueueThread objects in the WorkQueue. If in
  /// mainWillDonate mode then the first entry will represent the 'main'
  /// thread.
  const size_t numWorkers;
  WorkQueueThread *workers = nullptr;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  SharedThreadState sharedState;

  /// The lock-free queue of pending tasks available for any worker.
  /// It may become full.
  LockFreeRingBuffer<TaskFunction> taskList;
  /// The mutex-protected queue of pending tasks available for any worker.
  /// Only used when the taskList is full.
  std::mutex overflowMutex; // protects overflowTaskList
  SmallVector<TaskFunction> overflowTaskList;
};
} // namespace

ThreadPoolWorkQueue::ThreadPoolWorkQueue(const std::vector<size_t> &cpuIDs,
                                         size_t taskListCapacity,
                                         bool mainWillDonate, bool paranoid)
    : numWorkers(cpuIDs.size()), sharedState(mainWillDonate, paranoid),
      taskList(taskListCapacity) {
  assert(numWorkers <= kMaxWorkers && "Too many workers for bitvec width");
  // Initialize each thread with its required state.
  // Note that we're constructing the array manually since WorkQueueThreads have
  // non-moveable atomics.
  workers = static_cast<WorkQueueThread *>(
      malloc(sizeof(WorkQueueThread) * numWorkers));
  assert(workers);
  for (size_t workerID = 0; workerID < numWorkers; ++workerID)
    new (workers + workerID)
        WorkQueueThread(sharedState, taskList, overflowMutex, overflowTaskList,
                        workerID, cpuIDs[workerID]);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
  // Note we can't assert state == kShutdown since queue may be created
  // and destroyed without ever being included in a runtime.
  assert(!taskList.dequeue());

  // Destroy all the threads datastructures.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].~WorkQueueThread();
  free(workers);
}

void ThreadPoolWorkQueue::shutdown() {
#if MODULAR_PARANOID
  WorkQueueState expected = kReady;
  assert(sharedState.state.compare_exchange_strong(expected, kShuttingDown));
#endif

  TimeTraceScope scope(InternalProfilerEntry::create("llcl.shutdown"));

  WorkQueueThread *callingWorker = getOwningWorkQueueThread();

  if (sharedState.mainWillDonate) {
    assert(callingWorker && callingWorker->workerID == 0 &&
           "must shutdown from the 'main' thread in mainWillDonate mode");
  } else {
    assert(
        !callingWorker &&
        "must shutdown from a 'foreign' thread if not in mainWillDonate mode");
  }

  if (callingWorker) {
    // Donate this thread to help drain the work queue if there's anything left.
    callingWorker->runItemsOnOwningThread(
        /*earlyStopPredicate=*/[]() { return false; }, // Always loop
        /*lateStopPredicate=*/[]() { return false; },  // Always loop
        /*waitForTasks=*/false,
        /*spinningLabel=*/"llcl.shutdown.spinning",
        /*sleepingLabel=*/"llcl.shutdown.sleeping");
  }
  // else: the existing workers will keep processing work items until they
  // test the lateStopPredicate. This is as good a synchronization we can
  // guarantee if not in mainWillDonate mode.

  // Tell all the threads to exit.
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

#if MODULAR_PARANOID
  expected = kShuttingDown;
  assert(sharedState.state.compare_exchange_strong(expected, kShutdown));
#endif
}

void ThreadPoolWorkQueue::addTask(TaskFunction &&work) {
  assert(work);
#if MODULAR_PARANOID
  // This is not a true interlock, but will at least catch obvious
  // use-after-shutdowns.
  assert(sharedState.state != kShutdown);
#endif

  // Try to add this work to the lock-free queue.
  if (taskList.enqueue(work)) {
    // If there are any suspended workers, kick one of them now to make sure
    // there's at least one worker still awake to pick up work.
    int workerIDToPoke = sharedState.takeAnySuspendedThread();
    if (workerIDToPoke != -1)
      getWorkQueueThread(static_cast<size_t>(workerIDToPoke))->sema.post();
    return;
  }

  // The lock-free queue is full. We now have four choices:
  //  - Run the task now on the callers stack. However, that risks overflow,
  //    and obviously would require us to give up on the 'tasks are never run
  //    immediately' API contract.
  //  - Push the task onto a local task list. That give up worker balancing,
  //    and won't work if the caller is a non-awaiting foreign thread (since
  //    the local task list is deliberately synchronization free).
  //  - Make the lock-free queue dynamically resizable. However, it's not clear
  //    how to do that without giving up its nice lock-free push and pop
  //    independence.
  //  - Push the task onto an overflow list, which we can mutex protect just
  //    like your grandfather would have written. Workers can check the
  //    overflow list when they would otherwise about to go to sleep. In this
  //    way the mutex overhead is only paid for in the uncommon case. However,
  //    we obviously risk starving these tasks.
  // We go for the last option.
  std::lock_guard<std::mutex> guard(overflowMutex);
  overflowTaskList.emplace_back(std::move(work));
}

void ThreadPoolWorkQueue::addLocalTask(TaskFunction &&work) {
  assert(work);
#if MODULAR_PARANOID
  // This is not a true interlock, but will at least catch obvious
  // use-after-shutdowns.
  assert(sharedState.state != kShutdown);
#endif
  WorkQueueThread *callerWorker = getOwningWorkQueueThread();
  if (callerWorker == nullptr) {
    // Called from a foreign thread, so there's no local task list we can
    // enqueue to on this thread. Add as a task instead.
    addTask(std::move(work));
    return;
  }

  // Called from either a worker thread or the 'main' therad. Safe to enqueue
  // directly.
  callerWorker->addLocalTask(std::move(work));
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
#if MODULAR_PARANOID
  // This is not a true interlock, but will at least catch obvious
  // use-after-shutdowns.
  assert(sharedState.state == kReady);
#endif

  // If all the values are ready, then we don't have to do anything.
  if (llvm::all_of(values, [](auto &av) { return av.isReady(); }))
    return;

  // Figure out which WorkerThread this is being invoked from. This could be
  // one of our workers, the 'main' thread, or a 'foreign' thread.
  WorkQueueThread *awaitingWorker = getOwningWorkQueueThread();

  // We are done when numRemaining drops to zero.
  std::atomic<ssize_t> numRemaining = values.size();

  if (awaitingWorker) {
    // The caller is a worker or main thread, so is willing to donate itself
    // to processing work items while awaiting.

    // As each value becomes available, we can decrement our counts.  When done,
    // we signal the semaphore for this worker to make sure to wake it up if it
    // fell asleep.
    for (auto &value : values) {
      value.andThenSync([&numRemaining, awaitingWorker, this]() {
        // Decrement the count of async values that we're waiting on.
        // TODO: This can probably use more relaxed memory consistency!
        if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
          return;

        // Get the thread ID of the thread running the andThenSync, for tracing.
        size_t workerID = workerIDInTLS;
        (void)workerID;

        // When it drops to zero, we're good to go and whatever thread is
        // waiting for this will exit out of its 'runItems' loop.  That said,
        // the thread may be suspended on a semaphore.  Check for this, and if
        // so, signal its semaphore so it wakes up and notes that it is done.
        auto awaitingWorkerID = awaitingWorker->workerID;

        // If the worker doing the await() has suspended, make sure to wake it
        // up so it notices that it is done.
        if (sharedState.takeSuspendedThread(awaitingWorkerID)) {
          // NOTE: We may post without a corresponding wait in
          // WorkQueueThread::runItems if the earlyStopPredicate &
          // takeSuspendedThread path executes just after our
          // takeSuspendedThread above. In that case a future wait will just go
          // around the work loop again.
          //
          // NOTE: This wakes up exactly one sleeping thread. Since we only
          // allow one foreign thread to be running a runItems loop at a time
          // the semaphore should have at most one waiter.
          awaitingWorker->sema.post();
        }
      });
    }

    // Run work items until all values are available.
    awaitingWorker->runItemsOnOwningThread(
        /*earlyStopPredicate=*/
        [&numRemaining]() {
          // Exit early as soon as numRemaining drops to zero.
          // TODO: Relaxed memory consistency!
          return numRemaining.load(std::memory_order_seq_cst) == 0;
        },
        /*lateStopPredicate=*/
        []() {
          // No additional shutdown check after waking, the early
          // check will suffice.
          return false;
        },
        /*waitForTasks=*/true,
        /*spinningLabel=*/"llcl.await.spinning",
        /*sleepingLabel=*/"llcl.await.sleeping");

  } else {
    // The caller is a 'foreign' thread. Sleep until all values are available,
    // letting the other workers do work on the caller's behalf.
    //
    // Ideally we'd sleep only until all our values are ready or the other
    // foreign thread is done with its runItems loop, whichever is sooner.
    Semaphore sema;
    // As each value becomes available, we can decrement our counts.  When done,
    // we signal the semaphore to wake up the awaiting foreign thread.
    for (auto &value : values) {
      value.andThenSync([&numRemaining, &sema]() {
        // Decrement the count of async values that we're waiting on.
        // TODO: This can probably use more relaxed memory consistency!
        if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
          return;
        sema.post();
      });
    }
    sema.wait();
  }

  assert(numRemaining.load() == 0);
#if MODULAR_PARANOID
  // Try to catch if the runtime was torn down while we were awaiting.
  assert(sharedState.state == kReady);
#endif
}

bool ThreadPoolWorkQueue::callerIsForeign() const {
  return getOwningWorkQueueThread() == nullptr;
}

//===----------------------------------------------------------------------===//
// createThreadPoolWorkQueue entrypoint
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue>
M::LLCL::createThreadPoolWorkQueue(size_t numThreads, bool mainWillDonate,
                                   bool paranoid) {
#if MODULAR_PARANOID
#ifdef NDEBUG
  llvm::dbgs() << "CAUTION: Asked for a MODULAR_PARANOID build with NDEBUG. "
                  "Asserts will not be active, which is unlikely to be what "
                  "you intended.\n";
#else  // NDEBUG
  if (paranoid)
    llvm::dbgs() << "CAUTION: Running a MODULAR_PARANOID build with additional "
                    "checks enabled by the paranoid flag.\n";
  else
    llvm::dbgs() << "CAUTION: Running a MODULAR_PARANOID build. Consider using "
                    "the paranoid flag for even more paranoia.\n";
#endif // NDEBUG
#else  // MODULAR_PARANOID
  if (paranoid)
    llvm::dbgs() << "CAUTION: The paranoid flag is ignored in non "
                    "MODULAR_PARANOID builds\n";
#endif // MODULAR_PARANOID

  // Using numThreads as a hint, figure out a CPU for each worker thread
  // and the main thread. The CPU ids may end up as kNoAffinity, but the
  // vector size will still guide the construction of worker threads.
  auto cpuIDOr = getThreadAffinityCpuIds(numThreads, kMaxWorkers);

  // TODO: This function should return the error back to caller.
  if (cpuIDOr.isError())
    llvm::report_fatal_error(cpuIDOr.getError());
  std::vector<size_t> cpuIDs = *cpuIDOr;
  assert(!cpuIDs.empty());
  size_t numCores = std::thread::hardware_concurrency();
  if (cpuIDs.size() != numCores)
    LLVM_DEBUG(
        llvm::dbgs()
        << "createThreadPoolWorkQueue: Number of threads (" << cpuIDs.size()
        << ") differs from number of cores (" << numCores
        << "), possibly since ignoring hyperthreading and other sockets.\n");

  size_t taskListCapacity =
      std::max(kMinTaskListCapacity, numThreads * kTaskListSlotsPerThread);
  LLVM_DEBUG(llvm::dbgs()
             << "createThreadPoolWorkQueue: Task list has capacity of at least "
             << taskListCapacity << " slots.\n");

  return std::make_unique<ThreadPoolWorkQueue>(cpuIDs, taskListCapacity,
                                               mainWillDonate, paranoid);
}
