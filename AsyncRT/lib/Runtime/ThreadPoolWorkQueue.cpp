//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a multi-threaded work queue implementation.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/Chain.h"
#include "AsyncRT/Support/LockFreeRingBuffer.h"
#include "AsyncRT/Support/Semaphore.h"
#include "AsyncRT/Support/ThreadAffinity.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/Threading/Atomics.h"
#include "Support/Threading/SpinWaiter.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include <cmath>

#define DEBUG_TYPE "llcl"

using namespace M;
using namespace M::AsyncRT;

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

/// Number of task list slots per thread.
constexpr size_t kTaskListSlotsPerThread = 1024;

/// Max number of worker threads.
constexpr size_t kMaxWorkers = 1024;

//===----------------------------------------------------------------------===//
// WorkerThread
//===----------------------------------------------------------------------===//

namespace {
#if LLCL_WORKER_STATS
#define LLCL_PRINT_WORKER_STATS(X) X;
#else
#define LLCL_PRINT_WORKER_STATS(X)
#endif
/// Tracks the overall shutdown progress for the work queue.
enum WorkQueueState : uint8_t { kReady = 0, kShuttingDown = 1, kShutdown = 2 };
enum WorkType : uint8_t { kLocal = 0, kAffinity = 1, kGlobal = 2 };
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

/// Provides the state needed to synchronize the workers in the thread pool.
/// We use a uint64_t bit-vec (SuspendedThreadsBitvec) to represent the
/// suspended bit of each thread. This comes with a limitation that the system
/// could have just 64 threads. However most modern server cpu's have more than
/// 64 cores in a node. We implement scaling support through a simple
/// multi-cast scheme where each bit of the bit-vec represents more
/// than 1 thread, ie a `workerGroup` instead of a `worker`.
/// For example for a 128 cpu machine, bit 0 represents {worker0, worker1}.
/// If bit0 is set, it means either worker0 | worker1 is suspended.
/// This results in ambiguity when we query the bit-vec for addTask()/await()
/// to wakeup threads because the exact sleeping workerID is unknown to post
/// the appropriate semaphore. We handle this in the following way,
/// 1) when workerId is unknown, we will wakeup all the threads
/// represented by the bit-vec bit. For example, 0 -> {worker0->post(),
/// worker1->post()} This can be expensive, but hopefully, we do not have to
/// sleep/wake up threads often during model execution.
/// 2) when workerID is known, we always post the semaphore since bit-vec
/// information may have interference from other threads. This can lead to
/// spurious posts() but nonetheless ensures, the threads wake up to execute
/// the task.

/// Bit index i is true if any thread in the workedGroupID i is suspended.
using SuspendedThreadsBitvec = uint64_t;
constexpr size_t bitVectorWidth = sizeof(SuspendedThreadsBitvec) * 8;
constexpr SuspendedThreadsBitvec
getSuspendedThreadIdMask(size_t workerGroupID) {
  return UINT64_C(1) << workerGroupID;
}

struct SharedThreadState {
  static_assert(std::atomic<SuspendedThreadsBitvec>::is_always_lock_free,
                "suspendedThreads should always be lock free");

  SharedThreadState(CompactRuntimePtr runtimePtr, bool mainWillDonate,
                    bool paranoid, size_t numWorkers)
      : runtimePtr(runtimePtr), mainWillDonate(mainWillDonate)
#if MODULAR_PARANOID
        ,
        paranoid(paranoid)
#endif
  {
    // Keeping numWorkers in a workerGroup a power of 2 to simplify arithmetic.
    multicastFactor =
        numWorkers > bitVectorWidth
            ? static_cast<size_t>(std::ceil(
                  std::log2(numWorkers / static_cast<float>(bitVectorWidth))))
            : 0;
  }

  /// The runtime on behalf of which this thread is processing work items.
  CompactRuntimePtr runtimePtr;

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
  /// computed so that number of workers per groups is 2^multicastFactor.
  size_t multicastFactor;
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
  /// can know to wake it up when more work materializes. We set the bit of the
  /// corresponding workerGroupID
  void markSuspended(size_t workerID) {
    // number of workers per groups is 2^multicastFactor.
    auto workerGroupID = workerID >> multicastFactor;
    suspendedThreads.fetch_or(getSuspendedThreadIdMask(workerGroupID),
                              std::memory_order_seq_cst);
  }

  /// If the specified workerID is suspended, take its bit out of the
  /// suspendedThreads bitset and return true.  Otherwise return false.
  /// NOTE: takeSuspended may unset even if some other threads in the
  /// same workerGroup are suspended. This is fine since, we will always call
  /// the workerID->sema.post().
  bool takeSuspendedThread(size_t workerID) {
    // number of workers per groups is 2^multicastFactor.
    auto workerGroupID = workerID >> multicastFactor;
    SuspendedThreadsBitvec workerBit = getSuspendedThreadIdMask(workerGroupID);
    auto oldValue =
        suspendedThreads.fetch_and(~workerBit, std::memory_order_seq_cst);
    return oldValue & workerBit;
  }

  /// If there are any workerGroup's with suspended threads, return the id for
  /// one of them. Otherwise return -1. Since we do not know the workerID which
  /// is suspeneded, we assume all worker's are suspended and hence will post
  /// all the semaphores.
  int takeAnySuspendedThread() {
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

  /// 'Local' work items which can be run on this thread as they become
  /// available. No threading synchronization is required here since work items
  /// are added to and removed only by the unique thread (currently) tied to
  /// this object. However, we do need to protect against runItems being called
  /// recursively.
  ///
  /// Work items on this list always take precedence over those in taskList and
  /// overflowTaskList.
  size_t nextLocalTaskListIndex = 0;
  SmallVector<WorkItem, 6> localTaskList;
  /// Thread Local Queue of tasks processed according to taskId ordering.
  /// This is like localTaskList but can have multiple producers.
  LockFreeRingBuffer<WorkItem> affinityTaskList;
  /// The lock-free queue of pending tasks available for any worker to
  /// process.
  ///
  /// Work items on this list always take precedence over those in
  /// overflowTaskList.
  LockFreeRingBuffer<WorkItem> &taskList;

  /// The mutex-protected queue of pending 'overflow' work items available for
  /// any worker to process. Since synchronization is expensive, should only be
  /// checked before the worker thread would otherwise sleep.
  std::mutex &overflowMutex; // Protects overflowTaskList
  SmallVectorImpl<WorkItem> &overflowTaskList;

  /// Spill queue for the affinityTaskList and its mutex. If the
  /// affinityTaskList is full, we spill over to this queue and later execute
  /// from the localTaskList maintaining the affinity. This is assumed to be
  /// a rare event and hence okay with slow handling like overflowTaskList.
  std::mutex localSpillQueueMutex; // Protects localSpillQueue
  SmallVector<WorkItem> localSpillQueue;
  /// Unique index for this thread.
  size_t workerID;

  /// The CPU we'd prefer this worker to have affinity for, or ~0 if no
  /// affinity is intended for this worker.
  size_t cpuID;

  /// Amount of time to spend spinning while waiting for work before going to
  /// sleep on a semaphore. Tuning this number is especially important for
  /// cases that interact with other threadpools. Ideally we should autotune
  /// this during the `warmup` phase or come up with heuristics based on
  /// the fallback ops distribution.
  std::chrono::microseconds busyWaitTime;

  /// This is a per-worker semaphore that this blocks on when they run
  /// out of things to do.
  Semaphore sema;

  /// The system's identifier for the thread associated with this
  /// WorkQueueThread, either a 'worker' or the 'main' thread if in
  /// mainWillDonate mode.
  uint64_t threadID = 0;

  /// The underlying worker thread, or none if this WorkQueueThread represents
  /// the 'main' thread in mainWillDonate mode.
  std::optional<std::thread> thread;

#if MODULAR_PARANOID
  /// Uses stack.
  SmallVector<ResourceUse> useStack;
#endif
  // The thread identifier prefix used to name the threads
  std::string_view poolName;
#if LLCL_WORKER_STATS
  uint64_t affinityAccessCount = 0;
  uint64_t globalAccessCount = 0;
  std::chrono::duration<double, std::micro> affinityListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> localListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> taskListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> affinityWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> localWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> taskListWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinAffinityListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinAffinityWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinTaskListAccessTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> spinTaskListWorkTime =
      std::chrono::microseconds(0);
  std::chrono::duration<double, std::micro> sleepTime =
      std::chrono::microseconds(0);
#endif
  /// Create a WorkQueueThread representing the worker with workerID. If
  /// necessary, the underlying worker thread will be created and it will
  /// enter its runItems loop.
  WorkQueueThread(SharedThreadState &sharedState,
                  LockFreeRingBuffer<WorkItem> &taskList,
                  std::mutex &overflowMutex,
                  SmallVectorImpl<WorkItem> &overflowTaskList, size_t workerID,
                  size_t cpuID, std::chrono::microseconds busyWaitTime,
                  std::string_view poolName)
      : sharedState(sharedState), affinityTaskList(kTaskListSlotsPerThread),
        taskList(taskList), overflowMutex(overflowMutex),
        overflowTaskList(overflowTaskList), workerID(workerID), cpuID(cpuID),
        busyWaitTime(busyWaitTime), poolName(poolName) {
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

  ~WorkQueueThread() {
    if (workerID == 0) {
      LLCL_PRINT_WORKER_STATS(
          llvm::dbgs() << "WorkerID,schedulerTasks(us),affinityQueueAccess(us),"
                          "affinityQueueWork(us),affinityAccessCount,"
                          "globalAccess(us),globalWork("
                          "us),globalAccessCount,sleep+wakeup(us)\n");
    }
    LLCL_PRINT_WORKER_STATS(
        llvm::dbgs()
        << "Thread" << workerID << "," << (localWorkTime).count() << ","
        << (affinityListAccessTime - affinityWorkTime).count() +
               (spinAffinityListAccessTime - spinAffinityWorkTime).count()
        << "," << (affinityWorkTime).count() + (spinAffinityWorkTime).count()
        << "," << affinityAccessCount << ","
        << (taskListAccessTime - taskListWorkTime).count() +
               (spinTaskListAccessTime - spinTaskListWorkTime).count()
        << "," << (taskListWorkTime).count() + (spinTaskListWorkTime).count()
        << "," << globalAccessCount << "," << sleepTime.count() << "\n");
    assert(localTaskList.empty() &&
           "destroying workqueuethread with pending local work items");
    std::lock_guard<std::mutex> guard(localSpillQueueMutex);
    assert(localSpillQueue.empty() &&
           "destroying Workqueuethread with pending fallback work items");
  }

  /// Schedule this work item on the localTaskList to be executed on the next
  /// runItems loop.
  void addLocalTask(WorkItem &&workItem) {
    localTaskList.emplace_back(std::move(workItem));
  }

  /// Schedules work on to the thread local queue of this worker. If the
  /// lockFreeRingBuffer is full, enqueue into the spill queue.
  void addAffinityTask(WorkItem &&workItem) {
    if (!affinityTaskList.enqueue(workItem)) {
      std::lock_guard<std::mutex> guard(localSpillQueueMutex);
      localSpillQueue.emplace_back(std::move(workItem));
    }
  }

  /// Joins the thread. Asserts that `sharedState.done` is true because
  /// otherwise the thread will never join.
  void join() {
    assert(sharedState.doneFlag.load() &&
           "must not destroy a WorkQueueThread object that is not pending "
           "completion.");
    if (thread.has_value())
      thread->join();
  }

  // Execute a single work item, which may have come from either addTask
  // or addLocalTask (via an AsyncValue waiter).
  template <bool IsWaiter>
  void doWork(WorkItem &&workItem, WorkType type) {
#if LLCL_WORKER_STATS
    auto start = std::chrono::high_resolution_clock::now();
#endif
#if MODULAR_PARANOID
    // Tickle race conditions.
    if (sharedState.paranoid)
      randomSleep();

    // Propagate use.
    useStack.emplace_back(std::move(workItem.use));
#endif
    // Do the work.
    {
      TimeTraceScope scope(AllWorkItemsProfilerEntry::create(
          IsWaiter ? "llcl.waiter" : "llcl.doWork"));
      workItem.task();
    }
#if LLCL_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();

    if (type == kLocal)
      localWorkTime += end - start;
    if (!IsWaiter) {
      if (type == kAffinity)
        affinityWorkTime += end - start;
      else if (type == kGlobal)
        taskListWorkTime += end - start;
    } else {
      if (type == kAffinity)
        spinAffinityWorkTime += end - start;
      else if (type == kGlobal)
        spinTaskListWorkTime += end - start;
    }
#endif
#if MODULAR_PARANOID
    // Pop use stack. The top may already have been reset.
    assert(!useStack.empty() &&
           "unbalanced pushes/pops to active lifetime stack");
    useStack.pop_back();
    assert(sharedState.state != kShutdown &&
           "ThreadPoolWorkQueue was shutdown while work item was in-flight.");
#endif
  }

  /// This implements the main worker loop, used by runOnThread, await and
  /// shutdown. The loop runs until earlyStopPredicate or lateStopPredicate
  /// return true. The "early" predicate is called for every work item that
  /// is executed, and the "late" one is called when waking up from a
  /// suspended state.
  ///
  /// The loop will busy wait or sleep waiting for new work items only if
  /// waitForTasks is true, otherwise the loop will exit once the work queue
  /// and local task list is empty.
  ///
  /// The given labels are used only for profiling entries when spinning or
  /// sleeping.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsOnOwningThread(EarlyStopPredicateFn earlyStopPredicate,
                              LateStopPredicateFn lateStopPredicate,
                              bool waitForTasks, StringLiteral spinningLabel,
                              StringLiteral sleepingLabel);

  /// As above, but without setting thread affinity for calls from the 'main'
  /// thread.
  template <typename EarlyStopPredicateFn, typename LateStopPredicateFn>
  void runItemsImpl(EarlyStopPredicateFn earlyStopPredicate,
                    LateStopPredicateFn lateStopPredicate, bool waitForTasks,
                    StringLiteral spinningLabel, StringLiteral sleepingLabel);

private:
  /// The main function invoked by std::thread.
  void runOnThread();
};
} // namespace

void WorkQueueThread::runOnThread() {
  assert((!sharedState.mainWillDonate || workerID != 0) &&
         "the WorkQueueThread for the main thread should not be run");

  // Set the current workerID in thread local storage so we can find it later
  // when re-entering.
  workerIDInTLS = workerID;

  // Set the current runtime in thread local storage.
  CompactRuntimePtr::setCurrentRuntime(sharedState.runtimePtr);

  // Capture the worker's thread id so we can distinguish worker threads
  // from different work queues.
  threadID = llvm::get_threadid();
  assert(threadID && "get_threadid returned zero for a worker thread");

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  llvm::set_thread_name(poolName + llvm::Twine(workerID));

  // On systems that support it, give the thread affinity for one CPU.
  AsyncRT::setThreadAffinity(cpuID);

  // Run work items until the system is asked to shut down.
  runItemsOnOwningThread(
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
    StringLiteral spinningLabel, StringLiteral sleepingLabel) {
  if (sharedState.mainWillDonate && workerID == 0) {
    // Temporarily set the main thread's affinity while it is processing work.
    AsyncRT::runWithThreadAffinity(cpuID, [&]() {
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
                                   bool waitForTasks,
                                   StringLiteral spinningLabel,
                                   StringLiteral sleepingLabel) {
  while (true) {
  KeepRunning:
    // Stop immediately if there is nothing to do.
    if (earlyStopPredicate())
      return;

    // Prefer to run local work items as soon as they are available.
    // CAUTION: A work function may append to this list, and may even
    //          invoke runItems recursively.
    auto start = std::chrono::high_resolution_clock::now();
    while (nextLocalTaskListIndex < localTaskList.size()) {
#if MODULAR_PARANOID
      // Try to tickle bugs by working through work items in random order.
      size_t i = rand() % localTaskList.size();
      WorkItem workItem = std::move(localTaskList[i]);
      localTaskList.erase(localTaskList.begin() + i);
#else
      WorkItem workItem = std::move(localTaskList[nextLocalTaskListIndex++]);
#endif

      // May append to localTaskList.
      // May re-enter this loop.
      doWork</*IsWaiter=*/true>(std::move(workItem), kLocal);
    }
    localTaskList.clear();
    nextLocalTaskListIndex = 0;
#if LLCL_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();
    localListAccessTime +=
        std::chrono::duration<double, std::micro>(end - start);
    start = std::chrono::high_resolution_clock::now();
#endif
    // Check for tasks in local taskId affinitized queue.
    if (auto workItem = affinityTaskList.dequeue()) {
      doWork</*IsWaiter=*/false>(std::move(workItem), kAffinity);
#if LLCL_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      affinityListAccessTime += (end - start);
      affinityAccessCount++;
#endif
      goto KeepRunning;
    }
#if LLCL_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    affinityListAccessTime += (end - start);
    start = std::chrono::high_resolution_clock::now();
#endif
    // In the normal case we happily pick up and do work.

    if (WorkItem workItem = taskList.dequeue()) {
      doWork</*IsWaiter=*/false>(std::move(workItem), kGlobal);
#if LLCL_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      taskListAccessTime += (end - start);
      globalAccessCount++;
#endif
      goto KeepRunning;
    }
#if LLCL_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    taskListAccessTime += (end - start);
#endif

    if (!waitForTasks)
      return;

    {
      auto spinning =
          InternalProfilerEntry::create(spinningLabel, (uint64_t)workerID);

      // If we've run out of work to do, we need to quiesce and ultimately block
      // in the kernel on the semaphore.  However, we don't want to immediately
      // give up hope, because we may be "right about to" get new work incoming.
      // We also want to make sure to use exponential backoff to avoid pummeling
      // the memory hierarchy of the threads that are doing useful work.  As
      // such, we use a BusyWaitSpinWaiter.
      BusyWaitSpinWaiter spinWaiter(busyWaitTime);

      start = std::chrono::high_resolution_clock::now();
      // Spin until we find some work to do.
      while (!spinWaiter.wait()) {
        // If we ever succeed in finding work to do, go back to running like
        // normal.

        if (auto workItem = affinityTaskList.dequeue()) {
          doWork</*IsWaiter=*/true>(std::move(workItem), kAffinity);
#if LLCL_WORKER_STATS
          auto end = std::chrono::high_resolution_clock::now();
          spinAffinityListAccessTime += (end - start);
          affinityAccessCount++;
#endif
          goto KeepRunning;
        }
#if LLCL_WORKER_STATS
        end = std::chrono::high_resolution_clock::now();
        spinAffinityListAccessTime += (end - start);
        start = std::chrono::high_resolution_clock::now();
#endif

        if (WorkItem workItem = taskList.dequeue()) {
          doWork</*IsWaiter=*/true>(std::move(workItem), kGlobal);
#if LLCL_WORKER_STATS
          auto end = std::chrono::high_resolution_clock::now();
          globalAccessCount++;
          spinTaskListAccessTime += (end - start);
#endif
          goto KeepRunning;
        }
#if LLCL_WORKER_STATS
        end = std::chrono::high_resolution_clock::now();
        spinTaskListAccessTime += (end - start);
#endif
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
    // of the overflow/localSpill task queues into the lock-free queues.
    // Note we don't worry about preserving order for the overflow tasks since
    // there's no guarantee of fairness anyway.
    {
      std::lock_guard<std::mutex> guard(localSpillQueueMutex);
      if (!localSpillQueue.empty()) {
        while (!localSpillQueue.empty()) {
          WorkItem workItem = localSpillQueue.pop_back_val();
          localTaskList.emplace_back(std::move(workItem));
        }
        goto KeepRunning;
      }
    }

    {
      std::lock_guard<std::mutex> guard(overflowMutex);
      if (!overflowTaskList.empty()) {
        while (!overflowTaskList.empty()) {
          WorkItem workItem = overflowTaskList.pop_back_val();
          if (!taskList.enqueue(workItem)) {
            // Oops, went too far.
            overflowTaskList.emplace_back(std::move(workItem));
            break;
          }
        }
        goto KeepRunning;
      }
    }

    // We've waited long enough for new work to show up, so check one last time
    // and yield the thread to the OS so we don't burn power and starve other
    // tasks on the system.

    sharedState.markSuspended(workerID);
    // Lets reason about ordering of markSuspended here and takeSuspended in
    // addTask.
    // T0(scheduler)                            T1(worker)
    // if(takeSuspended())                      markSuspended()
    // sema.post()                              sema.wait()
    //
    // Ordering 1: markSuspeneded() andThen takeSuspended().
    // if sema.post() andThen sema.wait() T1 does not go to sleep.
    // else T1 sleeps and wakes immediately.
    //
    // Ordering 2: takeSuspended() andThen markSuspended()
    // T0 is not going to post semaphore, but the task is already
    // enqueued. Run it now, unMark and go back to KeepRunning.

    start = std::chrono::high_resolution_clock::now();
    if (auto labelledTask = affinityTaskList.dequeue()) {
      doWork</*IsWaiter=*/false>(std::move(labelledTask), kAffinity);
#if LLCL_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      affinityAccessCount++;
      affinityListAccessTime += (end - start);
#endif
      goto KeepRunning;
    }
#if LLCL_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    affinityListAccessTime += (end - start);
    start = std::chrono::high_resolution_clock::now();
#endif
    // The same ordering explanation as above holds for the taskList too.
    // Let's say there are 2 threads in the pool with both threads busy waiting
    // on their way to sleep. The addTask() sees them as busy and does not post
    // any semaphores. However they both go to sleep not to be woken up by
    // anyone. We prefer checking for a dequeue here rather than always posting
    // a semaphore after enqueue in the addTask(). Also scenario is highly
    // unlikely for numThreads > 1.

    if (auto labelledTask = taskList.dequeue()) {
      doWork</*IsWaiter=*/false>(std::move(labelledTask), kGlobal);
#if LLCL_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      globalAccessCount++;
      taskListAccessTime += (end - start);
#endif
      goto KeepRunning;
    }
#if LLCL_WORKER_STATS
    end = std::chrono::high_resolution_clock::now();
    taskListAccessTime += (end - start);
#endif

    if (earlyStopPredicate()) {
      return;
    }

    {
      start = std::chrono::high_resolution_clock::now();
      // Ok, finally block.
      TimeTraceScope scope(
          InternalProfilerEntry::create(sleepingLabel, (uint64_t)workerID));
      sema.wait();
#if LLCL_WORKER_STATS
      auto end = std::chrono::high_resolution_clock::now();
      sleepTime += (end - start);
#endif
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
  ThreadPoolWorkQueue(CompactRuntimePtr runtimePtr, ArrayRef<size_t> cpuIDs,
                      size_t taskListCapacity, bool mainWillDonate,
                      std::chrono::microseconds threadBusyWaitTime,
                      bool paranoid, std::string_view poolName);

  ~ThreadPoolWorkQueue() override;

  void shutdown() override;

  void addTask(WorkItem &&workItem, int taskId = -1) override;

  void addLocalTask(WorkItem &&workItem) override;

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  bool callerIsForeign() const override;

  size_t getParallelismLevel() const final {
    // `numWorkers` is set to the number of worker threads that are created
    // by the work queue, plus one for the 'main' thread if in mainWillDonate
    // mode.
    // TODO(#1903): This is a poor heuristic for subdividing work.
    return numWorkers;
  }

#if MODULAR_PARANOID
  void pushDefaultUse(ResourceUse use) override {
    assert(use && "cannot push a null sue");
    WorkQueueThread *callerWorker = getOwningWorkQueueThread();
    assert(callerWorker && "cannot push a use from a foreign thread");
    callerWorker->useStack.emplace_back(std::move(use));
  }

  void popDefaultUse() override {
    WorkQueueThread *callerWorker = getOwningWorkQueueThread();
    assert(callerWorker && "cannot pop a use from a foreign thread");
    assert(!callerWorker->useStack.empty() &&
           "unbalanced pushes/pops on use stack");
    callerWorker->useStack.pop_back();
  }

  void taskIsDone() override {
    WorkQueueThread *callerWorker = getOwningWorkQueueThread();
    assert(callerWorker && "cannot mark task as done from a foreign thread");
    if (!callerWorker->useStack.empty())
      callerWorker->useStack.back().reset();
  }
#endif

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
    assert(workerID < numWorkers && "invalid worker id");
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

  /// The outer runtime, if any, for the thread using this work queue.
  CompactRuntimePtr outerRuntime;

  /// The lock-free queue of pending tasks available for any worker.
  /// It may become full.
  LockFreeRingBuffer<WorkItem> taskList;
  /// The mutex-protected queue of pending tasks available for any worker.
  /// Only used when the taskList is full.
  std::mutex overflowMutex; // protects overflowTaskList
  SmallVector<WorkItem> overflowTaskList;
  /// Log2(number of threads per bit of SuspendedThreadsBitvec)
  size_t multicastFactor = 0;
  std::string poolName;
#ifdef LLCL_WORKER_STATS
  AlignedAtomic<double> affinityEnqueueTime = 0.0f;
  AlignedAtomic<double> taskListEnqueueTime = 0.0f;
  AlignedAtomic<uint64_t> taskListEnqueCount = 0;
  AlignedAtomic<uint64_t> affinityEnqueCount = 0;
#endif
};
} // namespace

ThreadPoolWorkQueue::ThreadPoolWorkQueue(
    CompactRuntimePtr runtimePtr, ArrayRef<size_t> cpuIDs,
    size_t taskListCapacity, bool mainWillDonate,
    std::chrono::microseconds threadBusyWaitTime, bool paranoid,
    std::string_view poolName)
    : numWorkers(cpuIDs.size()),
      sharedState(runtimePtr, mainWillDonate, paranoid, numWorkers),
      outerRuntime(CompactRuntimePtr::getCurrentRuntime()),
      taskList(taskListCapacity), poolName(poolName) {
  assert(numWorkers <= kMaxWorkers && "too many workers for bitvec width");

  // Keeping numWorkers in a workerGroup a power of 2 to simplify arithmetic.
  multicastFactor = numWorkers > bitVectorWidth
                        ? static_cast<size_t>(std::ceil(std::log2(
                              numWorkers / static_cast<float>(bitVectorWidth))))
                        : 0;
  // Initialize each thread with its required state.
  // Note that we're constructing the array manually since WorkQueueThreads have
  // non-moveable atomics.
  workers = static_cast<WorkQueueThread *>(
      malloc(sizeof(WorkQueueThread) * numWorkers));
  assert(workers && "malloc of workers failed");
  for (size_t workerID = 0; workerID < numWorkers; ++workerID)
    new (workers + workerID) WorkQueueThread(
        sharedState, taskList, overflowMutex, overflowTaskList, workerID,
        cpuIDs[workerID], threadBusyWaitTime, this->poolName);

  if (mainWillDonate) {
    // Associate this thread with the given runtime, possibly overwriting
    // any existing runtime association.
    CompactRuntimePtr::setCurrentRuntime(runtimePtr);
  }
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
// Note we can't assert state == kShutdown since queue may be created
// and destroyed without ever being included in a runtime.
#if LLCL_WORKER_STATS
  llvm::dbgs() << "affinityEnqueueTime,affinityEnqueCount,taskListEnqueueTime,"
                  "taskListEnqueCount\n";
  llvm::dbgs() << affinityEnqueueTime << "," << affinityEnqueCount << ","
               << taskListEnqueueTime << "," << taskListEnqueCount << "\n";
#endif
  assert(!taskList.dequeue() &&
         "destroying ThreadPoolWorkQueue with pending work items");

  if (sharedState.mainWillDonate) {
    // Restore the association of this thread with the outer runtime, if any.
    CompactRuntimePtr::setCurrentRuntime(outerRuntime);
  }

  // Destroy all the threads datastructures.
  for (size_t i = 0; i < numWorkers; ++i)
    workers[i].~WorkQueueThread();
  free(workers);
}

void ThreadPoolWorkQueue::shutdown() {
#if MODULAR_PARANOID
  WorkQueueState expected = kReady;
  assert(sharedState.state.compare_exchange_strong(expected, kShuttingDown) &&
         "work pool is not ready");
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
  assert(sharedState.state.compare_exchange_strong(expected, kShutdown) &&
         "work pool is not shutting down");
#endif
}

void ThreadPoolWorkQueue::addTask(WorkItem &&workItem, int taskId) {
  assert(workItem);
#if MODULAR_PARANOID
  // This is not a true interlock, but will at least catch obvious
  // use-after-shutdowns.
  assert(sharedState.state != kShutdown &&
         "adding task to shutdown work queue");

  WorkQueueThread *callerWorker = getOwningWorkQueueThread();
  if (callerWorker && !workItem.use && !callerWorker->useStack.empty()) {
    // Propagate the current use (if any) onto this work item.
    workItem.use = callerWorker->useStack.back().copy();
  }
#endif
#if LLCL_WORKER_STATS
  auto start = std::chrono::high_resolution_clock::now();
#endif
  if (taskId >= 0) {
    auto workThread = getWorkQueueThread(taskId);
    // Either add to thread local lock-free queues or to its spill queue.
    // Any task with taskId >=0 always finds a place in either of these
    // two queues.
    workThread->addAffinityTask(std::move(workItem));
    // Wake up the thread just in case.
    // NOTE: This may be a spurious post() because the thread may already be
    // awake. It does not cause any harm because the worst that can happen
    // is that the thread goes to sleep the next iteration of runItemsImpl
    // rather than now.
    if (multicastFactor == 0) {
      if (sharedState.takeSuspendedThread(taskId))
        workThread->sema.post();
    } else {
      // TODO: post() should be low overhead if thread is already awake.
      // Nevertheless profile and check.
      workThread->sema.post();
    }
#if LLCL_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();
    atomicAdd(affinityEnqueCount, (uint64_t)1);
    atomicAdd(affinityEnqueueTime,
              std::chrono::duration<double, std::micro>(end - start).count());
#endif
    return;
  }
  // Try to add this work to the lock-free queue.
  if (taskList.enqueue(workItem)) {
    // If there are any suspended workers, kick one of them now to make sure
    // there's at least one worker still awake to pick up work.
    int workerIDToPoke = sharedState.takeAnySuspendedThread();
    if (workerIDToPoke != -1) {
      if (multicastFactor == 0)
        getWorkQueueThread(static_cast<size_t>(workerIDToPoke))->sema.post();
      else {
        size_t start = workerIDToPoke << multicastFactor;
        size_t range = 1 << multicastFactor;
        for (size_t i = start; i < start + range; i++) {
          if (i < numWorkers)
            getWorkQueueThread(i)->sema.post();
        }
      }
    }
#if LLCL_WORKER_STATS
    auto end = std::chrono::high_resolution_clock::now();
    atomicAdd(taskListEnqueCount, (uint64_t)1);
    atomicAdd(taskListEnqueueTime,
              std::chrono::duration<double, std::micro>(end - start).count());
#endif
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
  overflowTaskList.emplace_back(std::move(workItem));
}

void ThreadPoolWorkQueue::addLocalTask(WorkItem &&workItem) {
  assert(workItem && "invalid work item");
#if MODULAR_PARANOID
  // This is not a true interlock, but will at least catch obvious
  // use-after-shutdowns.
  assert(sharedState.state != kShutdown &&
         "adding local task to shutdown work queue");
#endif

  WorkQueueThread *callerWorker = getOwningWorkQueueThread();

#if MODULAR_PARANOID
  assert(
      callerWorker &&
      "cannot add local tasks from foreign threads in MODULAR_PARANOID mode");
#endif

  if (callerWorker == nullptr) {
    // Called from a foreign thread, so there's no local task list we can
    // enqueue to on this thread. Add as a task instead.
    addTask(std::move(workItem));
    return;
  }

#if MODULAR_PARANOID
  if (!workItem.use && !callerWorker->useStack.empty())
    // Propagate the current use (if any) onto this work item.
    workItem.use = callerWorker->useStack.back().copy();
#endif

  // Called from either a worker thread or the 'main' thread. Safe to enqueue
  // directly.
  callerWorker->addLocalTask(std::move(workItem));
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
#if MODULAR_PARANOID
  // This is not a true interlock, but will at least catch obvious
  // use-after-shutdowns.
  assert(sharedState.state == kReady &&
         "awaiting work queue which is not ready");
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
      value.andThenSync([&numRemaining, awaitingWorker
#if MODULAR_PARANOID
                         ,
                         this
#endif
      ]() {
        // Decrement the count of async values that we're waiting on.
        // TODO: This can probably use more relaxed memory consistency!
        if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
          return;
#if MODULAR_PARANOID
        // Exclude this waiter from any lifetime assertions before the awaiter
        // can continue.
        taskIsDone();
#endif

        // When it drops to zero, we're good to go and whatever thread is
        // waiting for this will exit out of its 'runItems' loop.  That said,
        // the thread may be suspended on a semaphore.  Check for this, and if
        // so, signal its semaphore so it wakes up and notes that it is done.
        // If the worker doing the await() has suspended, make sure to wake it
        // up so it notices that it is done.
        // NOTE: This may be a spurious post() because the thread may already be
        // awake. It does not cause any harm because the worst that can happen
        // is that the thread goes to sleep the next iteration of runItemsImpl
        // rather than now.
        awaitingWorker->sema.post();
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
      value.andThenSync([&numRemaining, &sema
#if MODULAR_PARANOID
                         ,
                         this
#endif
      ]() {
        // Decrement the count of async values that we're waiting on.
        // TODO: This can probably use more relaxed memory consistency!
        if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) != 1)
          return;

#if MODULAR_PARANOID
        // Exclude this waiter from any lifetime assertions before the
        // awaiter can continue.
        taskIsDone();
#endif

        sema.post();
      });
    }
    sema.wait();
  }

  assert(numRemaining.load() == 0 &&
         "exited await loop without all values being ready");
#if MODULAR_PARANOID
  // Try to catch if the runtime was torn down while we were awaiting.
  assert(sharedState.state != kShutdown &&
         "work queue was shutdown while waiting");
#endif
}

bool ThreadPoolWorkQueue::callerIsForeign() const {
  return getOwningWorkQueueThread() == nullptr;
}

//===----------------------------------------------------------------------===//
// createThreadPoolWorkQueue entrypoint
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue> M::AsyncRT::createThreadPoolWorkQueue(
    CompactRuntimePtr runtimePtr, size_t numThreads, size_t maxThreads,
    bool mainWillDonate, bool withAffinity,
    std::chrono::microseconds threadBusyWaitTime, std::string_view poolName,
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
#if LLCL_NO_AFFINITY
  withAffinity = false;
#endif // LLCL_NO_AFFINITY
  // Using numThreads as a hint, figure out a CPU for each worker thread and
  // the main thread. The CPU ids may end up as kNoAffinity, but the vector
  // size will still guide the construction of worker threads.
  //
  // TODO: This function should return the error back to caller.
  if (maxThreads == 0 || maxThreads > kMaxWorkers)
    maxThreads = kMaxWorkers;
  auto cpuIDOr = getThreadAffinityCpuIds(withAffinity, numThreads, maxThreads);
  if (cpuIDOr.isError())
    llvm::report_fatal_error(cpuIDOr.getError());
  std::vector<size_t> cpuIDs = std::move(*cpuIDOr);
  assert(!cpuIDs.empty() && "no cpu ids");

  // cpuIDs.size() is guaranteed to be at least 1 here.
  const size_t taskListCapacity = cpuIDs.size() * kTaskListSlotsPerThread;
  LLVM_DEBUG(llvm::dbgs()
             << "createThreadPoolWorkQueue: Task list has capacity of at least "
             << taskListCapacity << " slots.\n");

  return std::make_unique<ThreadPoolWorkQueue>(
      runtimePtr, cpuIDs, taskListCapacity, mainWillDonate, threadBusyWaitTime,
      paranoid, poolName);
}
