//===- ThreadPoolWorkQueue.cpp --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/LockFreeRingBuffer.h"
#include "LLCL/Support/Semaphore.h"
#include "LLCL/Support/Signposts.h"
#include "LLCL/Support/SpinWaiter.h"
#include "Support/CommandLine.h"
#include "llvm/ADT/ArrayRef.h"
#include <thread>

using namespace LLCL;
using llvm::ArrayRef;
using llvm::Optional;
using mlir::failure;
using mlir::LogicalResult;
using mlir::success;

namespace {
/// This is like SpinWaiter but uses explensive syscalls to optionally hard-wait
/// for long periods of time.
class SemaphoreSpinWaiter {
public:
  SemaphoreSpinWaiter(std::chrono::nanoseconds busyWaitTime)
      : busyWaitTime(busyWaitTime) {}

  /// Wait for another step using progressively more heavy-weight mechanisms.
  /// This returns true if we should block on a semaphore.
  bool wait() {
    // If we are cheap-waiting, just return quickly.
    if (!waiter.isDoneWithNopSpins()) {
      waiter.wait();
      return false;
    }

    // Otherwise we're going to intentionally burn time.  If this is the first
    // iteration of this, figure out what wall time we are.
    //
    // NOTE: Busy-waiting logic below calls `std::chrono::steady_clock::now()`
    // from the loop, which may perform expensive operations in its
    // implementation that make busy-waiting not working as expected.
    // https://github.com/modularml/modular/issues/1092 for monitoring this.
    if (busyWaitTime != std::chrono::nanoseconds::zero()) {
      if (!busyWaitEndTime.hasValue())
        busyWaitEndTime = std::chrono::steady_clock::now() + busyWaitTime;

      // If we haven't reached out busy wait end time, then continue spinning.
      if (std::chrono::steady_clock::now() < *busyWaitEndTime)
        return false;
    }

    return true;
  }

private:
  /// This is a spin waiter that never yields to the OS with sched_yield etc.
  /// We would rather block on the semaphore.
  SpinWaiter<false> waiter;

  /// This is how long to spin on the waiter.
  /// TODO: This should eventually go away or turn into a constant.
  std::chrono::nanoseconds busyWaitTime;

  /// This is the time we should stop busy waiting.
  Optional<std::chrono::steady_clock::time_point> busyWaitEndTime;
};
} // end anonymous namespace

namespace {

/// This class provides a thread-pool that implements the WorkQueue interface.
/// It starts a dynamic number of threads and distributes work to it by means of
/// a concurrent-safe queue.
class ThreadPoolWorkQueue : public WorkQueue {
public:
  /// Initialize the thread pool and start up the worker threads. By the time
  /// the constructor finishes, all the worker threads have started and shall
  /// only be cancelled by the destructor.
  explicit ThreadPoolWorkQueue(size_t numWorkerThreads, unsigned busyWaitNs);
  /// Cleans up all threads in the thread pool cleanly.
  ~ThreadPoolWorkQueue() override;

  void addTask(TaskFunction work) override {
    // Try to add this work to the RingBuffer.  If that fails, then the ring
    // buffer is full: we take an item out of queue and do it to try to make
    // more space then try again.
    while (!taskList->enqueue(work)) {
      [[maybe_unused]] auto r = popAndDoWork(*taskList);
    }
    syncState.sema.post();
  }

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  int getParallelismLevel() const final {
    // `poolSize` is set to the number of worker threads that are created by the
    // work queue. However, we expect to have an external "main" thread that
    // has an access to the work queue by calling "await". Therefore, we return
    // `poolSize + 1` here.
    return poolSize + 1;
  }

private:
  /// Pop a single item off the queue and do the task.
  static LogicalResult popAndDoWork(LockFreeRingBuffer<TaskFunction> &q) {
    auto callable = q.dequeue();
    if (!callable)
      return failure();
    callable();
    return success();
  }

  /// Provides the state needed to synchronize the threads in the thread pool
  /// for the required exit functionality.
  struct ThreadSyncState {
    std::atomic<bool> doneFlag;
    Semaphore sema;
  };

  /// RAII wrapper around a thread to simplify handling of each thread in the
  /// thread pool.
  struct Thread {
    ThreadSyncState &sync;
    LockFreeRingBuffer<TaskFunction> &taskList;
    size_t threadPoolNumber;
    std::chrono::nanoseconds busyWaitNs;
    std::thread thread;

    /// Create a `Thread` from a sync state reference and a reference to a
    /// task list. This also starts the std::thread, so the sync state and
    /// task list must be initialized by the time this is called.
    Thread(ThreadSyncState &sync, LockFreeRingBuffer<TaskFunction> &taskList,
           size_t threadPoolNumber, unsigned busyWaitNs)
        : sync(sync), taskList(taskList), threadPoolNumber(threadPoolNumber),
          busyWaitNs(busyWaitNs), thread(&Thread::run, this) {}

    /// Joins the thread. Asserts that `sync.done` is true because otherwise
    /// the thread will never join.
    ~Thread() {
      assert(
          sync.doneFlag.load() &&
          "Must not destroy a Thread object that is not pending completion.");
      thread.join();
    }

    /// Thread's main run function. Loops until (1) the work queue is empty,
    /// and (2) `sync.done` is set to true, at which point it exits
    /// gracefully.
    void run();
  };

  const size_t poolSize;
  // Uses a raw pointer here because operator new[] doesn't allow constructor
  // arguments.
  Thread *pool;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  ThreadSyncState syncState;
  std::unique_ptr<LockFreeRingBuffer<TaskFunction>> taskList;

  // busy wait duration in nanoseconds.
  std::chrono::nanoseconds busyWaitNs;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue function implementations
//===----------------------------------------------------------------------===//

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numWorkerThreads,
                                         unsigned busyWaitNs)
    : poolSize(numWorkerThreads),
      pool((Thread *)malloc(poolSize * sizeof(Thread))), syncState{false, {}},
      busyWaitNs(busyWaitNs) {
  taskList = std::make_unique<LockFreeRingBuffer<TaskFunction>>();
  // Initialize each thread with its required state.
  for (size_t i = 0; i < poolSize; ++i)
    new (&pool[i]) Thread(syncState, *taskList, i, busyWaitNs);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
  // Donate the client thread to help empty the queue if there's anything left.
  while (succeeded(popAndDoWork(*taskList)))
    ;

  // Now we can tell all the threads to exit.
  syncState.doneFlag.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake it if it's waiting.
  for (size_t i = 0; i < poolSize; ++i)
    syncState.sema.post();

  // Call the destructors to join the threads.
  for (size_t i = 0; i < poolSize; ++i)
    pool[i].~Thread();

  // Free the memory we allocated with malloc.
  free(pool);
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
  //  We are done when values_remaining drops to zero.
  std::atomic<size_t> numRemaining = values.size();

  // Set up a private semaphore so we can just wait on the values that we care
  // about finishing, without waiting on the whole work queue's semaphore. This
  // is applicable in the case where we are waiting on something, but there's no
  // new work being added.
  Semaphore allValuesDone;

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value->andThen([&numRemaining, &allValuesDone]() {
      // TODO: This can probably use more relaxed memory consistency!
      if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) == 1)
        allValuesDone.post();
    });

  // Donate the client thread to doing useful work until there's no more useful
  // work to do.
KeepRunning:
  // TODO: This can probably use more relaxed memory consistency!
  while (numRemaining.load(std::memory_order_seq_cst) != 0) {
    // While we are waiting, we might as well do work for other tasks that need
    // to be done.  Take a work item and do it.
    if (succeeded(popAndDoWork(*taskList)))
      continue;

    // Otherwise if we ran out of work to do, and we are still waiting on
    // things, then other threads must be doing the work we are waiting on.
    // Do a busy wait for awhile, and eventually block this thread on the
    // 'allValuesDone' semaphore as needed.
    SemaphoreSpinWaiter spinWaiter(busyWaitNs);

    // Spin until we find some work to do.
    while (!spinWaiter.wait()) {
      // If we ever succeed in finding work to do, go back to running like
      // normal.
      if (succeeded(popAndDoWork(*taskList)))
        goto KeepRunning;

      // If we successfully resolved all the AV's then we're done.
      // TODO: This can probably use more relaxed memory consistency!
      if (numRemaining.load(std::memory_order_seq_cst) == 0)
        break;
    }

    // If we've waited long enough, block on the `allValuesDone` semaphore to
    // yield the thread to the OS so we don't burn power and starve other tasks
    // on the system.
    //
    // TODO: This code has a problem - once the taskList has been drained the
    // client thread is now sleeping on its semaphore.  If someone else adds
    // more work for the system to do, this thread currently has no way of
    // waking up to help out with that.
    break;
  }

  // Ok, we successfully saw that all values are done.  Do a final wait on the
  // semaphore to make sure the last 'andThen' block has executed the post.  We
  // don't want them to fire after this function has been returned, because that
  // will destory 'allValuesDone' itself.
  allValuesDone.wait();
  assert(numRemaining.load() == 0);
}

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue::ThreadContext implementation
//===----------------------------------------------------------------------===//

void ThreadPoolWorkQueue::Thread::run() {
  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  // TODO: I think this is widely supported on linux and windows apparently has
  // SetThreadName.
#ifdef __APPLE__
  char threadName[30];
  sprintf(threadName, "LLCL TPWQ Thread %d", (int)threadPoolNumber);
  pthread_setname_np(threadName);
#endif

  // Continuously execute work units.
KeepRunning:
  while (true) {
    // In the normal case we happily pick up and do work.
    if (succeeded(popAndDoWork(taskList)))
      continue;

    // If we've run out of work to do, we need to quiesce and ultimately block
    // in the kernel on the semaphore.  However, we don't want to immediately
    // give up hope, because we may be "right about to" get new work incoming.
    // We also want to make sure to use exponential backoff to avoid pummeling
    // the memory hierarchy of the threads that are doing useful work.  As such,
    // we use a SemaphoreSpinWaiter.
    SemaphoreSpinWaiter spinWaiter(busyWaitNs);

    // Spin until we find some work to do.
    while (!spinWaiter.wait()) {
      // If we ever succeed in finding work to do, go back to running like
      // normal.
      if (succeeded(popAndDoWork(taskList)))
        goto KeepRunning;
    }

    // Otherwise, we we've waited long enough, yield the thread to the OS so we
    // don't burn power and starve other tasks on the system.
    sync.sema.wait();

    // On wakeup, check to see if we're supposed to shutdown.  If so, wind down
    // the thread.
    if (sync.doneFlag.load(std::memory_order_acquire))
      return;
  }
}

//===----------------------------------------------------------------------===//
// LLCL top level implementations
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue>
LLCL::createThreadPoolWorkQueue(size_t numThreads, unsigned busyWaitNs) {
  if (numThreads == 0)
    numThreads = std::thread::hardware_concurrency();
  // We expect `numThreads` to be the total numbers of threads that are
  // accessing the work queue. As there will be an external thread that will
  // access the work queue and take items from it by calling `await`, we create
  // `numThreads - 1` worker threads from the thread pool work queue.
  return std::make_unique<ThreadPoolWorkQueue>(numThreads - 1, busyWaitNs);
}
