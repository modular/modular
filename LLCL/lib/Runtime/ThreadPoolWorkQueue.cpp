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
#include "Support/CommandLine.h"
#include "llvm/ADT/ArrayRef.h"
#include <thread>

using namespace LLCL;
using llvm::ArrayRef;
using mlir::failure;
using mlir::LogicalResult;
using mlir::success;

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

  /// Loop around `popAndDoWork`, just do work until the queue is empty.
  static void doWork(LockFreeRingBuffer<TaskFunction> &q) {
    while (succeeded(popAndDoWork(q)))
      ;
  }

  /// Performs busy-waiting until `cond()` returns mlir::success(). If that
  /// doesn't happen for `busyWait` duration, start passive-waiting with
  /// the semaphore `sema`.
  template <typename CondFn, typename DurationT>
  static void busyWaitThenBlock(Semaphore &sema, DurationT busyWait,
                                CondFn cond) {
    if (succeeded(cond()))
      return;

    // Busy-wait for a given duration.
    // NOTE: Busy-waiting logic below calls `std::chrono::steady_clock::now()`
    // from the loop, which may perform expensive operations in its
    // implementation that make busy-waiting not working as expected.
    // https://github.com/modularml/modular/issues/1092 for monitoring this.
    if (busyWait != DurationT::zero()) {
      auto busyWaitUntil =
          std::chrono::steady_clock::now() + DurationT(busyWait);
      while (busyWaitUntil > std::chrono::steady_clock::now())
        if (succeeded(cond()))
          return;
    }

    // Start passive waiting.
    sema.wait();
  }

  /// Provides the state needed to synchronize the threads in the thread pool
  /// for the required exit functionality.
  struct ThreadSyncState {
    std::atomic<bool> done;
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
          sync.done.load() &&
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
  doWork(*taskList);

  // Now we can tell all the threads to exit.
  syncState.done.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake it if it's waiting.
  for (size_t i = 0; i < poolSize; ++i)
    syncState.sema.post();

  // Call the destructor.
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
      allValuesDone.post();
      --numRemaining;
    });

  // Donate the client thread to doing useful work until there's no more useful
  // work to do. The thread should wake up and this function should return as
  // soon as the work that it's waiting on has finished.
  // TODO: This code has a problem - once the taskList has been drained the
  //   client thread is now sleeping on its semaphore. If someone else adds more
  //   work, this thread currently has no way of waking up to check again if
  //   there's more work to be done.
  auto busyWaitCond = [this, &numRemaining]() {
    if (numRemaining.load() == 0 || succeeded(popAndDoWork(*taskList)))
      return success();
    return failure();
  };
  while (numRemaining.load() > 0)
    if (failed(popAndDoWork(*taskList)))
      for (auto &value : values)
        if (!value->isReady())
          busyWaitThenBlock(allValuesDone, busyWaitNs, busyWaitCond);
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

  // While we haven't been told to finish up, attempt to dequeue and execute
  // work.
  while (true) {
    // Wait for any work that might be on its way in. If there's no work, then
    // this thread will be slept by the kernel.
    busyWaitThenBlock(sync.sema, busyWaitNs,
                      [this]() { return popAndDoWork(taskList); });

    if (succeeded(popAndDoWork(taskList)))
      continue;

    if (sync.done.load(std::memory_order_acquire))
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
