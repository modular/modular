//===- ThreadPoolWorkQueue.cpp --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/ConcurrentQueue.h"
#include "LLCL/Support/Semaphore.h"
#include "llvm/ADT/ArrayRef.h"

#include <thread>
using namespace LLCL;

namespace {
/// This class provides a thread-pool that implements the WorkQueue interface.
/// It starts a dynamic number of threads and distributes work to it by means of
/// a concurrent-safe queue.
class ThreadPoolWorkQueue : public WorkQueue {
public:
  /// Initialize the thread pool and start up the worker threads. By the time
  /// the constructor finishes, all the worker threads have started and shall
  /// only be cancelled by the destructor.
  explicit ThreadPoolWorkQueue(size_t numThreads);
  /// Cleans up all threads in the thread pool cleanly.
  ~ThreadPoolWorkQueue() override;

  void await(llvm::ArrayRef<AnyAsyncValueRef> values) override;
  int getParallelismLevel() const final { return poolSize; }

protected:
  void addTaskInternal(TaskFunctionBase *work) override {
    taskList.enqueue(work);
    syncState.sema.post();
  }

private:
  /// Pop a single item off the queue and do the task.
  static mlir::LogicalResult
  popAndDoWork(ConcurrentQueue<TaskFunctionBase> &q) {
    auto item = q.dequeue();
    if (!item)
      return mlir::failure();

    item->call();
    return mlir::success();
  }

  /// Loop around `popAndDoWork`, just do work until the queue is empty.
  static void doWork(ConcurrentQueue<TaskFunctionBase> &q) {
    while (succeeded(popAndDoWork(q)))
      ;
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
    ConcurrentQueue<TaskFunctionBase> &taskList;

    std::thread thread;

    /// Create a `Thread` from a sync state reference and a reference to a
    /// task list. This also starts the std::thread, so the sync state and
    /// task list must be initialized by the time this is called.
    Thread(ThreadSyncState &sync, ConcurrentQueue<TaskFunctionBase> &taskList)
        : sync(sync), taskList(taskList), thread(&Thread::run, this) {}
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
  ConcurrentQueue<TaskFunctionBase> taskList;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue function implementations
//===----------------------------------------------------------------------===//

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numThreads)
    : poolSize(numThreads),
      pool((Thread *)malloc(poolSize * sizeof(Thread))), syncState{false, {}} {
  // Initialize each thread with its required state.
  for (size_t i = 0; i < poolSize; ++i)
    new (&pool[i]) Thread(syncState, taskList);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
  // Donate the client thread to help empty the queue if there's anything left.
  doWork(taskList);

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

void ThreadPoolWorkQueue::await(llvm::ArrayRef<AnyAsyncValueRef> values) {
  // We are done when values_remaining drops to zero.
  std::atomic<size_t> numRemaining = values.size();

  // Set up a private semaphore so we can just wait on the values that we care
  // about finishing, without waiting on the whole work queue's semaphore. This
  // is applicable in the case where we are waiting on something, but there's no
  // new work being added.
  Semaphore allValuesDone;

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value->andThen([&numRemaining, &allValuesDone]() {
      --numRemaining;
      allValuesDone.post();
    });

  // Donate the client thread to doing useful work until there's no more useful
  // work to do. The thread should wake up and this function should return as
  // soon as the work that it's waiting on has finished.
  // TODO: This code has a problem - once the taskList has been drained the
  //   client thread is now sleeping on its semaphore. If someone else adds more
  //   work, this thread currently has no way of waking up to check again if
  //   there's more work to be done.
  while (numRemaining.load() > 0)
    if (mlir::failed(popAndDoWork(taskList)))
      for (size_t i = 0, e = values.size(); i < e; ++i)
        allValuesDone.wait();
}

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue::ThreadContext implementation
//===----------------------------------------------------------------------===//

void ThreadPoolWorkQueue::Thread::run() {
  // While we haven't been told to finish up, attempt to dequeue and execute
  // work.
  while (true) {
    // Wait for any work that might be on its way in. If there's no work, then
    // this thread will be slept by the kernel.
    sync.sema.wait();

    if (mlir::succeeded(popAndDoWork(taskList)))
      continue;

    if (sync.done.load(std::memory_order_acquire))
      return;
  }
}

//===----------------------------------------------------------------------===//
// LLCL top level implementations
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue> LLCL::createThreadPoolWorkQueue(size_t numThreads) {
  return std::make_unique<ThreadPoolWorkQueue>(
      numThreads == 0 ? std::thread::hardware_concurrency() : numThreads);
}
