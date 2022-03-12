//===- ThreadPoolWorkQueue.cpp --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/ConcurrentQueue.h"
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

  void addTask(TaskFunction work) override {
    taskList.enqueue(std::move(work));
  }

  void await(llvm::ArrayRef<RCRef<AsyncValue>> values) override;

private:
  /// Pop a single item off the queue and do the task.
  static mlir::LogicalResult popAndDoWork(ConcurrentQueue<TaskFunction> &q) {
    mlir::FailureOr<TaskFunction> frontOr = q.dequeue();
    if (succeeded(frontOr))
      (std::move(*frontOr))();

    return frontOr;
  }

  /// Loop around `popAndDoWork`, just do work until the queue is empty.
  static void doWork(ConcurrentQueue<TaskFunction> &q) {
    while (succeeded(popAndDoWork(q)))
      ;
  }

  /// Provides the state needed to synchronize the threads in the thread pool
  /// for the required exit functionality.
  struct ThreadSyncState {
    std::atomic<bool> done;
  };

  /// RAII wrapper around a thread to simplify handling of each thread in the
  /// thread pool.
  struct Thread {
    ThreadSyncState &sync;
    ConcurrentQueue<TaskFunction> &taskList;

    std::thread thread;

    /// Create a `Thread` from a sync state reference and a reference to a task
    /// list. This also starts the std::thread, so the sync state and task list
    /// must be initialized by the time this is called.
    Thread(ThreadSyncState &sync, ConcurrentQueue<TaskFunction> &taskList)
        : sync(sync), taskList(taskList), thread(&Thread::run, this) {}
    /// Joins the thread. Asserts that `sync.done` is true because otherwise the
    /// thread will never join.
    ~Thread() {
      assert(
          sync.done.load() &&
          "Must not destroy a Thread object that is not pending completion.");
      thread.join();
    }

    /// Thread's main run function. Loops until (1) the work queue is empty, and
    /// (2) `sync.done` is set to true, at which point it exits gracefully.
    void run();
  };

  const size_t poolSize;
  // Uses a raw pointer here because operator new[] doesn't allow constructor
  // arguments.
  Thread *pool;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  ThreadSyncState syncState;
  ConcurrentQueue<TaskFunction> taskList;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue function implementations
//===----------------------------------------------------------------------===//

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numThreads)
    : poolSize(numThreads),
      pool((Thread *)malloc(poolSize * sizeof(Thread))), syncState{false} {
  // Initialize each thread with its required state.
  for (size_t i = 0; i < poolSize; ++i)
    new (&pool[i]) Thread(syncState, taskList);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
  // Donate the client thread to help empty the queue if there's anything left.
  doWork(taskList);

  // Now we can tell all the threads to exit.
  syncState.done.store(true, std::memory_order_release);
  // Call the destructor.
  for (size_t i = 0; i < poolSize; ++i)
    pool[i].~Thread();

  // Free the memory we allocated with calloc.
  free(pool);
}

void ThreadPoolWorkQueue::await(llvm::ArrayRef<RCRef<AsyncValue>> values) {
  // We are done when values_remaining drops to zero.
  std::atomic<size_t> numRemaining = values.size();

  // As each value becomes available, we can decrement our counts.
  for (auto &value : values)
    value->andThen([&numRemaining]() { --numRemaining; });

  // Donate the client thread to popping tasks off the queue. `popAndDoWork`
  // will return failure if `taskList.dequeue()` returns failure, which
  // indicates there's nothing in the queue. This could mean that something has
  // already been kicked-off and will enqueue more work in the process of
  // executing, but we need to wait for it to complete for those tasks to become
  // visible.
  while (numRemaining.load() > 0)
    if (mlir::failed(popAndDoWork(taskList)))
      std::this_thread::yield();
}

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue::ThreadContext implementation
//===----------------------------------------------------------------------===//

void ThreadPoolWorkQueue::Thread::run() {
  // While we haven't been told to finish up, attempt to dequeue and execute
  // work.
  while (true) {
    if (mlir::succeeded(popAndDoWork(taskList)))
      continue;

    if (sync.done.load(std::memory_order_acquire))
      return;

    std::this_thread::yield();
  }
}

//===----------------------------------------------------------------------===//
// LLCL top level implementations
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue> LLCL::createThreadPoolWorkQueue(size_t numThreads) {
  return std::make_unique<ThreadPoolWorkQueue>(
      numThreads == 0 ? std::thread::hardware_concurrency() : numThreads);
}
