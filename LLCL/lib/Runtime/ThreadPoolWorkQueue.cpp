//===- ThreadPoolWorkQueue.cpp --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/WorkQueue.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/LockFreeRingBuffer.h"
#include "LLCL/Support/Semaphore.h"
#include "LLCL/Support/SpinWaiter.h"
#include "Support/CommandLine.h"
#include "llvm/ADT/ArrayRef.h"

using namespace LLCL;
using llvm::ArrayRef;
using llvm::Optional;
using mlir::failure;
using mlir::LogicalResult;
using mlir::success;

// Optionally enable tracing.
//
// TRACE_LEVEL is the density of logging.  0 = Off, 1 = Events, 2 = All work
// items, 3 includes shutdown activity.
#define TRACE_LEVEL 0
#include "PrintfTracing.h"

/// This value is set to a number for workqueue threads.
static thread_local ssize_t threadIDInTLS = -1;

/// TRACE - When trace level is >= the specified level, we print the log.
#define TRACE(LEVEL, FORMAT, ...)                                              \
  TRACE_IMPL(threadPoolNumber + 1, LEVEL, FORMAT, ##__VA_ARGS__)

/// CTX_TRACE - This is used in contexts that don't have a threadPoolNumber
/// available.  This pulls the thread pool # out of TLS.
#define CTX_TRACE(LEVEL, FORMAT, ...)                                          \
  TRACE_IMPL(threadIDInTLS + 1, LEVEL, FORMAT, ##__VA_ARGS__)

//===----------------------------------------------------------------------===//
// WorkQueueThread
//===----------------------------------------------------------------------===//

/// Pop a single item off the queue and do the task.  This returns failure if
/// the queue is empty.
static LogicalResult popAndDoWork(LockFreeRingBuffer<TaskFunction> &q,
                                  size_t threadPoolNumber) {
  auto callable = q.dequeue();
  if (!callable) // If the queue is empty, return failure.
    return failure();

  TRACE(2, "work start.");
  callable();
  TRACE(2, "work end.");
  return success();
}

namespace {
/// Provides the state needed to synchronize the threads in the thread pool
/// for the required exit functionality.
struct SharedThreadState {
  std::atomic<bool> doneFlag;
  Semaphore sema;
  std::chrono::nanoseconds busyWaitNs;
};
} // end anonymous namespace

namespace {
/// RAII wrapper around a thread to simplify handling of each thread in the
/// thread pool.
struct WorkQueueThread {
  SharedThreadState &sharedState;
  LockFreeRingBuffer<TaskFunction> &taskList;
  size_t threadPoolNumber;
  std::thread thread;

  /// Create a `WorkQueueThread` from a sync state reference and a reference to
  /// a task list. This also starts the std::thread, so the sync state and task
  /// list must be initialized by the time this is called.
  WorkQueueThread(SharedThreadState &sharedState,
                  LockFreeRingBuffer<TaskFunction> &taskList,
                  size_t threadPoolNumber)
      : sharedState(sharedState), taskList(taskList),
        threadPoolNumber(threadPoolNumber),
        thread(&WorkQueueThread::run, this) {}

  WorkQueueThread(WorkQueueThread &&) = default;

  /// Joins the thread. Asserts that `sharedState.done` is true because
  /// otherwise the thread will never join.
  ~WorkQueueThread() {
    assert(sharedState.doneFlag.load() &&
           "Must not destroy a WorkQueueThread object that is not pending "
           "completion.");
    thread.join();
  }

  /// The main run function run by std::thread.
  void run();
};
} // end anonymous namespace

void WorkQueueThread::run() {
  // Set the current thread ID # in thread local storage so we can find it later
  // when re-entering.
  threadIDInTLS = threadPoolNumber;

  // On systems that support it, give the thread a symbolic name that will show
  // up in profilers and debuggers.
  // TODO: I think this is widely supported on linux and windows apparently has
  // SetThreadName.
#ifdef __APPLE__
  char threadName[30];
  sprintf(threadName, "LLCL TPWQ Thread %d", (int)threadPoolNumber);
  pthread_setname_np(threadName);
#endif

  TRACE(1, "worker starting.");

  // Continuously execute work units.
KeepRunning:
  while (true) {
    // In the normal case we happily pick up and do work.
    if (succeeded(popAndDoWork(taskList, threadPoolNumber)))
      continue;

    // If we've run out of work to do, we need to quiesce and ultimately block
    // in the kernel on the semaphore.  However, we don't want to immediately
    // give up hope, because we may be "right about to" get new work incoming.
    // We also want to make sure to use exponential backoff to avoid pummeling
    // the memory hierarchy of the threads that are doing useful work.  As such,
    // we use a BusyWaitSpinWaiter.
    BusyWaitSpinWaiter spinWaiter(sharedState.busyWaitNs);

    // Spin until we find some work to do.
    while (!spinWaiter.wait()) {
      // If we ever succeed in finding work to do, go back to running like
      // normal.
      if (succeeded(popAndDoWork(taskList, threadPoolNumber)))
        goto KeepRunning;
    }

    TRACE(1, "worker sleeping.");

    // Otherwise, we we've waited long enough, yield the thread to the OS so we
    // don't burn power and starve other tasks on the system.
    sharedState.sema.wait();

    TRACE(1, "worker woke.");

    // On wakeup, check to see if we're supposed to shutdown.  If so, wind down
    // the thread.
    if (sharedState.doneFlag.load(std::memory_order_acquire)) {
      TRACE(3, "worker destroying.");
      return;
    }
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
  explicit ThreadPoolWorkQueue(size_t numWorkerThreads,
                               std::chrono::nanoseconds busyWaitNs);
  /// Cleans up all threads in the thread pool cleanly.
  ~ThreadPoolWorkQueue() override;

  void addTask(TaskFunction work) override {
    // Try to add this work to the RingBuffer.  If that fails, then the ring
    // buffer is full: we take an item out of queue and do it to try to make
    // more space then try again.
    CTX_TRACE(1, "addTask\t\t\t[%p %p]", __builtin_return_address(0),
              __builtin_return_address(1));
    while (!taskList->enqueue(work)) {
      ssize_t threadPoolNumber = -2; // TODO.
      TRACE(1, "WORK QUEUE FULL.");
      [[maybe_unused]] auto r = popAndDoWork(*taskList, threadPoolNumber);
    }
    sharedState.sema.post();
  }

  void await(ArrayRef<AnyAsyncValueRef> values) override;

  int getParallelismLevel() const final {
    // `poolSize` is set to the number of worker threads that are created by
    // the work queue. However, we expect to have an external "main" thread
    // that has an access to the work queue by calling "await". Therefore, we
    // return `poolSize + 1` here.
    return poolSize + 1;
  }

private:
  const size_t poolSize;
  std::vector<WorkQueueThread> pool;

  // Base synchronization state is held in this class, each thread holds a
  // reference to this structure.
  SharedThreadState sharedState;
  std::unique_ptr<LockFreeRingBuffer<TaskFunction>> taskList;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ThreadPoolWorkQueue function implementations
//===----------------------------------------------------------------------===//

ThreadPoolWorkQueue::ThreadPoolWorkQueue(size_t numWorkerThreads,
                                         std::chrono::nanoseconds busyWaitNs)
    : poolSize(numWorkerThreads), sharedState{false, {}, busyWaitNs} {
  taskList = std::make_unique<LockFreeRingBuffer<TaskFunction>>();

  pool.reserve(poolSize);
  // Initialize each thread with its required state.
  for (size_t i = 0; i < poolSize; ++i)
    pool.emplace_back(sharedState, *taskList, i);
}

ThreadPoolWorkQueue::~ThreadPoolWorkQueue() {
  int threadPoolNumber = threadIDInTLS;
  TRACE(3, "~ThreadPoolWorkQueue() start.");

  // Donate the client thread to help empty the queue if there's anything left.
  while (succeeded(popAndDoWork(*taskList, threadPoolNumber)))
    ;

  // Now we can tell all the threads to exit.
  sharedState.doneFlag.store(true, std::memory_order_release);

  // Post on the semaphore for every thread to wake it if it's waiting.
  for (size_t i = 0; i < poolSize; ++i)
    sharedState.sema.post();

  // Call the destructors to join the threads.
  pool.clear();

  TRACE(3, "~ThreadPoolWorkQueue() done.");
}

void ThreadPoolWorkQueue::await(ArrayRef<AnyAsyncValueRef> values) {
  // Get the current thread ID from TLS.
  ssize_t threadPoolNumber = threadIDInTLS;

  TRACE(1, "await() start.\t\t[%p %p]", __builtin_return_address(0),
        __builtin_return_address(1));

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
      if (numRemaining.fetch_sub(1, std::memory_order_seq_cst) == 1) {
        allValuesDone.post();
        // This trace is useful when looking at latency between work completing
        // and await returning in the blocked thread, but right now that isn't
        // the problem we're working on solving.
        // CTX_TRACE(1, "await() work completed.");
      }
    });

  // Donate the client thread to doing useful work until there's no more useful
  // work to do.
KeepRunning:
  // TODO: This can probably use more relaxed memory consistency!
  while (numRemaining.load(std::memory_order_seq_cst) != 0) {
    // While we are waiting, we might as well do work for other tasks that need
    // to be done.  Take a work item and do it.
    if (succeeded(popAndDoWork(*taskList, threadPoolNumber)))
      continue;

    // Otherwise if we ran out of work to do, and we are still waiting on
    // things, then other threads must be doing the work we are waiting on.
    // Do a busy wait for awhile, and eventually block this thread on the
    // 'allValuesDone' semaphore as needed.
    BusyWaitSpinWaiter spinWaiter(sharedState.busyWaitNs);

    // Spin until we find some work to do.
    while (!spinWaiter.wait()) {
      // If we ever succeed in finding work to do, go back to running like
      // normal.
      if (succeeded(popAndDoWork(*taskList, threadPoolNumber)))
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
    TRACE(1, "await() SUSPEND.");
    break;
  }

  // Ok, we successfully saw that all values are done.  Do a final wait on the
  // semaphore to make sure the last 'andThen' block has executed the post.  We
  // don't want them to fire after this function has been returned, because that
  // will destory 'allValuesDone' itself.
  allValuesDone.wait();
  assert(numRemaining.load() == 0);
  TRACE(1, "await() returning.");
}

//===----------------------------------------------------------------------===//
// createThreadPoolWorkQueue entrypoint
//===----------------------------------------------------------------------===//

std::unique_ptr<WorkQueue>
LLCL::createThreadPoolWorkQueue(size_t numThreads, unsigned busyWaitNs) {
  TRACE_INIT_START_TIME();

  // We expect `numThreads` to be the total numbers of threads that are
  // accessing the work queue. As there will be an external thread that will
  // access the work queue and take items from it by calling `await`, we create
  // `numThreads - 1` worker threads from the thread pool work queue.
  assert(numThreads > 0);
  return std::make_unique<ThreadPoolWorkQueue>(
      numThreads - 1, std::chrono::nanoseconds(busyWaitNs));
}
