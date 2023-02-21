//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::LLCL::WorkQueue interface, which allows clients of
// LLCL to implement work queues that map onto their systems in a nice way.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_WORKQUEUE_H
#define LLCL_RUNTIME_WORKQUEUE_H

#include "LLCL/ForwardDecls.h"
#include "LLCL/Support/Atomics.h"
#include "LLCL/Support/Profiling.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <memory>

namespace M::LLCL {
class LLCLAllocator;

/// Work functions to execute for a 'task'.
using TaskFunction = llvm::unique_function<void()>;

/// Time profiling entries for capturing the running time of tasks.
using WorkProfilerEntry =
    TimeTraceProfilerEntry<Trace::EnableTrace(Trace::kLLCL, 1)>;

/// Time profiling entries for capturing the waiting time of tasks and
/// other internal LLCL measurements.
using InternalProfilerEntry =
    TimeTraceProfilerEntry<Trace::EnableTrace(Trace::kLLCL, 2)>;

/// This is an interface to various implementations of work queues:
/// different execution methods which are often current. These
/// implementations may be very domain or host system specific, but the
/// interface to them is kept intentionally simple to just `addTask` (which
/// adds a block of work to be done as a C++ lambda), and `await` which runs
/// work items until some specific values are ready to go.
///
/// This is aligned to hardware_destructive_interference_size because
/// implementations of this often need that alignment, and without this the
/// destructor unique_ptr destructor is invoked incorrectly.
class alignas(hardware_destructive_interference_size) WorkQueue {
public:
  virtual ~WorkQueue() = default;

  /// Enqueue a work item, usually for later execution, possibly on another
  /// thread. Thread-safe.
  ///
  /// If enabled, the profilerEntry will be used to record two flavors of
  /// profiling entries:
  ///  - Waiting: The time between adding and beginning the task is recorded
  ///    with the name of profilerEntry with an additional '.waiting' suffix.
  ///    This captures the time the task sits waiting in the task list for a
  ///    worker.
  ///  - Running: The time between beginning and ending the task is recorded
  ///    using profilerEntry directly. However, should the task call await,
  ///    the running clock will be stopped early while other work items are
  ///    processed. Once the await returns the running entry will be restarted,
  ///    but with an additional '.post' suffix. In this way work items can
  ///    be timed independently of unrelated work items.
  /// Additional details may be added to the profile entries depending on the
  /// work queue implementation.
  ///
  /// CAUTION: The work item may be run immediately, on the callers stack,
  /// if it cannot be enqueued (eg because the queue is full).
  ///
  /// TODO: Consider returning AsyncValueRef<Chain>, where the task has been
  /// enqueued only if the result is ready.
  virtual void addTask(TaskFunction &&work,
                       WorkProfilerEntry &&profilerEntry =
                           WorkProfilerEntry::create("llcl.doWork")) = 0;

  /// Enqueue a block of work to be run 'locally' on the current thread.
  ///
  /// This method is appropriate for short running work items where the
  /// cost of thread context switching would likely dominate the cost of
  /// simply executing the block of work. For example, the AsyncValue machinery
  /// uses this method to ensure waiters are executed promptly, but off of
  /// the callers stack.
  ///
  /// CAUTION: The work item may be run immediately, on the callers stack,
  /// if it cannot be enqueued (eg because the queue is full).
  virtual void addLocalTask(TaskFunction work) = 0;

  /// Blocks until the given values are ready, either as emplaced values or
  /// as errors.
  ///
  /// For single threaded work queues, the runNewTasks flag is ignored.
  /// Otherwise, the runNewTasks flag indicates whether the callers thread
  /// may be used to process work items while waiting.
  ///
  /// If runNewTasks is true (default), the caller's thread will fill in time
  /// while waiting by processing pending tasks, sleeping only if no other
  /// work is available. Generally this option should only be used when
  /// awaiting at the 'top level', since it is possible for a work item to
  /// take much longer than needed for the values to become ready.
  ///
  /// Otherwise, no new tasks are run by the caller's thread, and it may
  /// sleep. Generally this setting should be preferred when awaiting within a
  /// concurrency primitive which may form part of a larger asynchronous
  /// computation. In particular, this flag is appropriate when the caller
  /// knows the given values should all be ready 'shortly', eg because they
  /// were launched as part of a 'parallelDo' with each shard roughly equal
  /// size, and the callers thread itself contributed to one such shard.
  virtual void await(ArrayRef<AnyAsyncValueRef> values,
                     bool runNewTasks = true) = 0;

  /// Return the pool size maintained by this work queue. Kernels can use
  /// this as a hint indicating the maximum useful number of work items
  /// they should break themselves into.
  virtual size_t getParallelismLevel() const = 0;

  /// Shutdown the thread pool and quiesce in preparation for destruction.
  virtual void shutdown() = 0;

protected:
  WorkQueue() = default;
  virtual void vtableAnchor();
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

/// A task work function along with its profiler entries to record both its
/// waiting time (between addTask and being scheduled), and its execution
/// time (executing the work function).
struct ProfiledTaskFunction {
  TaskFunction work;
  InternalProfilerEntry waiting;
  WorkProfilerEntry running;

  ProfiledTaskFunction(std::nullptr_t) {}
  ProfiledTaskFunction() = default;

  ProfiledTaskFunction(TaskFunction &&work, InternalProfilerEntry &&waiting,
                       WorkProfilerEntry &&running)
      : work(std::move(work)), waiting(std::move(waiting)),
        running(std::move(running)) {}

  operator bool() const { return work.operator bool(); }
};

/// Create a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue> createSingleThreadWorkQueue();

/// Create a thread pool. The busyWait and taskListCapacity parameters are
/// exposed only for unit testing.
std::unique_ptr<WorkQueue> createThreadPoolWorkQueue(
    size_t numThreads,
    std::chrono::nanoseconds busyWait = std::chrono::nanoseconds(1000000),
    size_t taskListCapacity = 128);

} // namespace M::LLCL

#endif // LLCL_RUNTIME_WORKQUEUE_H
