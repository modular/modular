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

/// Work functions to execute for a 'task'.
using TaskFunction = llvm::unique_function<void()>;

/// Time profiling entries for capturing the running time of tasks.
/// May supplied as optional argument to WorkQueue::addTask.
/// Also names: "llcl.addTask.now", "llcl.addLocalTask.task",
///             "llcl.addLocalTask.now"
using WorkProfilerEntry = ProfilerEntry<Trace::EnableTrace(Trace::kLLCL, 1)>;

/// Time profiling entries for capturing the waiting time of tasks and
/// other internal LLCL measurements.
/// Names: "llcl.shutdown", "llcl.shutdown.spinning", "llcl.shutdown.sleeping",
///        "llcl.runOnThread.spinning", "llcl.runOnThread.sleeping",
///        "llcl.await.spinning", "llcl.await.sleeping"
using InternalProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kLLCL, 2)>;

/// Time profiling entries for capturing every execution of a task or
/// local task when no explicit profiling entries were provided.
/// Names: "llcl.doWork", "llcl.waiter"
using AllWorkItemsProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kLLCL, 3)>;

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

  /// Enqueue a work item for later execution, possibly on another thread.
  /// Thread-safe. The work item will NEVER be run immediately. There is no
  /// intrinsic guarantee of fairness, and the caller is responsible for
  /// using AsyncValues or other mechanisms to prevent task starvation.
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
  virtual void addTask(TaskFunction &&work,
                       WorkProfilerEntry &&profilerEntry =
                           AllWorkItemsProfilerEntry::create("llcl.doWork")
                               .copy<WorkProfilerEntry>()) = 0;

  /// Enqueue a work item for later execution, but on the current thread where
  /// possible. The work item will NEVER be run immediately.
  ///
  /// This method is appropriate for short running work items where the
  /// cost of thread context switching would likely dominate the cost of
  /// simply executing the block of work. For example, the AsyncValue machinery
  /// uses this method to ensure waiters are executed promptly, but off of
  /// the callers stack.
  virtual void addLocalTask(TaskFunction work) = 0;

  /// Returns when the given values are ready, either as emplaced values or
  /// as errors.
  ///
  /// In single-threaded environments mayDonate is ignored. In multi-threaded
  /// environments mayDonate indicates if the caller's thread may be 'donated'
  /// towards running pending work items while waiting. However even with
  /// mayDonate the caller's thread may sleep.
  ///
  /// It is valid for await to be called recursively, ie a task being processed
  /// may itself call await. It is valid for the caller to be running on
  /// any thread, including a worker thread managed by this WorkQueue or any
  /// 'foreign' thread.
  virtual void await(ArrayRef<AnyAsyncValueRef> values,
                     bool mayDonate = true) = 0;

  /// Return the pool size maintained by this work queue. Kernels can use
  /// this as a hint indicating the maximum useful number of work items
  /// they should break themselves into.
  virtual size_t getParallelismLevel() const = 0;

  /// Shutdown the thread pool and quiesce in preparation for destruction.
  /// Must be called before the WorkQueue is destroyed. Must be called from
  /// outside of any task. No other 'foreign' thread may be running an await
  /// loop.
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

  explicit operator bool() const { return bool(work); }
};

/// Creates a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue> createSingleThreadWorkQueue();

/// Creates a thread pool able to distribute the execution of work items
/// across numThreads.
///
/// If numThreads is zero it will default to the number of 'physical' cores in
/// the first socket in the system. Generally this will ignore hyperthreading
/// to minimize cache contention, and will avoid cross-NUMA memory traffic.
///
/// If mainWillDonate is true then only numThreads - 1 worker threads will be
/// created, on the assumption the calling thread or some other distinguished
/// 'main' thread will eventually donate themselves to processing work items
/// by calling await with mayDonate true. This is most appropriate for systems
/// driven my a single main thread, such as an REPL or execution tool.
///
/// If mainWillDonate is false then numThreads worker threads will be created,
/// on the assumption await will only be called with mayDonate false. This is
/// most appropriate for multi-threaded servers which wish to share the same
/// threading work queue across multiple requesting threads. The requesting
/// threads are expected to add a task for their request and sleep.
///
/// The work queue must be shutdown before being destroyed.
std::unique_ptr<WorkQueue>
createThreadPoolWorkQueue(size_t numThreads = 0, bool mainWillDonate = true);

} // namespace M::LLCL

#endif // LLCL_RUNTIME_WORKQUEUE_H
