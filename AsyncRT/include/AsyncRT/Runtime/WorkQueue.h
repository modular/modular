//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::AsyncRT::WorkQueue interface, which allows clients
// of AsyncRT to implement work queues that map onto their systems in a nice
// way.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_WORKQUEUE_H
#define ASYNCRT_RUNTIME_WORKQUEUE_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/CompactRuntimePtr.h"
#include "AsyncRT/Support/Resource.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/Threading/Atomics.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <memory>

/// This is the default taskId for all tasks not originating from
/// async_parallelize. We set it to -1 to indicate that the task
/// should enqueue to Global queue.
constexpr int kDefaultTaskId = -1;

namespace M::AsyncRT {

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

namespace Detail {
// Extract the result type of a function passed to addTask(Runtime, fn).
template <typename T>
struct UnwrapErrorOr {
  using type = T;
};
template <typename T>
struct UnwrapErrorOr<ErrorOr<T>> {
  using type = T;
};

template <typename F>
using ResultType = typename UnwrapErrorOr<std::invoke_result_t<F>>::type;
} // namespace Detail

//===----------------------------------------------------------------------===//
// Common types
//===----------------------------------------------------------------------===//

/// Functions to execute for a 'task'.
using TaskFunction = llvm::unique_function<void()>;

/// Profiling entries for capturing the waiting time of tasks and other
/// internal AsyncRT measurements.
using InternalProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kAsyncRT, 2), Trace::kAsyncRT>;

/// Profiling entries for capturing every execution of a task or local task.
using AllWorkItemsProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kAsyncRT, 3), Trace::kAsyncRT>;

using namespace std::chrono_literals;

/// A work item to be added, or held, by a work queue. Contains the 'task'
/// function. Depending on build type may contain extra bookkeeping data.
struct WorkItem {
  TaskFunction task;
#if MODULAR_PARANOID
  /// If non-null, a representation of the implied 'use' this work item has
  /// of the resources it depends on. It is valid for the same work queue to be
  /// shared between, say, execution of many different models. This use can
  /// help detect when a work item has not been correctly threaded into the
  /// AsyncValue dependencies for such execution, which can cause the work
  /// item to 'overhang' destruction of the model's resources.
  ResourceUse use;
#endif

  WorkItem() = default;

  WorkItem(const WorkItem &) = delete;
  WorkItem &operator=(const WorkItem &) = delete;

  WorkItem(WorkItem &&) = default;
  WorkItem &operator=(WorkItem &&) = default;

  /// NOTE: Intentionally not marking explicit to promote from nullptr.
  WorkItem(std::nullptr_t null) {}

  /// NOTE: Intentionally not marking explicit to promote from function.
  template <typename FnTy, typename ResultTy = Detail::ResultType<FnTy>,
            std::enable_if_t<(std::is_void<ResultTy>()), int> = 0>
  WorkItem(FnTy f) : task(std::forward<FnTy>(f)) {}

#if MODULAR_PARANOID
  WorkItem(TaskFunction &&task, ResourceUse use)
      : task(std::move(task)), use(std::move(use)) {}
#endif

  explicit operator bool() { return (bool)task; }
};

//===----------------------------------------------------------------------===//
// WorkQueue
//===----------------------------------------------------------------------===//

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
  /// taskId, when >=0 indicates the thread local ring buffer to which
  /// this task needs to be enqueued. taskId = kDefaultTaskId indicates that
  /// the task will be pushed to the common taskList shared by all workers.
  /// In the current implementation, taskId gets a non-negative value only
  /// from `async_parallelize` from mojo. Every where else it should be
  /// kDefaultTaskId.
  virtual void addTask(WorkItem &&work, int taskId = kDefaultTaskId) = 0;

  /// Enqueue a work item for later execution, but on the current thread where
  /// possible. The work item will NEVER be run immediately.
  ///
  /// This method is appropriate for short running work items where the
  /// cost of thread context switching would likely dominate the cost of
  /// simply executing the block of work. For example, the AsyncValue machinery
  /// uses this method to ensure waiters are executed promptly, but off of
  /// the callers stack.
  virtual void addLocalTask(WorkItem &&workItem) = 0;

  /// Returns when the given values are ready, either as emplaced values or
  /// as errors. Depending on the WorkQueue implementation and the caller's
  /// thread, the await may sleep, may 'donate' itself to running work items,
  /// or both.
  ///
  /// It is valid for await to be called recursively, ie a task may itself
  /// call await, effectively 'blocking' it. However, that just means the
  /// task will start processing other tasks while 'waiting' for its values
  /// to become ready. Try to avoid this in favor of using synchronization
  /// via AsyncValues only.
  ///
  /// It is valid for the caller to be running on any thread, including a
  /// worker thread managed by this WorkQueue, a worker thread managed by
  /// some other WorkQueue, the 'main' thread which created the WorkQueue,
  /// or some 'foreign' thread.
  ///
  /// CAUTION: Though await will only return when all values are ready, that
  /// does NOT imply all the waiters for those values have been run (and any
  /// work triggered by those waiter have been run, and so on to quiescence).
  /// Furthermore, since await itself relies on waiters, two awaits on the
  /// same value from different threads can return in any order. Thus, care
  /// must be taken when using await to decide when a computation is 'done'
  /// and the resources it depends on can be destroyed. Generally, only a
  /// shutdown() can guarantee that all in-flight computation has completed.
  virtual void await(ArrayRef<AnyAsyncValueRef> values) = 0;

  /// Returns true if the calling thread is known to be 'foreign' to this
  /// work queue. What this means depends on the work queue implementation.
  virtual bool callerIsForeign() const = 0;

  /// Return the pool size maintained by this work queue. Kernels can use
  /// this as a hint indicating the maximum useful number of work items
  /// they should break themselves into.
  virtual size_t getParallelismLevel() const = 0;

  /// Shutdown the thread pool and quiesce in preparation for destruction.
  /// Must be called before the WorkQueue is destroyed. Must be called from
  /// outside of any task. Depending on WorkQueue implementation, may need
  /// to be called from the same thread which created the WorkQueue.
  virtual void shutdown() = 0;

#if MODULAR_PARANOID
  /// Pushes use onto this thread's internal 'use stack'. When a task or local
  /// task is added with a null use in its WorkItem (the default),
  /// the current stack top of the calling thread is taken to be its implicit
  /// use. While a work item is executing its use is similarly pushed onto the
  /// stack of its executing thread. In this way pushing a single use from the
  /// 'main' thread will cause it to be implicitly captured by the whole 'tree'
  /// of tasks it launches, over all threads.
  ///
  /// Cannot be called from a foreign thread.
  virtual void pushDefaultUse(ResourceUse use) = 0;

  /// Pop a use from this thread's internal 'use stack'. Will assert fail if
  /// use stack is empty.
  ///
  /// Cannot be called from a foreign thread.
  virtual void popDefaultUse() = 0;

  /// Indicates the caller's task is done for the purposes of detecting task
  /// overhangs at runtime. May be called any number of times. Should be called
  /// before an emplace() or setToError() which may cause the resource being
  /// tracked to be marked as 'free'.
  virtual void taskIsDone() = 0;
#endif

protected:
  WorkQueue() = default;
  virtual void vtableAnchor();
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

/// Creates a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue>
createSingleThreadWorkQueue(CompactRuntimePtr runtimePtr);

/// Creates a thread pool able to distribute the execution of work items
/// across numThreads.
///
/// If numThreads is zero it will default to a sensible number based on the
/// current physical system. The maxThreads parameter is used to bound
/// numThreads in this case. If maxThreads is zero, it is ignored.
///
/// If mainWillDonate is false then numThreads worker threads will
/// be created. Arbitrary threads may then addTasks and call await, but will not
/// themselves contribute to processing work items. This is most appropriate for
/// multi-threaded servers which wish to share the same work queue across
/// multiple request threads.
///
/// If mainWillDonate is true (currently the default) then only numThreads - 1
/// worker threads will be created, on the assumption the calling thread will
/// eventually call await and 'donate' itself to processing work items alongside
/// the worker threads. This is most appropriate for systems driven my a single,
/// distinguished main thread, such as a REPL or execution tool.
///
/// The work queue must be shutdown before being destroyed. Until shutdown has
/// returned any number of work items may be executing, so no resources they
/// depend on should be destroyed. If mainWillDonate is true, the calling
/// thread must be the one to call shutdown, at which point it may (again)
/// contribute to processing outstanding work items. Otherwise shutdown
/// may be called from any foreign thread.
///
/// If in a MODULAR_PARANOID build, the paranoid flag can be used to inject
/// random delays into work items to attempt to tickle race conditions.
std::unique_ptr<WorkQueue>
createThreadPoolWorkQueue(CompactRuntimePtr runtimePtr, size_t numThreads,
                          size_t maxThreads, bool mainWillDonate,
                          bool withAffinity,
                          std::chrono::microseconds threadBusyWaitTime,
                          std::string_view poolName, bool paranoid);

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_WORKQUEUE_H
