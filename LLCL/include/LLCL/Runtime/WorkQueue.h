//===- LLCL/Runtime/WorkQueue.h -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LLCL::WorkQueue interface, which allows clients of
// LLCL to implement work queues that map onto their systems in a nice way.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_WORKQUEUE_H
#define LLCL_RUNTIME_WORKQUEUE_H

#include "LLCL/ForwardDecls.h"
#include "Support/LLVM.h"
#include "llvm/ADT/FunctionExtras.h"

namespace LLCL {
class LLCLAllocator;
class TaskFunctionBase; // Defined below.

/// This is an interface to various implementations of work queues: different
/// execution methods which are often current. These implementations may be very
/// domain or host system specific, but the interface to them is kept
/// intentionally simple to just `addTask` (which adds a block of work to be
/// done as a C++ lambda), and `await` which runs work items until some specific
/// values are ready to go.
class WorkQueue {
public:
  virtual ~WorkQueue() = default;

  /// Enqueue a block of work. Thread-safe.
  template <typename CallableT>
  void addTask(CallableT &&work);

  /// Run work items until the specified values are ready, returning to the
  /// caller when they are ready (either as values or as errors).
  virtual void await(llvm::ArrayRef<AnyAsyncValueRef> values) = 0;

  /// Return the pool size maintained by this work queue. Kernels can use
  /// this as a hint indicating the maximum useful number of work items
  /// they should break themselves into.
  virtual int getParallelismLevel() const = 0;

protected:
  WorkQueue() = default;
  virtual void vtableAnchor();
  virtual void addTaskInternal(TaskFunctionBase *work) = 0;
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

/// This is a non-templated base class that can be passed into implementations
/// of workqueue.  It is subclassed by TaskFunction to hold a callable as a more
/// efficient way to type erase it than std::function or comparable (which has a
/// fixed size buffer but then overflows to a separate allocation).
class TaskFunctionBase {
public:
  explicit TaskFunctionBase() : next(nullptr) {}
  virtual ~TaskFunctionBase() {}
  virtual void call() = 0;

  std::unique_ptr<TaskFunctionBase> next;

private:
  TaskFunctionBase(const TaskFunctionBase &) = delete;
  void operator=(const TaskFunctionBase &) = delete;
};

/// Templated ItemBase implementation class that holds a anonymous lambda
/// function.
template <typename CallableT>
class TaskFunction final : public TaskFunctionBase {
public:
  explicit TaskFunction(CallableT newCallable)
      : callable(std::move(newCallable)) {}

  void call() override { callable(); }

  CallableT callable;
};

/// Enqueue a block of work. Thread-safe.
template <typename CallableT>
inline void WorkQueue::addTask(CallableT &&work) {
  addTaskInternal(new TaskFunction<CallableT>(std::move(work)));
}

/// Create a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue> createSingleThreadWorkQueue();

/// Create a thread pool. Setting 0 as the number of threads makes this default
/// to std::thread::hardware_concurrency().
std::unique_ptr<WorkQueue> createThreadPoolWorkQueue(size_t numThreads = 0);

} // namespace LLCL

#endif // LLCL_RUNTIME_WORKQUEUE_H
