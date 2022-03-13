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

using TaskFunction = llvm::unique_function<void()>;

/// This is an interface to various implementations of work queues: different
/// execution methods which are often current.  These implementations may be
/// very domain or host system specific, but the interface to them is kept
/// intentionally simple to just `addTask` (which adds a block of work to be
/// done as a C++ lambda), and `await` which runs work items until some
/// specific values are ready to go.
class WorkQueue {
public:
  virtual ~WorkQueue() = default;

  /// Enqueue a block of work. Thread-safe.
  virtual void addTask(TaskFunction work) = 0;

  /// Run work items until the specified values are ready, returning to the
  /// caller when they are ready (either as values or as errors).
  virtual void await(llvm::ArrayRef<RCRef<AsyncValue>> values) = 0;

protected:
  WorkQueue() = default;
  virtual void vtableAnchor();
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

/// Create a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue> createSingleThreadWorkQueue();

/// Create a thread pool. Setting 0 as the number of threads makes this default
/// to std::thread::hardware_concurrency().
std::unique_ptr<WorkQueue> createThreadPoolWorkQueue(size_t numThreads = 0);

} // namespace LLCL

#endif // LLCL_RUNTIME_WORKQUEUE_H
