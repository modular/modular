//===- LLCL/Runtime/WorkQueue.h ---------------------------------*- C++ -*-===//
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

/// This is an interface to various implementations of workqueues: different
/// execution methods which are often current.  These implementations may be
/// very domain or host system specific, but the interface to them is kept
/// intentionally simple to just `addTask` (which adds a block of work to be
/// done as a C++ closure), `await` which blocks until some specific values are
/// ready to go, and `quiesce` which blocks until all work is done.
class WorkQueue {
public:
  virtual ~WorkQueue() {}

protected:
  /// Clients should access WorkQueue's methods via LLCL::Runtime.
  friend class Runtime;

  /// Enqueue a block of work. Thread-safe.
  virtual void addTask(TaskFunction work) = 0;

  /// Block until the specified values are ready.  This should not be called by
  /// a thread managed by our work queue.
  virtual void await(llvm::ArrayRef<RCRef<AsyncValue>> values) = 0;

  /// Block until the system is quiescent (no pending work and no inflight
  /// work).
  ///
  /// This should not be called by a thread managed by the work queue.
  virtual void quiesce() = 0;

  WorkQueue() = default;

private:
  virtual void vtableAnchor();
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

/// Create a thread pool that only uses the host donor thread, involving no
/// synchronization.
std::unique_ptr<WorkQueue> createSingleThreadWorkQueue();

} // namespace LLCL

#endif // LLCL_RUNTIME_WORKQUEUE_H
