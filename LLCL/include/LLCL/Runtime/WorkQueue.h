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

#include "llvm/ADT/FunctionExtras.h"

namespace LLCL {
class LLCLAllocator;

using TaskFunction = llvm::unique_function<void()>;

class WorkQueue {
public:
  virtual ~WorkQueue() {}

protected:
  // Clients should access WorkQueue's methods via LLCL::Runtime.
  friend class Runtime;

  // Enqueue a block of work. Thread-safe.
  virtual void addTask(TaskFunction work) = 0;

  // TODO: Await.

  // Block until the system is quiescent (no pending work and no inflight work).
  //
  // This should not be called by a thread managed by the work queue.
  virtual void quiesce() = 0;

  WorkQueue() = default;

private:
  virtual void vtableAnchor();
  WorkQueue(const WorkQueue &) = delete;
  void operator=(const WorkQueue &) = delete;
};

// Create a thread pool that only uses the host donor thread, involving no
// synchronization.
std::unique_ptr<WorkQueue> createSingleThreadWorkQueue();

} // namespace LLCL

#endif // LLCL_RUNTIME_WORKQUEUE_H
