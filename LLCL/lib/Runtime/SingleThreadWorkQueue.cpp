//===- SingleThreadWorkQueue.cpp - Simple WorkQueue implementation --------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SingleThreadWorkQueue.
//
//===----------------------------------------------------------------------===//

#include "LLCL/WorkQueue.h"
#include <vector>
using namespace LLCL;

namespace {

// This class implements a work queue that uses the client thread to execute all
// work. It spawns no additional threads and has no internal synchronization.
// threads and performs no synchronization. The only thread used is the host
// thread when it gets donated.
class SingleThreadedWorkQueue : public WorkQueue {
public:
  SingleThreadedWorkQueue() {}

  void addTask(TaskFunction work) override;
  void quiesce() override;

private:
  std::vector<TaskFunction> workItems;
};
} // end anonymous namespace

// Enqueue a block of work. This does not use synchronization since this
void SingleThreadedWorkQueue::addTask(TaskFunction work) {
  workItems.push_back(std::move(work));
}

// Block until the system is quiescent (no pending work and no inflight work).
// Because we are single threaded, we *have* to use the host thread to run
// work - there is no one else to do it.
void SingleThreadedWorkQueue::quiesce() {
  std::vector<TaskFunction> localWorkItems;
  while (!workItems.empty()) {
    // Work items can add new items to the vector, and we generally want to run
    // things in order, so make sure we explicitly pop the item off before a new
    // one is added.
    std::swap(localWorkItems, workItems);
    for (auto &item : localWorkItems)
      item();
    localWorkItems.clear();
  }
}

std::unique_ptr<WorkQueue> LLCL::createSingleThreadWorkQueue() {
  return std::make_unique<SingleThreadedWorkQueue>();
}
