//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/CompilerSupport/LLVMThreadPool.h"
#include "MLRT/AsyncRT/Runtime/Algorithms.h"
#include "MLRT/AsyncRT/Runtime/CPUDevice.h"

using namespace M;
using namespace MLRT;
using llvm::ThreadPoolTaskGroup;
using AsyncChain = AsyncValueRef<Chain>;

// The `llvm::ThreadPoolInterface` admittedly asks for quite an awkward API.
// Looking at the `StdThreadPool` implementation (the default LLVM threadpool),
// it doesn't lend itself to being very efficient. Among other things, a hash
// map is required, because the `ThreadPoolTaskGroup` provided by the API only
// serves as a pointer ID for the task group.

bool LLVMThreadPool::TurnStile::taskComplete() {
  if (--counter == 0) {
    chain.copy().emplace();
    return true;
  }
  return false;
}

void LLVMThreadPool::TurnStile::waitAndReset(MLRT::CPUDevice &cpuDevice) {
  // Decrement the counter, and if there are still tasks running, wait.
  if (!taskComplete())
    await(chain);
  counter = 1;
  chain = MLRT::AsyncValueRef<Chain>::allocate(cpuDevice);
}

LLVMThreadPool::~LLVMThreadPool() {
  poolTurnStile.waitAndReset(cpuDevice);
  std::move(poolTurnStile.chain).emplace();
}

void LLVMThreadPool::asyncEnqueue(llvm::unique_function<void()> task,
                                  ThreadPoolTaskGroup *group) {
  // Grab the turnstile for this task group, or create one.
  TurnStile *turnstile = groupTurnStiles.modify([this, group](auto &map) {
    std::unique_ptr<TurnStile> &turnstile = map[group];
    if (!turnstile)
      turnstile = std::make_unique<TurnStile>(cpuDevice);
    return turnstile.get();
  });

  // Increment the number of active tasks.
  ++turnstile->counter;

  cpuDevice.getWorkQueue()->addTask(
      [this, turnstile, func = std::move(task)]() mutable {
        // Run the task.
        func();
        // If this is the last task in the taskgroup to complete, erase it from
        // the map to keep the map size bounded.
        turnstile->taskComplete();
        // Indicate that a threadpool task has completed.
        poolTurnStile.taskComplete();
      });
}

void LLVMThreadPool::wait() { poolTurnStile.waitAndReset(cpuDevice); }

void LLVMThreadPool::wait(ThreadPoolTaskGroup &group) {
  TurnStile *turnstile =
      groupTurnStiles.read([group = &group](auto &map) -> TurnStile * {
        if (auto it = map.find(group); it != map.end())
          return it->second.get();
        return nullptr;
      });
  // No tasks scheduled for this group.
  if (!turnstile)
    return;
  if (!turnstile->taskComplete())
    await(turnstile->chain);
  groupTurnStiles.modify([group = &group](auto &map) { map.erase(group); });
}

unsigned LLVMThreadPool::getMaxConcurrency() const {
  return cpuDevice.getWorkQueue()->getParallelismLevel();
}
