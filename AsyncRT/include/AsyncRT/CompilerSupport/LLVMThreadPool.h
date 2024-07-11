//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_COMPILERSUPPORT_RUNTIME_H
#define LLCL_COMPILERSUPPORT_RUNTIME_H

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Threading/Shared.h"
#include "llvm/Support/ThreadPool.h"

namespace M::LLCL {
class Runtime;

/// This is an implement of the LLVM thread pool interface that wraps an LLCL
/// runtime. This can be used to inject an LLCL runtime as the thread pool
/// implementation inside an MLIR context.
class LLVMThreadPool : public llvm::ThreadPoolInterface {
public:
  LLVMThreadPool(LLCL::Runtime &runtime)
      : runtime(runtime), poolTurnStile(runtime) {}
  ~LLVMThreadPool();

  void asyncEnqueue(std::function<void()> task,
                    llvm::ThreadPoolTaskGroup *Group) override;
  void wait() override;
  void wait(llvm::ThreadPoolTaskGroup &group) override;
  unsigned getMaxConcurrency() const override;

private:
  /// The wrapped LLCL runtime.
  LLCL::Runtime &runtime;

  /// Turnstile for a task group or for the whole queue that can be waited on.
  /// This operates under the assumption that tasks cannot be added to a queue
  /// or group while it is being waited on.
  struct TurnStile {
    TurnStile(LLCL::Runtime &runtime)
        : counter(1), chain(LLCL::AsyncValueRef<Chain>::allocate(runtime)) {}

    bool taskComplete();
    void waitAndReset(LLCL::Runtime &runtime);

    std::atomic<unsigned> counter;
    LLCL::AsyncValueRef<Chain> chain;
  };

  /// Shared table of turnstiles for all active task groups. The elements are
  /// allocated in unique pointers so that tasks functions can carry the
  /// reference safely.
  Shared<DenseMap<llvm::ThreadPoolTaskGroup *, std::unique_ptr<TurnStile>>>
      groupTurnStiles;
  /// Turnstile for the whole task group.
  TurnStile poolTurnStile;
};
} // namespace M::LLCL

#endif // LLCL_COMPILERSUPPORT_RUNTIME_H
