//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_SUPPORT_FORKJOIN_H
#define ASYNCRT_SUPPORT_FORKJOIN_H

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"

namespace M::AsyncRT {
/// This class can be used to track completion of various tasks dispatched to
/// the workqueue. For example:
///
/// ```C++
/// ForkJoin state(runtime);
/// for (int i = 0; i < limit; ++i)
///   state.fork([i]{ doWork(i); });
/// state.join();
/// ```
///
/// This is useful when briding async code with sync code, where tasks may be
/// launched in unpredictable ways, but which are all required to complete
/// before moving on.
struct ForkJoin {
public:
  explicit ForkJoin(Runtime &runtime)
      : runtime(runtime), done(AsyncValueRef<Chain>::allocate(runtime)) {}

  /// Add a new work item to track.
  template <typename FnT>
  void fork(FnT &&fn) {
    numWorkItems.fetch_add(1);
    runtime.getWorkQueue()->addTask([fn = std::forward<FnT>(fn), this] {
      fn();
      endWork();
    });
  }
  /// Called by the main thread, this function waits for all work to complete.
  void join() {
    endWork();
    await(done);
  }

private:
  /// Decrement the atomic counter and emplace the chain if all are complete.
  void endWork() {
    if (numWorkItems.fetch_sub(1) == 1)
      done.copy().emplace();
  }

  /// The runtime to use.
  Runtime &runtime;
  /// This chain is set when all in-flight work items are processed.
  AsyncValueRef<Chain> done;
  /// This is the number of in-flight work items, plus 1 for synchronization.
  std::atomic<size_t> numWorkItems = 1;
};
} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_FORKJOIN_H
