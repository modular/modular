//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_AWAITINGMUTEX_H
#define LLCL_SUPPORT_AWAITINGMUTEX_H

#include "AsyncRT/Runtime/Runtime.h"

namespace M::AsyncRT {
/// This class implements an "awaiting" mutex where threads waiting on a mutex
/// donate themselves to the workqueue instead of sleeping. This is useful for
/// parallel programming patterns with LLCL that contain suspension points
/// inside exclusive code guarded by a mutex.
///
/// ```c++
/// mtx.lock();
/// AsyncRT::await(ch, /*mayDonate=*/false);
/// mtx.unlock();
/// ```
///
/// The thread inside the critical section cannot donate itself, because then it
/// can deadlock by recursing into the same code. In addition, the process can
/// deadlock if there are not enough free threads to process the tasks required
/// to complete `ch`.
///
/// The `AwaitingMutex` solves this problem by donating threads waiting on the
/// mutex to the runtime so that they can continue to run tasks.
class AwaitingMutex {
public:
  /// Initialize the awaiting mutex with a runtime. Threads will donate
  /// themselves to the workqueue in this runtime.
  explicit AwaitingMutex(Runtime &runtime) : runtime(runtime) {}

  /// Attempt to acquire the mutex. If it fails, the thread blocks and donates
  /// itself to the runtime.
  void lock() {
    while (true) {
      // Acquire exclusive access to `ch`.
      mutex.lock();
      switch (state) {
      case State::FREE:
        // Acquired with no contention.
        state = State::ACQUIRED_ONCE;
        mutex.unlock();
        return;
      case State::ACQUIRED_ONCE: {
        // First thread to acquire the mutex with contention.
        state = State::ACQUIRED;
        ch = AsyncValueRef<Chain>::allocate(runtime);
        mutex.unlock();
        break;
      }
      case State::ACQUIRED:
        // Contention but the chain is already allocated.
        mutex.unlock();
        break;
      }

      // Wait on the chain.
      AsyncRT::await(ch);
    }
  }

  /// Release the mutex.
  void unlock() {
    llvm::sys::SmartScopedWriter<true> guard(mutex);
    if (state == State::ACQUIRED)
      ch.copy().emplace();
    state = State::FREE;
  }

public:
  /// The runtime to use.
  Runtime &runtime;

  /// Don't allocate a chain if there is no contention. This state keeps track
  /// of when a chain needs to be allocated.
  enum class State { FREE, ACQUIRED_ONCE, ACQUIRED };
  State state = State::FREE;

  /// The chain to spin on if the mutex failed to be acquired.
  AsyncValueRef<Chain> ch;

  /// An actual mutex guarding `ch`.
  llvm::sys::SmartRWMutex<true> mutex;
};
} // namespace M::AsyncRT

#endif // LLCL_SUPPORT_AWAITINGMUTEX_H
