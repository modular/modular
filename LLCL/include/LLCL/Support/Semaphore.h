//===- Semaphore.h --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_SEMAPHORE_H
#define LLCL_SUPPORT_SEMAPHORE_H

#include <memory>

namespace LLCL {
/// This is an interface to a basic semaphore with post and timed wait
/// functionality. This is essentially a lowest-common-denominator interface
/// that is meant to be able to be backed by a GCD semaphore, or a POSIX
/// semaphore, or in the worst case a counter protected by a mutex.
class Semaphore {
public:
  /// Create and destroy a semaphore.
  Semaphore();
  ~Semaphore();

  /// Increments the semaphore.
  void post();

  /// Attempts to decrement the semaphore, but waits for `ns` nanoseconds.
  /// Returns true if the decrement timed out. Conceptually similar to
  /// POSIX sem_timedwait. You can also pass in a special value: -1 means wait
  /// forever.
  ///
  /// Generally, you should pass -1 here; timeouts usually aren't the answer!
  bool wait(int64_t ns = -1);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace LLCL

#endif // LLCL_SUPPORT_SEMAPHORE_H
