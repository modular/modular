//===- Semaphore.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/Semaphore.h"
#include "llvm/Support/ErrorHandling.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#elif defined(MODULAR_HAVE_SEM_TIMEDWAIT)
#include <cassert>
#include <semaphore.h>
#else
#include <condition_variable>
#include <mutex>
#endif

using namespace LLCL;

/// This class provides the implementation for the Semaphore object. Because we
/// have so many different implementation details, we encapsulate the
/// platform-specific details into this pImpl class.
class Semaphore::Impl {
public:
  /// Manage semaphore lifetime. In cases where this wraps other APIs, this
  /// should be used to (for example) call sem_destroy.
  Impl();
  ~Impl();

  /// Increment the semaphore.
  void post();
  /// Decrement the semaphore. Passing -1 as the parameter means no timeout for
  /// waiting.
  bool wait(int64_t ns);

private:
#if defined(__APPLE__)
  dispatch_semaphore_t sema;
#elif defined(MODULAR_HAVE_SEM_TIMEDWAIT)
  sem_t sema;
#else
  int counter;
  std::mutex mut;
  std::condition_variable cv;
#endif
};

//===----------------------------------------------------------------------===//
// Semaphore::Impl function implementations
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)
//===----------------------------------------------------------------------===//
// Semaphore::Impl for Apple platforms
//===----------------------------------------------------------------------===//

Semaphore::Impl::Impl() : sema(dispatch_semaphore_create(0)) {}
Semaphore::Impl::~Impl() { dispatch_release(sema); }
void Semaphore::Impl::post() { dispatch_semaphore_signal(sema); }
bool Semaphore::Impl::wait(int64_t ns) {
  dispatch_time_t timeout;
  if (ns == -1)
    timeout = DISPATCH_TIME_FOREVER;
  else
    timeout = dispatch_time(DISPATCH_TIME_NOW, /*nsecToAdd*/ ns);

  return 0 != dispatch_semaphore_wait(sema, timeout);
}

#elif defined(MODULAR_HAVE_SEM_TIMEDWAIT)
//===----------------------------------------------------------------------===//
// Semaphore::Impl for POSIX platforms with sem_timedwait
//===----------------------------------------------------------------------===//

Semaphore::Impl::Impl() {
  if (-1 == sem_init(&sema, 0, 0))
    llvm::report_fatal_error("Unable to initialize an unnamed semaphore.");
}

Semaphore::Impl::~Impl() {
  int rc = sem_destroy(&sema);
  assert(rc == 0 && "Unable to destroy the unnamed semaphore.");
}

void Semaphore::Impl::post() { sem_post(&sema); }

bool Semaphore::Impl::wait(int64_t ns) {
  int rc;
  // If we have no timeout, then we just have check for having been interrupted
  // by a signal handler.
  if (ns == -1) {
    while ((rc = sem_wait(&sema)) == -1 && errno == EINTR)
      continue;

    // If sem_wait returned 0 then we're good, we acquired the semaphore.
    // Otherwise, we hit an error and were unable to acquire the semaphore.
    return rc != 0;
  }

  // Get the current time - the timeout on sem_timedwait is an absolute timeout
  // since the epoch.
  struct timespec ts;
  if (-1 == clock_gettime(CLOCK_REALTIME, &ts))
    llvm::report_fatal_error("Unable to call clock_gettime");

  ts.tv_nsec += ns;
  // The semaphore may be interrupted by a signal handler, so check for this
  // case and continue if that is what happens.
  while ((rc = sem_timedwait(&sema, &ts)) == -1 && errno == EINTR)
    continue;

  // Semaphore successfully decremented, return no error.
  if (rc == 0)
    return false;

  // Timeout occurred.
  if (rc == -1 && errno == ETIMEDOUT)
    return true;

  llvm::report_fatal_error(
      "sem_timedwait failed for a reason other than EINTR or ETIMEDOUT.");
}

#else
//===----------------------------------------------------------------------===//
// Backup Semaphore::Impl using std::mutex and std::condition_variable.
//===----------------------------------------------------------------------===//

Semaphore::Impl::Impl() : counter(0) {}

Semaphore::Impl::~Impl() {}

void Semaphore::Impl::post() {
  {
    // Acquire the lock and increment the counter.
    std::lock_guard lock(mut);
    ++counter;
  }

  cv.notify_one();
}

bool Semaphore::Impl::wait(int64_t ns) {
  // Use the condition variable to wait for counter to be greater than 0.
  std::unique_lock lock(mut);

  // If there is no timeout specified, use cv.wait to wait forever. cv.wait's
  // return type is `void` so the actual return value here is `false`.
  if (ns == -1)
    return cv.wait(lock, [&] { return counter > 0; }), false;

  // Otherwise, there's a timeout specified so wait for that number of
  // nanoseconds.
  using namespace std::chrono_literals;
  bool condition = cv.wait_for(lock, ns * 1ns, [&] { return counter > 0; });
  if (!condition)
    return true;

  // We now own the lock and we know the counter is greater than 0, so
  // decrement it.
  --counter;
  return false;
}
#endif

//===----------------------------------------------------------------------===//
// Semaphore function implementations (just forward to Semaphore::Impl)
//===----------------------------------------------------------------------===//

Semaphore::Semaphore() : impl(std::make_unique<Semaphore::Impl>()) {}

// Empty destructor needed here so we can forward declare Semaphore::Impl into
// the header.
Semaphore::~Semaphore() {}

void Semaphore::post() { impl->post(); }

bool Semaphore::wait(int64_t ns) { return impl->wait(ns); }
