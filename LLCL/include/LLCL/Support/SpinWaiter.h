//===- SpinWaiter.h -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_SPINWAITER_H
#define LLCL_SUPPORT_SPINWAITER_H

#include <chrono>
#include <thread>
#ifdef _MSC_VER
#include <immintrin.h> // _mm_pause
#endif

namespace LLCL {

/// This class is used in busy-wait loops to provide exponential backoff and
/// to defer to the OS under long waits.  This helps improve situations with
/// high contention, by allowing the thread we're waiting for to have proper
/// access to the memory hierarchy and CPU cores needed to make forward
/// progress.
///
/// This is "free" to initialize in cases where it isn't used, just setting a
/// non-atomic integer to zero.
template <bool shouldYieldToOS = true>
class SpinWaiter {
public:
  SpinWaiter() = default;

  enum {
    // This is the number of times we will spin without doing any system
    // operations.
    rawSpins = 4,
    // This is the number of times we spin with pause, no-op, or equivalent
    // instructions.
    nopSpins = 64,
    // If that doesn't work we yield the thread back to the OS.
    yieldSpins = 128,
    // If that doesn't work we sleep the thread.
  };

  /// This method is called by spinning algorithms that realize they need to try
  /// again.  This returns false when a non-appreciable amount of time has
  /// elapsed (which happens in the first few iterations), and true if this
  /// waited for a longer time.
  bool wait() {
    // Directly spin a few times.
    if (++iterations < rawSpins)
      return false;

    // If a direct spin didn't resolve the issue, do a more serious SMT-aware
    // pause if we know of one.
    if (iterations < nopSpins ||
        // The client can disable the more expensive yielding mechanisms below
        // by setting "shouldYieldToOS" to true.
        !shouldYieldToOS) {
#if defined __i386__ || defined __x86_64__
#ifdef _MSC_VER
      _mm_pause();
#else
      __builtin_ia32_pause();
#endif
#elif __ARM_ARCH_7A__ || __aarch64__
      __asm__ __volatile__("yield" ::: "memory");
#else
      // Hail mary to slow this thread down so other threads can make progress
      // without us fully occupying the load/store unit.
      __asm volatile("nop; nop; nop; nop" : : : "memory");
#endif
      return true;
    }

    // If that didn't work, we yield the thread back to the OS.  This is much
    // heavier weight but can cause the OS to reschedule the problematic thread.
    if (iterations < yieldSpins) {
      std::this_thread::yield();
      return true;
    }

    // Otherwise, we're in pretty serious trouble, actually go to sleep for
    // longer times.
    std::this_thread::sleep_for(std::chrono::microseconds(iterations / 128));
    iterations += 128;
    return true;
  }

  /// Return true if this waiter is going to do heavy weight OS operations to
  /// slow the current thread's progress.
  bool isDoneWithNopSpins() const { return iterations >= nopSpins; }

private:
  size_t iterations = 0;
};
} // namespace LLCL

#endif // LLCL_SUPPORT_SPINWAITER_H
