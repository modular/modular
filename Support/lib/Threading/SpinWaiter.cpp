//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Threading/SpinWaiter.h"

using namespace M;

bool Detail::SpinWaiterBase::yieldToOS() {
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
