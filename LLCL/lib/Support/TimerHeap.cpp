//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/TimerHeap.h"
#include "LLCL/Runtime/Algorithms.h"

using namespace M::LLCL;

void TimerHeap::push(const deadline expiration, AsyncValueRef<Chain> chain) {
  std::unique_lock<std::mutex> lk(mu);
  entries.emplace(Entry(expiration, std::move(chain)));
  cv.notify_one();
}

void TimerHeap::stop() {
  {
    std::unique_lock<std::mutex> lk(mu);
    if (!running)
      return;
    running = false;
    cv.notify_one();
  }
  thread.join();
}

void TimerHeap::run() {
  std::unique_lock<std::mutex> lk(mu);
  while (running) {
    if (entries.empty()) {
      cv.wait(lk, [&] { return !running || !entries.empty(); });
      continue; // Recheck running.
    }
    const Entry &next = entries.top();
    auto now = std::chrono::steady_clock::now();
    if (now >= next.expiration) {
      // The entry has expired, trigger the chain and drop it.
      next.expired.copy().emplace();
      entries.pop();
    } else {
      // Wait until this point, or we are signalled.
      auto delta = next.expiration - now;
      cv.wait_for(lk, delta, [&] { return !running; });
    }
  }
}
