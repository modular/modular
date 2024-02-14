//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ProgressBar.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace M;
using namespace std::chrono_literals;

// A thread safe progress bar that emits dots based on the configured update
// period.
class ThreadSafeDotEmittingProgressBar : public ProgressBar {
public:
  ThreadSafeDotEmittingProgressBar(llvm::raw_ostream &os) : os(os) {}
  ~ThreadSafeDotEmittingProgressBar() override { disable(); }
  void enable() override;
  void disable() override;

private:
  void emit(std::chrono::time_point<std::chrono::system_clock> now);
  void update();

  llvm::raw_ostream &os;

  std::thread updater;
  std::mutex mu;
  std::condition_variable cv;
  bool enabled = false;

  std::chrono::time_point<std::chrono::system_clock> lastUpdate;
};

void ThreadSafeDotEmittingProgressBar::enable() {
  std::lock_guard<std::mutex> lk(mu);
  if (enabled)
    return;

  enabled = true;
  os << "Compiling model";
  updater = std::thread([this] { this->update(); });
}

void ThreadSafeDotEmittingProgressBar::disable() {
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!enabled)
      return;

    enabled = false;
    cv.notify_all();
  }
  updater.join();
  os << "\nDone!\n";
}

void ThreadSafeDotEmittingProgressBar::update() {
  std::unique_lock<std::mutex> lk(mu);
  while (enabled) {
    auto now = std::chrono::system_clock::now();
    emit(now);
    cv.wait_until(lk, now + 100ms);
  }
}

void ThreadSafeDotEmittingProgressBar::emit(
    const std::chrono::time_point<std::chrono::system_clock> now) {
  auto elapsedTime =
      std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate);
  if (elapsedTime >= refreshPeriod) {
    os << ".";
    lastUpdate = now;
  }
}

ErrorOr<std::unique_ptr<ProgressBar>>
M::makeProgressBar(llvm::raw_ostream &os) {
  os.SetUnbuffered();
  std::unique_ptr<ProgressBar> r =
      std::make_unique<ThreadSafeDotEmittingProgressBar>(os);
  return std::move(r);
}
