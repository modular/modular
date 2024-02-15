//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ProgressBar.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

using namespace M;
using namespace std::chrono_literals;

namespace {

constexpr std::chrono::seconds kDefaultRefreshPeriod{1};
constexpr size_t kDefaultMaxBarLength{5};

// A thread safe progress bar that emits dots based on the configured
// update period.
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
  std::size_t currWidth = 0;
  std::chrono::seconds refreshPeriod = kDefaultRefreshPeriod;
  size_t maxBarLength = kDefaultMaxBarLength;
  bool enabled = false;

  std::chrono::time_point<std::chrono::system_clock> lastUpdate;
};

} // namespace

void ThreadSafeDotEmittingProgressBar::enable() {
  std::lock_guard<std::mutex> lk(mu);
  if (enabled)
    return;

  enabled = true;
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
    if (currWidth >= maxBarLength)
      currWidth = 0;
    else
      ++currWidth;

    lastUpdate = now;

    std::string prefix{"Compiling model"};
    llvm::raw_string_ostream ss(prefix);

    // print dots up to maxBarLength, then goes back to zero dots and repeat.
    ss << std::string(currWidth, '.');
    ss << std::string(maxBarLength - currWidth, ' ');
    os << "\r" << ss.str();
  }
}

ErrorOr<std::unique_ptr<ProgressBar>>
M::makeProgressBar(llvm::raw_ostream &os) {
  std::unique_ptr<ProgressBar> r =
      std::make_unique<ThreadSafeDotEmittingProgressBar>(os);
  return std::move(r);
}
