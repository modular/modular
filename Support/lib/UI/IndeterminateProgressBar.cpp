//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/UI/IndeterminateProgressBar.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <string>
#include <utility>

using namespace M;
using namespace std::chrono_literals;

IndeterminateProgressBar::IndeterminateProgressBar(llvm::raw_ostream &os,
                                                   std::string label)
    : os(os), label(std::move(label)) {}

IndeterminateProgressBar::~IndeterminateProgressBar() { disable(); }

void IndeterminateProgressBar::enable() {
  std::lock_guard<std::mutex> lk(mu);
  if (enabled)
    return;

  enabled = true;
  updater = std::thread([this] { this->update(); });
}

void IndeterminateProgressBar::disable() {
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

void IndeterminateProgressBar::update() {
  std::unique_lock<std::mutex> lk(mu);
  while (enabled) {
    auto now = std::chrono::system_clock::now();
    emit(now);
    cv.wait_until(lk, now + 100ms);
  }
}

void IndeterminateProgressBar::emit(
    const std::chrono::time_point<std::chrono::system_clock> now) {
  auto elapsedTime =
      std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate);
  if (elapsedTime >= refreshPeriod) {
    if (currWidth >= maxBarLength)
      currWidth = 0;
    else
      ++currWidth;

    lastUpdate = now;

    std::string barBuf = label;
    llvm::raw_string_ostream ss(barBuf);

    // print dots up to maxBarLength, then goes back to zero dots and repeat.
    ss << std::string(currWidth, '.');
    ss << std::string(maxBarLength - currWidth, ' ');
    os << "\r" << ss.str();
  }
}
