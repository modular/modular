//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_UI_INDETERMINATE_PROGRESS_BAR_H
#define SUPPORT_UI_INDETERMINATE_PROGRESS_BAR_H

#include "Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace M {

// A thread safe progress bar that emits dots based on the configured
// update period.
class IndeterminateProgressBar {
public:
  IndeterminateProgressBar(llvm::raw_ostream &os, std::string label);
  ~IndeterminateProgressBar();

  /// enable is used to enable the progress meter. This should be called once
  /// after creation.
  void enable();

  /// disable is used to flush and disable the progress meter. No output should
  /// be emitted after this call.
  void disable();

private:
  void emit(std::chrono::time_point<std::chrono::system_clock> now);
  void update();

  llvm::raw_ostream &os;

  std::thread updater;
  std::mutex mu;
  std::condition_variable cv;
  std::size_t currWidth = 0;
  static constexpr std::chrono::seconds kDefaultRefreshPeriod{1};
  std::chrono::seconds refreshPeriod = kDefaultRefreshPeriod;
  static constexpr size_t kDefaultMaxBarLength{5};
  size_t maxBarLength = kDefaultMaxBarLength;
  bool enabled = false;

  std::chrono::time_point<std::chrono::system_clock> lastUpdate;
  std::string label;
};

} // namespace M

#endif // SUPPORT_UI_INDETERMINATE_PROGRESS_BAR_H
