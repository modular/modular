//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_UI_DATA_PROGRESS_BAR_H
#define SUPPORT_UI_DATA_PROGRESS_BAR_H

#include "Support/ErrorOr.h"
#include "Support/UI/ProgressBar.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <deque>
#include <memory>

namespace M {

// A tty progress bar that displays work on a given amount of data.
// This could be used to display progress on a file download or file extraction.
// The progress bar will display the progress the rate of progress if showRate
// is true.
class DataProgressBar : public SimpleProgressBar {
public:
  /// Create a new progress bar that will display progress on the given stream.
  /// If showRate is true, the progress bar will also display the rate of
  /// progress.
  DataProgressBar(llvm::raw_ostream &os, uint64_t exepectedLength,
                  bool showRate = false, std::string label = "Processing",
                  llvm::raw_ostream::Colors color =
                      llvm::raw_ostream::Colors::BRIGHT_WHITE);

  ~DataProgressBar() override {
    showRate = false;
    setSuffix(generateSuffix());
  }

  /// Add new bytes that will be progressed. This will immediately update the
  /// progress bar. This should be called as soon as each chunk of data is
  /// processed. Rate information is caculated on the time between calls to
  /// progress.
  void addProgress(uint64_t progress) override {
    uint64_t newProgress = progress + this->progress;
    calculateRate(newProgress);
    setSuffix(generateSuffix());
    SimpleProgressBar::setProgress(newProgress);
  }

  /// setWorkDone is used to set the total number of bytes that have been
  /// processed. This will clear any rate information.
  void setProgress(uint64_t progress) override {
    rate = 0.0;
    lastRate = 0.0;
    maxRate = 0.0;
    timePoints.clear();
    setSuffix(generateSuffix());
    SimpleProgressBar::setProgress(progress);
  }

private:
  void calculateRate(uint64_t newProgress);

  std::string generateSuffix() const;

  /// If true, the progress bar will display the rate of progress.
  bool showRate = false;

  /// Last time the progress bar was updated.
  double rate = 0.0;
  double lastRate = 0.0;
  double maxRate = 0.0;

  struct TimePoint {
    std::chrono::time_point<std::chrono::system_clock> time;
    uint64_t bytes;
  };

  /// The time points used to calculate the rate of progress.
  std::deque<TimePoint> timePoints;

  // Adjust as needed for rate caculation smoothing
  const size_t maxTimePoints = 20;
  const size_t minPointsNeededForRate = 3;
  // The smoothing factor for the rate caculation using exponential
  // moving average
  static constexpr double alpha = 0.9;
};

} // namespace M

#endif // SUPPORT_UI_DATA_PROGRESS_BAR_H
