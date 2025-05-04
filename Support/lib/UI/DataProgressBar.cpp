//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/UI/DataProgressBar.h"

#include "Support/UI/ProgressBar.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

using namespace M;
using namespace std::chrono_literals;

static std::string prettyBytes(size_t bytes) {
  auto twoDigits = [](size_t value) -> std::string {
    if (value < 10)
      return std::string("0") + std::to_string(value);
    return std::to_string(value);
  };

  // These are well-known abbreviations that are not subject to our normal
  // variable naming style.
  constexpr size_t KiB = 1024;               // NOLINT
  constexpr size_t MiB = 1024 * 1024;        // NOLINT
  constexpr size_t GiB = 1024 * 1024 * 1024; // NOLINT
  if (bytes < KiB)
    return std::to_string(bytes) + "B";
  if (bytes < MiB)
    return std::to_string(bytes / KiB) + "KiB";
  if (bytes < 10 * MiB)
    return std::to_string(bytes / MiB) + "." +
           twoDigits((bytes % MiB) / (11 * KiB)) + "MiB";
  if (bytes < 100 * MiB)
    return std::to_string(bytes / MiB) + "." +
           std::to_string((bytes % MiB) / (103 * KiB)) + "MiB";
  if (bytes < GiB)
    return std::to_string(bytes / MiB) + "MiB";
  if (bytes < 10 * GiB)
    return std::to_string(bytes / GiB) + "." +
           twoDigits((bytes % GiB) / (11 * MiB)) + "GiB";
  if (bytes < 100 * GiB)
    return std::to_string(bytes / GiB) + "." +
           std::to_string((bytes % GiB) / (103 * MiB)) + "GiB";
  return std::to_string(bytes / GiB) + "GiB";
}

void DataProgressBar::calculateRate(uint64_t newProgress) {
  auto now = std::chrono::system_clock::now();

  uint64_t progress = getProgress() + newProgress;
  // Prune old time points
  while (timePoints.size() >= maxTimePoints)
    timePoints.pop_front();

  // Add new time point
  timePoints.push_back({now, progress});

  if (timePoints.size() < minPointsNeededForRate) {
    return; // Not enough data to calculate rate
  }

  // Calculate rate over the span of the timePoints
  auto &first = timePoints.front();
  auto &last = timePoints.back();
  double elapsed =
      std::chrono::duration<double>(last.time - first.time).count();
  if (elapsed < 0.01) // Avoid division by very small numbers
    return;

  double bytesDiff = static_cast<double>(last.bytes - first.bytes);
  rate = bytesDiff / elapsed;

  // Use EMA for smoothing
  lastRate = ((1.0 - alpha) * rate) + (alpha * lastRate);
  if (lastRate > maxRate)
    maxRate = lastRate;
}

DataProgressBar::DataProgressBar(llvm::raw_ostream &os,
                                 uint64_t exepectedLength, bool showRate,
                                 std::string label,
                                 llvm::raw_ostream::Colors color)
    : SimpleProgressBar(os, exepectedLength, std::move(label), color),
      showRate(showRate) {}

std::string DataProgressBar::generateSuffix() const {
  uint64_t totalDisplayedWork = std::max(progress, expectedWork);
  std::stringstream ss;
  ss << prettyBytes(progress) << "/" << prettyBytes(totalDisplayedWork);

  if (enabled && showRate)
    ss << " @ " << std::setw(8) << prettyBytes(static_cast<size_t>(rate))
       << "/s";

  return ss.str();
}
