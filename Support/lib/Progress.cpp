//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Progress.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

using namespace M;
using namespace std::chrono_literals;

namespace {
class CLIProgress : public Progress {
public:
  CLIProgress(llvm::raw_ostream &os) : os(os) {}
  virtual ~CLIProgress() { disable(); }
  virtual void addFile() override { totalFiles += 1; }
  virtual void addBytes(size_t bytes) override { totalBytes += bytes; }
  virtual void finishedBytes(size_t bytes) override { doneBytes += bytes; }
  virtual void skippedBytes(size_t bytes) override { doneBytes += bytes; }
  virtual void finishedFile() override { doneFiles += 1; }
  virtual void skippedFile() override { doneFiles += 1; }
  virtual void enable() override;
  virtual void disable() override;

private:
  void emit(const std::chrono::time_point<std::chrono::system_clock> now);
  void update();

  std::atomic<size_t> totalFiles = 0;
  std::atomic<size_t> doneFiles = 0;
  std::atomic<size_t> totalBytes = 0;
  std::atomic<size_t> doneBytes = 0;

  llvm::raw_ostream &os;

  std::thread updater;
  std::mutex mu;
  std::condition_variable cv;
  bool enabled = false;

  std::chrono::time_point<std::chrono::system_clock> lastUpdate;
  size_t lastDoneBytes = 0;
  double lastRate = 0.0;
  double maxRate = 0.0;
};
} // namespace

static std::string prettyBytes(size_t bytes) {
  constexpr size_t KiB = 1024;
  constexpr size_t MiB = 1024 * 1024;
  constexpr size_t GiB = 1024 * 1024 * 1024;
  if (bytes < KiB)
    return std::to_string(bytes) + "B";
  if (bytes < MiB)
    return std::to_string(bytes / KiB) + "KiB";
  if (bytes < 10 * MiB)
    return std::to_string(bytes / MiB) + "." +
           std::to_string((bytes % MiB) / (11 * KiB)) + "MiB";
  if (bytes < 100 * MiB)
    return std::to_string(bytes / MiB) + "." +
           std::to_string((bytes % MiB) / (103 * KiB)) + "MiB";
  if (bytes < GiB)
    return std::to_string(bytes / MiB) + "MiB";
  if (bytes < 10 * GiB)
    return std::to_string(bytes / GiB) + "." +
           std::to_string((bytes % GiB) / (11 * MiB)) + "GiB";
  if (bytes < 100 * GiB)
    return std::to_string(bytes / GiB) + "." +
           std::to_string((bytes % GiB) / (103 * MiB)) + "GiB";
  return std::to_string(bytes / GiB) + "GiB";
}

static std::string prettyBar(size_t width, double percentage) {
  // Basic building blocks. These are not globals, but are still static. We
  // just want them to be lazily initialized when needed.
  static const std::string fullStr = "█";
  static const std::vector<std::string> partialStr = {" ", "▏", "▎", "▍",
                                                      "▌", "▋", "▊", "▉"};
  static size_t partialCount = partialStr.size();
  static double partialThreshold = 1.0 / static_cast<double>(partialCount);

  std::stringstream ss;

  // Construct the set of full blocks first.
  double blocks = percentage * static_cast<double>(width);
  size_t wholeWidth = std::floor(blocks);
  if (wholeWidth > width)
    wholeWidth = width;
  for (size_t i = 0; i < wholeWidth; i++)
    ss << fullStr;
  size_t remainderWidth = width - wholeWidth;
  if (remainderWidth == 0)
    return ss.str(); // Completely done.

  // Construct the partial block. Note that for partial, our calculations the
  // block to be rendered as completely empty.
  remainderWidth--;
  double partial = blocks - static_cast<double>(wholeWidth);
  size_t partialIndex = static_cast<size_t>(partial / partialThreshold);
  ss << partialStr[partialIndex];
  if (remainderWidth == 0)
    return ss.str(); // No blank spots.

  // Construct all completely empty blocks.
  for (size_t i = 0; i < remainderWidth; i++)
    ss << partialStr[0];
  return ss.str();
}

void CLIProgress::enable() {
  std::lock_guard<std::mutex> lk(mu);
  if (enabled)
    return;
  enabled = true;
  updater = std::thread([this] { this->update(); });
}

void CLIProgress::disable() {
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!enabled)
      return;
    enabled = false;
    cv.notify_all();
  }
  updater.join();
  emit(std::chrono::system_clock::now());
  os << "\n"; // Flush output.
}

void CLIProgress::update() {
  std::unique_lock<std::mutex> lk(mu);
  while (enabled) {
    auto now = std::chrono::system_clock::now();
    emit(now);
    cv.wait_until(lk, now + 100ms);
  }
}

void CLIProgress::emit(
    const std::chrono::time_point<std::chrono::system_clock> now) {
  // Calculate the floating rate using a simple expoentially weighted moving
  // average. We want this to be reasonably responsive, so allow it to decay
  // quickly and use only the average of the current & last rates.
  double elapsed = std::chrono::duration<double>(now - lastUpdate).count();
  double rate = static_cast<double>(doneBytes - lastDoneBytes) / elapsed;
  constexpr double alpha = 0.9;
  rate = ((1.0 - alpha)) * rate + (alpha * lastRate);
  lastUpdate = now;
  lastDoneBytes = doneBytes;
  lastRate = rate;
  if (rate > maxRate)
    maxRate = rate;

  // Construct our update. The overall percentage here is an attempt to be
  // conservative since we may not know the full size, nor the full set of
  // files. So take the minimum of files completed and bytes done.
  std::stringstream files;
  files << doneFiles << "/" << totalFiles;

  std::stringstream bytes;
  bytes << prettyBytes(doneBytes) << "/" << prettyBytes(totalBytes);

  std::stringstream details;
  details << "[files " << std::setw(10) << files.str() << "]";
  details << "[bytes " << std::setw(16) << bytes.str() << " @ " << std::setw(8)
          << prettyBytes(static_cast<size_t>(rate)) << "/s]";

  double percentage = std::min(
      static_cast<double>(doneFiles) / static_cast<double>(totalFiles),
      static_cast<double>(doneBytes) / static_cast<double>(totalBytes));
  std::stringstream perstr;
  if (doneFiles == totalFiles) {
    perstr << "💯%"; // Emoji width is two characters.
    percentage = 1.0;
  } else {
    perstr << std::setw(2) << static_cast<int>(100.0 * percentage) << "%";
  }
  std::string perbar = prettyBar(30, percentage);

  // We assume that we own the current line out the given output stream, and we
  // use carriage returns to emit the progress bar with a fixed width. At the
  // end of each emit call, we do a carriage return in preparation for the
  // next. The final disable will be used to emit a newline character.
  os << "\r" << details.str() << "[";
  if (doneFiles == totalFiles || rate >= 0.5 * maxRate) {
    // If we're either finished or we have something within 50% of our top
    // rate, make the progress bar green if colors are available. We keep
    // the green range very wide because we expect variability naturally.
    os.changeColor(llvm::raw_ostream::Colors::GREEN);
  } else if (rate >= 0.1 * maxRate) {
    // If we're at least 10% our top rate, then make it yellow.
    os.changeColor(llvm::raw_ostream::Colors::YELLOW);
  } else {
    // Otherwise, make it red.
    os.changeColor(llvm::raw_ostream::Colors::RED);
  }
  os << perbar;
  os.resetColor();
  os << "] " << perstr.str();
}

ErrorOr<std::unique_ptr<Progress>> M::makeProgress(llvm::raw_ostream &os) {
  // Check if the output is a tty.
  if (os.is_displayed()) {
    std::unique_ptr<Progress> r = std::make_unique<CLIProgress>(os);
    return std::move(r);
  }
  return Error("not a tty"); // Handled appropriately by the caller.
}
