//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/UI/ProgressBar.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace M;
using namespace std::chrono_literals;

/// Pretty print a progress bar with a given width and percentage.
static std::string prettyBar(uint16_t width, uint64_t progress,
                             uint64_t totalDisplayedWork) {
  // Basic building blocks. These are not globals, but are still static. We
  // just want them to be lazily initialized when needed.
  static const std::string fillStr = "█";
  static const std::string nextUnitStr = "░";
  static const std::vector<std::string> partialStr = {" ", "▏", "▎", "▍",
                                                      "▌", "▋", "▊", "▉"};
  static size_t partialCount = partialStr.size();
  static double partialThreshold = 1.0 / static_cast<double>(partialCount);

  double percentage = 0;

  if (totalDisplayedWork > 0)
    percentage = std::min(static_cast<double>(progress) /
                              static_cast<double>(totalDisplayedWork),
                          1.0);

  std::stringstream ss;

  // Construct the set of full blocks first.
  double blocks = percentage * static_cast<double>(width);
  uint16_t wholeWidth = std::floor(blocks);
  if (wholeWidth > width)
    wholeWidth = width;
  uint16_t remainderWidth = width - wholeWidth;

  for (size_t i = 0; i < wholeWidth; i++)
    ss << fillStr;

  // If we are on a "chunky" bar (where work units is less than term width),
  // we should fill the next unit of work with
  if (width >= totalDisplayedWork) {
    uint16_t singleUnit = remainderWidth;
    if (totalDisplayedWork > 0)
      singleUnit = std::floor(static_cast<double>(width) /
                              static_cast<double>(totalDisplayedWork));

    for (size_t i = 0; i < singleUnit && remainderWidth > 0; i++) {
      remainderWidth--;
      ss << nextUnitStr;
    }

  } else if (remainderWidth > 0) {
    // If we are on a "smooth" bar, we should fill the next unit of work with
    // the appropriate partial block.
    // Construct the partial block. Note that for partial, our calculations the
    // block to be rendered as completely empty.
    remainderWidth--;
    double partial = blocks - static_cast<double>(wholeWidth);
    size_t partialIndex = static_cast<size_t>(partial / partialThreshold);
    ss << partialStr[partialIndex];
    if (remainderWidth == 0)
      return ss.str(); // No blank spots.
  }

  if (remainderWidth == 0)
    return ss.str(); // Completely done.

  // Construct all completely empty blocks.
  for (size_t i = 0; i < remainderWidth; i++)
    ss << partialStr[0];
  return ss.str();
}

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

std::pair<uint16_t, uint16_t> M::terminalSize() {
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  int columns, rows;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
  cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
  rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
  return {static_cast<uint16_t>(rows), static_cast<uint16_t>(columns)};
}

uint16_t M::terminalWidth() { return M::terminalSize().second; }
#else
#include <sys/ioctl.h> //ioctl() and TIOCGWINSZ
#include <unistd.h>    // for STDOUT_FILENO

std::pair<uint16_t, uint16_t> M::terminalSize() {
  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  return {static_cast<uint16_t>(w.ws_row), static_cast<uint16_t>(w.ws_col)};
}

uint16_t M::terminalWidth() { return M::terminalSize().second; }
#endif

void BaseProgressBar::paint() {
  uint64_t totalDisplayedWork = std::max(progress, expectedWork);
  std::stringstream prefix;
  prefix << std::left << std::setw(prefixWidth) << prefixCache << " [ "
         << std::right;

  auto prefixStr = prefix.str();

  double percentage = 0;
  if (totalDisplayedWork == 0 || progress == 0)
    percentage = 0;
  else
    // Clamp to 1.0.
    percentage = std::min(static_cast<double>(progress) /
                              static_cast<double>(totalDisplayedWork),
                          1.0);

  std::stringstream postfix;
  postfix << " ] " << std::setw(3) << static_cast<int>(100.0 * percentage)
          << "% " << std::left << std::setw(suffixWidth) << suffixCache;

  auto postfixStr = postfix.str();

  // Calculate the width of the bar. We assume that we have a fixed width
  // for the prefix and postfix, and we want to fill the remaining space
  // with the progress bar. We also want to leave a little bit of space.
  ssize_t barWidth = terminalWidth() - prefixStr.size() - postfixStr.size() - 2;

  // Clamp the bar width to a reasonable range (5 to 200 characters wide)
  barWidth = std::min(std::max(barWidth, static_cast<ssize_t>(minBarWidth)),
                      static_cast<ssize_t>(maxBarWidth));
  std::string perbar = prettyBar(barWidth, progress, totalDisplayedWork);

  // We assume that we own the current line out the given output stream, and we
  // use carriage returns to emit the progress bar with a fixed width. At the
  // end of each emit call, we do a carriage return in preparation for the
  // next. The final disable will be used to emit a newline character.
  os << "\r" << prefixStr;
  os.changeColor(barColor);
  os << perbar;
  os.resetColor();
  os << postfixStr;
}

void BaseProgressBar::setBarColor(llvm::raw_ostream::Colors barColor) {
  this->barColor = barColor;
  display();
}

// Called from the destructor to ensure that the progress bar is properly
// terminated.
void BaseProgressBar::enable() {
  if (enabled)
    return;
  enabled = true;
  if (os.is_displayed())
    paint();
}

void BaseProgressBar::disable(bool flush) {
  if (!enabled)
    return;
  enabled = false;
  if (os.is_displayed())
    paint();
  if (flush)
    os << "\n";
}

void BaseProgressBar::display() {
  if (!enabled || !os.is_displayed())
    return;
  paint();
}

void SimpleProgressBar::updateLabel(const std::string &label) {
  setPrefix(label);
  display();
  if (!os.is_displayed())
    os << label << "\n";
}
