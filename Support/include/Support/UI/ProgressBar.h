//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_UI_PROGRESS_BAR_H
#define SUPPORT_UI_PROGRESS_BAR_H

#include <utility>

#include "llvm/Support/raw_ostream.h"

namespace M {

// Query the size of the terminal. Returns rows and columns.
std::pair<uint16_t, uint16_t> terminalSize();

// Query the width of the terminal.
uint16_t terminalWidth();

// A tty base progress bar that displays a bar with optional percentage
// based on the total number of work units.
class BaseProgressBar {
public:
  BaseProgressBar(llvm::raw_ostream &os, uint64_t expectedWork,
                  llvm::raw_ostream::Colors barColor =
                      llvm::raw_ostream::Colors::BRIGHT_WHITE)
      : os(os), expectedWork(expectedWork), barColor(barColor) {}

  /// Destroy the progress bar, flushing any remaining output.
  virtual ~BaseProgressBar() { disable(true); }

  /// getProgress returns the progress so far.
  uint64_t getProgress() const { return progress; }

  /// Adds progress. This will immediately paint the progress bar.
  virtual void addProgress(uint64_t progress) {
    this->progress += progress;
    display();
  }

  /// setProgress sets the progress bar to the given work units. This will
  /// immediately update the progress bar.
  virtual void setProgress(uint64_t progress) {
    this->progress = progress;
    display();
  }

  /// getExpectedWork returns the total amount work that will be processed.
  uint64_t getExpectedWork() const { return expectedWork; }

  /// Reset the total expected work. Updates immediately.
  void setExpectedWork(uint64_t expectedWork) {
    this->expectedWork = expectedWork;
    display();
  }

  /// setColor is used to set the color of the progress meter.
  void setBarColor(llvm::raw_ostream::Colors barColor);

  /// enabled is used to enable the progress meter. This will paint the progress
  /// bar to the output stream.
  void enable();

  /// disable is used to flush and disable the progress meter.
  void disable(bool flush = false);

  /// Refresh the progress bar to the output stream.
  void display();

protected:
  /// Paint the progress bar to the output stream.
  void paint();

  /// The stream that the progress bar will be painted to.
  llvm::raw_ostream &os;

  /// If false, the progress bar will not paint updates.
  bool enabled = true;

  /// The min and max width of the progress bar. Tries to use the terminal width
  /// to determine the width of the progress bar. If the terminal width cannot
  /// be determined, the min width is used.
  static const size_t kDefaultMinBarWidth{5};
  size_t minBarWidth = kDefaultMinBarWidth;
  static const size_t kDefaultMaxBarWidth{200};
  size_t maxBarWidth = kDefaultMaxBarWidth;

  /// The width of the prefix and suffix of the progress bar.
  static const size_t kDefaultPrefixWidth{20};
  size_t prefixWidth = kDefaultPrefixWidth;
  static const size_t kDefaultSuffixWidth{30};
  size_t suffixWidth = kDefaultSuffixWidth;

  /// Generate the prefix and suffix for the progress bar. Called when display()
  /// is invoked.
  void setPrefix(std::string prefix) { prefixCache = std::move(prefix); }
  void setSuffix(std::string suffix) { suffixCache = std::move(suffix); }

  /// The total work done so far.
  uint64_t progress = 0;

  /// The total number of work units we expect to do.
  uint64_t expectedWork = 0;

  /// The color of the progress bar.
  llvm::raw_ostream::Colors barColor = llvm::raw_ostream::Colors::BRIGHT_WHITE;

private:
  /// Cache the last painted progress bar for destruction.
  std::string prefixCache = "";
  std::string suffixCache = "";
};

/// A simple progress bar that displays a bar with a label.
class SimpleProgressBar : public BaseProgressBar {
public:
  SimpleProgressBar(llvm::raw_ostream &os, uint64_t expectedWork,
                    std::string label = "Processing",
                    llvm::raw_ostream::Colors barColor =
                        llvm::raw_ostream::Colors::BRIGHT_WHITE)
      : BaseProgressBar(os, expectedWork, barColor) {
    setPrefix(std::move(label));
  }

  ~SimpleProgressBar() override = default;

  /// updateLabel is used to set the label for the progress meter.
  /// For non-tty displays this will print the label to the output stream.
  /// For tty displays this will update the label in place.
  void updateLabel(const std::string &label);
};

} // namespace M

#endif // SUPPORT_UI_PROGRESS_BAR_H
