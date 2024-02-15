//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROGRESS_BAR_H
#define SUPPORT_PROGRESS_BAR_H

#include "Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <memory>

namespace M {

/// ProgressBar is a base class for progress in different subsystems.
class ProgressBar {
public:
  virtual ~ProgressBar() = default;

  /// enable is used to enable the progress meter. This should be called once
  /// after creation.
  virtual void enable() = 0;

  // disable is used to flush and disable the progress meter. No output should
  // be emitted after this call.
  virtual void disable() = 0;
};

/// makeProgressBar returns an instance of the ProgressBar class which
/// emits a simple progress bar to the console (if appropriate).
ErrorOr<std::unique_ptr<ProgressBar>> makeProgressBar(llvm::raw_ostream &os);

} // namespace M

#endif // SUPPORT_PROGRESS_BAR_H
