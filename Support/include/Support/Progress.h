//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROGRESS_H
#define SUPPORT_PROGRESS_H

#include "Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace M {

/// Progress is a base class for progress in different subsystems. For these,
/// these hooks are optionally called by backends during insert and find.
///
/// The support progress library provides a helpful CLI implementation for
/// this class, which can be used directly in this case.
class Progress {
public:
  virtual ~Progress() {}

  /// Add a new file to be processed. Files may be tracked independently from
  /// bytes. Each call to addFile should be followed by a corresponding call
  /// to either finishedFile or skippedFile. This should be called as early as
  /// possible, and can be called any number of times.
  virtual void addFile() = 0;

  /// Add new bytes that will be processed (followed by a corresponding call to
  /// either finishedBytes or skippedBytes). Like addFile, this should be called
  /// as early as possible in order to provide an accurate progress meter. It
  /// can be called any number of times.
  virtual void addBytes(size_t bytes) = 0;

  /// Indicate that some bytes were successfully processed.
  virtual void finishedBytes(size_t bytes) = 0;

  /// Indicate that some bytes were skipped in the download. These may be
  /// indicated differently than successfully downloaded bytes.
  virtual void skippedBytes(size_t bytes) = 0;

  /// Indicate that a file (added with addFile) is now completed.
  virtual void finishedFile() = 0;

  /// Indicate that a file is not completed, but will be skipped.
  virtual void skippedFile() = 0;

  /// enable is used to enable the progress meter. This should be called once
  /// after creation.
  virtual void enable() = 0;

  // disable is used to flush and disable the progress meter. No output should
  // be emitted after this call.
  virtual void disable() = 0;
};

/// makeProgress returns an instance of the Progress class which emits a
/// simple progress bar to the console (if appropriate).
ErrorOr<std::unique_ptr<Progress>> makeProgress(llvm::raw_ostream &os);

} // namespace M

#endif // SUPPORT_PROGRESS_H
