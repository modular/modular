//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_SANITIZERS_H
#define SUPPORT_COMPILER_SANITIZERS_H

#include "llvm/Support/raw_ostream.h"

namespace M {
/// The sanitizers enabled for the compilation.
class Sanitizers {
public:
  /// The various sanitizers that can be enabled.
  enum SanitizerKind { kAddress, kThread };

  Sanitizers(unsigned sanitizerMask = 0) : sanitizerMask(sanitizerMask) {}

  /// Check if the given sanitizer is enabled.
  bool has(SanitizerKind sanitizer) const {
    return sanitizerMask & (1 << sanitizer);
  }

  /// Returns if any sanitizer is enabled.
  operator bool() const { return sanitizerMask != 0; }

  /// Print the active sanitizers to `os`.
  void print(llvm::raw_ostream &os) const {
    if (has(Sanitizers::kAddress))
      os << " address";
    if (has(Sanitizers::kThread))
      os << " thread";
  }

private:
  unsigned sanitizerMask;
};
} // namespace M

#endif // SUPPORT_COMPILER_SANITIZERS_H
