//===- LLCL/Support/Atomics.h - std::atomic helpers -------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides helper functions for working with std::atomic.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_ATOMICS_H
#define LLCL_SUPPORT_ATOMICS_H

namespace LLCL {

/// This method atomically updates 'maxValue' to 'value' if it is less than it
/// is already.  This exists because std::atomic doesn't provide a native max
/// operation.
template <typename T>
static void atomicMax(std::atomic<T> &maxValue, const T &value) {
  T previousMax = maxValue;

  // Note that compare_exchange_weak updates `previousMax` on failure.
  while (previousMax < value &&
         !maxValue.compare_exchange_weak(previousMax, value)) {
  }
}

} // namespace LLCL

#endif // LLCL_SUPPORT_ATOMICS_H
