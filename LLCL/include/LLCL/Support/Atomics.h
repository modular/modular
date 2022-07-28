//===- LLCL/Support/Atomics.h ---------------------------------------------===//
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

#include "LLCL/Support/SpinWaiter.h"

#include <atomic>
#include <type_traits>

namespace LLCL {

/// This method atomically adds 'accumValue' into 'accum'. This exists because
/// std::atomic doesn't provide a native add operation for floating point
/// values. It only works on arithmetic values.
template <typename T>
static void atomicAdd(std::atomic<T> &accumValue, const T &value) {
  static_assert(std::is_arithmetic_v<T>,
                "the input type T must be an arithmetic type.");
  if constexpr (std::is_integral_v<T>) {
    // We use the native add operation for integral values.
    accumValue += value;
  } else {
    // No add operation exists in the C++17 standard so we perform the addition
    // via a compare_exchange_weak loop.
    T prevAccumValue = accumValue;

    LLCL::SpinWaiter waiter;
    while (!accumValue.compare_exchange_weak(prevAccumValue,
                                             prevAccumValue + value)) {
      // Wait a bit and retry.
      waiter.wait();
    }
  }
}

/// This method atomically updates 'maxValue' to 'value' if it is less than it
/// is already.  This exists because std::atomic doesn't provide a native max
/// operation.
template <typename T>
static void atomicMax(std::atomic<T> &maxValue, const T &value) {
  T previousMax = maxValue;

  // Note that compare_exchange_weak updates `previousMax` on failure.
  LLCL::SpinWaiter waiter;
  while (previousMax < value &&
         !maxValue.compare_exchange_weak(previousMax, value)) {
    // Wait a bit and retry.
    waiter.wait();
  }
}

} // namespace LLCL

#endif // LLCL_SUPPORT_ATOMICS_H
