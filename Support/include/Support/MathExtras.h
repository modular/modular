//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MATHEXTRAS_H
#define SUPPORT_MATHEXTRAS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

namespace M {

/// Checks if the two input values are numerically the same.
/// When the type is integral, then equality is checked. When the type is
/// floating point, then this checks if the two input values are numerically the
/// close using the abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol) formula.
/// The default absolute and relative tolerances are picked from the numpy
/// default values. If IsNanSensitive is false, then two NaNs are considered
/// equal.
template <typename T, bool IsNanSensitive = true>
static bool isClose(T a, T b, double absoluteTolerance = 1.0E-5,
                    double relativeTolerance = 1.0E-8) {
  static_assert(std::is_arithmetic_v<T>, "isClose requires an arithmetic type");
  if constexpr (std::is_integral_v<T>) {
    return a == b;
  } else {
    if (LLVM_UNLIKELY(!IsNanSensitive && std::isnan(a) && std::isnan(b)))
      return true;
    if (LLVM_UNLIKELY(std::isnan(a) || std::isnan(b)))
      return false;
    return std::fabs(a - b) <=
           std::max(static_cast<T>(relativeTolerance) *
                        std::max(std::fabs(a), std::fabs(b)),
                    static_cast<T>(absoluteTolerance));
  }
}

/// Computes the mean of the input array.
template <typename Container>
inline auto mean(const Container &values)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  using value_type =
      std::remove_reference_t<decltype(*llvm::adl_begin(values))>;
  value_type init(0);
  if (values.empty())
    return init;
  return std::accumulate(llvm::adl_begin(values), llvm::adl_end(values), init) /
         std::size(values);
}

/// Computes the median of the input array assuming it is sorted.
template <typename Container>
inline auto median(const Container &values)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  assert(llvm::is_sorted(values) && "values are assumed to be sorted");

  // Get the size of the container.
  auto size = std::size(values);

  // If the array is less than or equal to 2 elements, then the median is the
  // mean.
  if (size < 3)
    return mean(values);

  auto mid = size / 2;
  auto midValue = values[mid];
  // If the size is odd, the center is the median.
  if (size % 2 == 1)
    return midValue;
  // Otherwise, the average of the two elements in the center are the median.
  auto midValue2 = values[mid - 1];
  return (midValue + midValue2) / 2;
}

/// Computes the percentile of the input array assuming it is sorted.
template <typename Container>
inline auto percentile(double p, const Container &values)
    -> std::remove_reference_t<decltype(*llvm::adl_begin(values))> {
  assert(llvm::is_sorted(values) && "values are assumed to be sorted");
  assert(p >= 0.0 && p <= 1.0 && "percentile must be in [0, 1]");
  return values[static_cast<size_t>(values.size() * p)];
}

} // namespace M

#endif // SUPPORT_MATHEXTRAS_H
