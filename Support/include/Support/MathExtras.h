//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MATHEXTRAS_H
#define SUPPORT_MATHEXTRAS_H

#include <algorithm>
#include <cmath>
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
    if constexpr (IsNanSensitive) {
      if (std::isnan(a) || std::isnan(b))
        return false;
    } else if (std::isnan(a) && std::isnan(b)) {
      return true;
    }
    return std::fabs(a - b) <=
           std::max(static_cast<T>(relativeTolerance) *
                        std::max(std::fabs(a), std::fabs(b)),
                    static_cast<T>(absoluteTolerance));
  }
}
} // namespace M

#endif // SUPPORT_MATHEXTRAS_H
