//===- Support/MathExtras.h -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace M {

/// Checks if the two input values are numerically the same.
/// When the type is integral, then equality is checked. When the type is
/// floating point, then this checks if the two input values are numerically the
/// close using the abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol) formula.
/// The default absolute and relative tollerances are picked from the numpy
/// default values.
template <typename T>
static bool isClose(T a, T b, double absoluteTolerance = 1.0E-5,
                    double relativeTolerance = 1.0E-8) {
  static_assert(std::is_arithmetic_v<T>, "isClose requires an arithmetic type");
  if constexpr (std::is_integral_v<T>)
    return a == b;
  else
    return std::fabs(a - b) <=
           std::max(static_cast<T>(relativeTolerance) *
                        std::max(std::fabs(a), std::fabs(b)),
                    static_cast<T>(absoluteTolerance));
}
} // namespace M
