//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Utilities for working with integer ranges, both at compile time and runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_RANGEUTILS_H
#define SUPPORT_ML_RANGEUTILS_H

#include "Support/ErrorOr.h"
#include <cstdint>

namespace M {

/// Return the number of results given the start/limit/step values. Returns an
/// error if the given values are invalid.
ErrorOr<int64_t> getRangeNumElements(int64_t start, int64_t limit,
                                     int64_t step);

} // namespace M

#endif // SUPPORT_ML_RANGEUTILS_H
