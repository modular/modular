//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_FILL_H
#define SUPPORT_ML_FILL_H

#include "Support/ErrorOr.h"
#include "Support/ML/DType.h"
#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Scalar value generation
//===----------------------------------------------------------------------===//

/// This kernel fills the specified generic buffer with a single "1" or "1.0"
/// real value.  Complex numbers have their imaginary component set to zero.
ErrorOrSuccess getScalarOne(void *destPtr, DType eltType);

/// This kernel fills the specified generic buffer with a single "-1" or "-1.0"
/// real value.  Complex numbers have their imaginary component set to zero.
ErrorOrSuccess getScalarNegativeOne(void *destPtr, DType eltType);

//===----------------------------------------------------------------------===//
// Memory Fills
//===----------------------------------------------------------------------===//

/// This kernel fills the specified generic buffer with a constant value
/// specified by "element".  This returns a non-empty error on failure.
ErrorOrSuccess fillHomogeneous(void *destPtr, size_t numElements, DType eltType,
                               const void *elementPtr);

/// This kernel fills the specified generic buffer with a random values.
/// This returns a non-empty error on failure.
///
/// TODO: This is not implemented in a very general way, the bounds should be
/// passed it or something.
ErrorOrSuccess fillRandom(void *destPtr, size_t numElements, DType eltType);

} // namespace M

#endif // SUPPORT_ML_FILL_H
