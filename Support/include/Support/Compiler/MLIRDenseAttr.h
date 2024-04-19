//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_MLIRDENSEATTR_H
#define SUPPORT_COMPILER_MLIRDENSEATTR_H

#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

/// Returns true if an array with the given number of elements is sufficiently
/// large that out-of-line storage should be used. This indicates to the caller
/// that the data is big enough to treat specially, e.g. that it shouldn't be
/// stored in the MLIRContext, folded unconditionally, etc.
inline bool shouldUseOutOfLineAttrStorage(size_t numElements) {
  // A sufficiently large element threshold is used to avoid treating large
  // arrays as "free". The storage, constant folding, etc. of large arrays
  // should be treated specially to ensure we don't bloat generated code, memory
  // use, and more.
  static constexpr size_t kLargeDataThreshold = 512;
  return numElements > kLargeDataThreshold;
}

/// Returns an attribute with the given `name` that represents the serialized
/// `data`. The data is always copied into the MLIR context.
DenseResourceElementsAttr
createResourceAttr(MLIRContext *ctx, ArrayRef<char> data, const Twine &name);

/// Returns an attribute with the given `name` that represents the serialized
/// `data`.
DenseResourceElementsAttr createResourceAttr(MLIRContext *ctx, BufferRef data,
                                             const Twine &name);
} // namespace M

#endif // SUPPORT_COMPILER_MLIRDENSEATTR_H
