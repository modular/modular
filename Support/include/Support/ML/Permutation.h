//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Utilities for working with permutations.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_PERMUTATION_H
#define SUPPORT_ML_PERMUTATION_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"

namespace M {

/// Returns a vector with the contents of the data vector rearranged according
/// to the permutation vector and stride.
///
/// Example:
///   data = [aa, ab, ba, bb]
///   permutation = [1, 0]
///   stride = 2
///
///   returns: [ba, bb, aa, ab]
///
/// TODO move out of ML as a more general-purpose array transformation utility.
template <typename T>
static SmallVector<T> permute(ArrayRef<T> data, ArrayRef<int64_t> permutation,
                              int64_t stride = 1) {
  SmallVector<T> output;

  for (int64_t permIdx : permutation)
    for (int64_t j = 0; j < stride; ++j)
      output.emplace_back(data[permIdx * stride + j]);

  return output;
}

template <typename T>
static SmallVector<T> permute(const SmallVector<T> &data,
                              ArrayRef<int64_t> permutation,
                              int64_t stride = 1) {
  ArrayRef<T> dataRef(data);
  return permute(dataRef, permutation, stride);
}

} // namespace M

#endif // SUPPORT_ML_PERMUTATION_H
