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

/// Returns a vector where applying `permutation` to permute dimensions achieves
/// output. Only supports stride 1 for now.
///
/// Example:
///   data        : [a, b, c, d] <-- returns `data` given the below
///   permutation : [3, 1, 2, 0]
///   output      : [d, b, c, a]
template <typename T>
static SmallVector<T> permuteReverse(ArrayRef<T> output,
                                     ArrayRef<int64_t> permutation) {
  SmallVector<T> input(output.begin(), output.end());
  for (auto [permIdx, outputValue] : llvm::zip(permutation, output)) {
    input[permIdx] = outputValue;
  }
  return input;
}

template <typename T>
static SmallVector<T> permuteReverse(const SmallVector<T> &data,
                                     ArrayRef<int64_t> permutation) {
  ArrayRef<T> dataRef(data);
  return permuteReverse(dataRef, permutation);
}

} // namespace M

#endif // SUPPORT_ML_PERMUTATION_H
