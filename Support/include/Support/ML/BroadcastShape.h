//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains broadcast helpers following the broadcasting logic of
// numpy and TensorFlow. This file should not depend on either of these
// libraries, but since their broadcasting logic is widely used, it is made
// available for all modules.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// File originates from:
//   Repo:   git@github.com:tensorflow/tensorflow.git
//   Commit: 0b7349102db619105fb282c2340a64c44e4adbe6
//   Path:   tensorflow/core/util/bcast.h
//
//===----------------------------------------------------------------------===//

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SUPPORT_ML_BROADCASTSHAPE_H
#define SUPPORT_ML_BROADCASTSHAPE_H

#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/TensorShape.h"

namespace M {

/// Safely multiplies dimensions taking into account dynamic shapes.
int64_t multiplySymDims(int64_t dim1, int64_t dim2);

/// Computes the shape of the output tensor given the shape of the two input
/// tensor shapes. This follows the NumPy broadcasting rules]
/// (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
/// If the shapes are not compatible, then an error is returned.
ErrorOr<TensorShape> broadcastedShape(const TensorShape &a,
                                      const TensorShape &b);

/// Returns the mapping from the output batch indices to the corresponding
/// input's batch indices, given the input's "reshape" and "bcast" shapes as
/// returned by the BCastList helper class. The i'th element denotes the
/// (flattened) batch index of the input that must be used to compute the i'th
/// batch output.
void computeBatchIndices(int64_t outputBatchSize, ArrayRef<int64_t> reshape,
                         ArrayRef<int64_t> bcast,
                         llvm::SmallVectorImpl<int64_t> &out_indices);

template <int N = 2>
class BCastList {
public:
  /// A vector of int64_t representing the shape of tensor. The 0-th element is
  /// the outer-most dimension and the last element is the inner-most dimension.
  /// Note that we do not use TensorShape since it's more convenient to
  /// manipulate Vec directly for this module.
  using Vec = SmallVector<int64_t, 4>;

  /// Constructs all helper shapes, following the aforementioned rules.
  ///
  /// If "fewerDimsOptimization" is set to true, the implementation tries to
  /// reduce intermediate dimensions needed to be more efficient. This is
  /// transparent to the caller. If false, all intermediate shapes have the same
  /// number of dimensions as the larger of the two inputs.
  explicit BCastList(ArrayRef<ArrayRef<int64_t>>,
                     bool fewerDimsOptimization = false);

  ~BCastList() = default;

  /// Returns true if and only if two operands are compatible according to the
  /// broadcasting rule.
  bool isValid() const { return valid; }
  bool isBroadcastingRequired() const { return broadcastingRequired; }

  /// If and only if isValid(), the following fields can be used in implementing
  /// a broadcasted binary tensor operation according to the broadcasting rule.
  ArrayRef<int64_t> getReshape(int i) const { return reshape[i]; }
  ArrayRef<int64_t> getBCast(int i) const { return bcast[i]; }
  ArrayRef<int64_t> getResultShape() const { return result; }
  ArrayRef<int64_t> getOutputShape() const { return output; }
  int64_t getOutputBatchSize() const { return outputBatchSize; }

  /// Returns the mapping from the flattened output batch indices to x's
  /// flattened batch indices. The result is a vector of length
  /// getOutputBatchSize(). To compute the i'th batch output, a binary
  /// matmul-like operation should use the `getXBatchIndices()[i]`th batch index
  /// of `x`. Note: Returns an empty vector if broadcasting is not required.
  /// Callers should only use this when isBroadcastingRequired() returns true.
  ArrayRef<int64_t> getBatchIindices(int i) const { return batch_indices_[i]; }

protected:
  bool valid = true;
  bool broadcastingRequired = true;
  SmallVector<Vec, N> reshape;
  SmallVector<Vec, N> bcast;
  Vec result;
  Vec output;

  int64_t outputBatchSize = 1;
  SmallVector<SmallVector<int64_t>, N> batch_indices_;

  static void reverse(llvm::SmallVectorImpl<int64_t> &shape) {
    std::reverse(shape.begin(), shape.end());
  }

  BCastList(const BCastList &) = delete;
  void operator=(const BCastList &) = delete;
};

/// BroadcastShape is a helper for broadcasting binary tensor operations,
/// following the rules of numpy (See
/// http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
/// The rule has the following properties:
///
///   1. Suffix matching: the rule starts with the right-most dimension, and
///   works towards the left-most dimension. Since numpy is row-major, the
///   right-most dimension (the last element in the shape of a tensor) is the
///   inner-most, a.k.a. the fastest changing, dimension.
///
///   2. Two dimensions are compatible for broadcasting if both are the same or
///   either is 1.
///
/// BroadcastShape takes the shape of two tensors and computes a few vectors of
/// int32 that are useful for the caller to reshape the tensors, apply the right
/// broadcasts to them, and compute the broadcasted operation. In a nutshell,
/// the caller is expected to compute the broadcasted operation as following:
///
///   BroadcastShape b(x.shape(), y.shape());
///   output = x.reshape(b.getXReshape()).broadcast(b.getXBCast())
///            _op_
///            y.reshape(b.getYReshape()).broadcast(b.getYBCast())
class BroadcastShape : public BCastList<2> {
public:
  /// Constructs all helper shapes, following the aforementioned rules.
  ///
  /// If "fewerDimsOptimization" is set to true, the implementation tries to
  /// reduce intermediate dimensions needed to be more efficient. This is
  /// transparent to the caller.
  ///
  /// If false, all intermediate shapes have the same number of dimensions as
  /// the larger of the two inputs.
  using Vec = SmallVector<int64_t, 4>;

  BroadcastShape(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                 bool fewerDimsOptimization = false);

  ~BroadcastShape() = default;

  /// If and only if isValid(), the following fields can be used in implementing
  /// a broadcasted binary tensor operation according to the broadcasting rule.
  ArrayRef<int64_t> getXReshape() const { return reshape[0]; }
  ArrayRef<int64_t> getXBCast() const { return bcast[0]; }
  ArrayRef<int64_t> getYReshape() const { return reshape[1]; }
  ArrayRef<int64_t> getYBCast() const { return bcast[1]; }

  /// Returns the mapping from the flattened output batch indices to x's
  /// flattened batch indices. The result is a vector of length
  /// getOutputBatchSize(). To compute the i'th batch output, a binary
  /// matmul-like operation should use the `getXBatchIndices()[i]`th batch index
  /// of `x`. Note: Returns an empty vector if broadcasting is not required.
  /// Callers should only use this when isBroadcastingRequired() returns true.

  ArrayRef<int64_t> getXBatchIndices() const { return batch_indices_[0]; }
  /// Returns the mapping from the flattened output batch indices to y's
  /// flattened batch indices. Similar to getXBatchIndices().
  /// Note: Returns an empty vector if broadcasting is not required. Callers
  /// should only use this when isBroadcastingRequired() returns true.
  ArrayRef<int64_t> getYBatchIndices() const { return batch_indices_[1]; }

private:
  BroadcastShape(const BroadcastShape &) = delete;
  void operator=(const BroadcastShape &) = delete;
};

} // namespace M

#endif // SUPPORT_ML_BROADCASTSHAPE_H
