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

#ifndef SUPPORT_ML_BCAST_H
#define SUPPORT_ML_BCAST_H

#include <algorithm>

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"

namespace M {

/// Returns the mapping from the output batch indices to the corresponding
/// input's batch indices, given the input's "reshape" and "bcast" shapes as
/// returned by the BCastList helper class. The i'th element denotes the
/// (flattened) batch index of the input that must be used to compute the i'th
/// batch output.
void ComputeBatchIndices(int64_t output_batch_size, ArrayRef<int64_t> reshape,
                         ArrayRef<int64_t> bcast,
                         llvm::SmallVectorImpl<int64_t> &out_indices);

template <int N>
class BCastList {
public:
  /// A vector of int64_t representing the shape of tensor. The 0-th element is
  /// the outer-most dimension and the last element is the inner-most dimension.
  /// Note that we do not use TensorShape since it's more convenient to
  /// manipulate Vec directly for this module.
  using Vec = SmallVector<int64_t, 4>;

  /// Constructs all helper shapes, following the aforementioned rules.
  ///
  /// If "fewer_dims_optimization" is set to true (the default), the
  /// implementation tries to reduce intermediate dimensions needed to be more
  /// efficient. This is transparent to the caller. If false, all intermediate
  /// shapes (except for grad_{x,y}_reduce_idx()) have the same number of
  /// dimensions as the larger of the two inputs.
  ///
  /// If return_flattened_batch_indices is true, the implementation will compute
  /// for each output member of the flattened output, which batch indices of
  /// each input correspond to it. This is disabled by default.
  explicit BCastList(ArrayRef<ArrayRef<int64_t>>,
                     bool fewer_dims_optimization = true,
                     bool return_flattened_batch_indices = false);

  ~BCastList() = default;

  /// Returns true if and only if two operands are compatible according to the
  /// broadcasting rule.
  bool IsValid() const { return valid_; }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  /// If and only if IsValid(), the following fields can be used in implementing
  /// a broadcasted binary tensor operation according to the broadcasting rule.
  ArrayRef<int64_t> reshape(int i) const { return reshape_[i]; }
  ArrayRef<int64_t> bcast(int i) const { return bcast_[i]; }
  ArrayRef<int64_t> result_shape() const { return result_; }
  ArrayRef<int64_t> output_shape() const { return output_; }
  ArrayRef<int64_t> grad_reduce_idx(int i) const { return grad_reduce_idx_[i]; }
  int64_t output_batch_size() const { return output_batch_size_; }

  /// Returns the mapping from the flattened output batch indices to x's
  /// flattened batch indices. The result is a vector of length
  /// output_batch_size(). To compute the i'th batch output, a binary
  /// matmul-like operation should use the `x_batch_indices()[i]`th batch index
  /// of `x`. Note: Returns an empty vector if broadcasting is not required.
  /// Callers should only use this when IsBroadcastingRequired() returns true.
  ArrayRef<int64_t> batch_indices(int i) const { return batch_indices_[i]; }

protected:
  bool valid_ = true;
  bool broadcasting_required_ = true;
  SmallVector<Vec, N> reshape_;
  SmallVector<Vec, N> bcast_;
  Vec result_;
  Vec output_;
  SmallVector<Vec, N> grad_reduce_idx_;

  int64_t output_batch_size_;
  SmallVector<SmallVector<int64_t>, N> batch_indices_;

  static void Reverse(llvm::SmallVectorImpl<int64_t> &shape) {
    std::reverse(shape.begin(), shape.end());
  }

  BCastList(const BCastList &) = delete;
  void operator=(const BCastList &) = delete;
};

/// BCast is a helper for broadcasting binary tensor operations, following the
/// rules of numpy (See
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
/// BCast takes the shape of two tensors and computes a few vectors of int32
/// that are useful for the caller to reshape the tensors, apply the right
/// broadcasts to them, compute the broadcasted operation, and possibly the
/// gradients. In a nutshell, the caller is expected to compute the broadcasted
/// operation as following:
///
///   BCast b(x.shape(), y.shape());
///   output = x.reshape(b.x_reshape()).broadcast(b.x_bcast())
///            _op_
///            y.reshape(b.y_reshape()).broadcast(b.y_bcast())
///
/// For the gradient computation,
///   grad_x = sum(grad * backprop_x(x, y), grad_x_reduce_idx)
///            .reshape(x.shape())
///   grad_y = sum(grad * backprop_y(x, y), grad_y_reduce_idx)
///            .reshape(y.shape())
/// backprop_x and backprop_y are functionals of the binary function "op", e.g.,
///   for +, backprop_x(x, y) = backprop_y(x, y) = 1;
///   for *, backprop_x(x, y) =  y, backprop_y(x, y) = x;
///   for /, backprop_x(x, y) = 1/y, backprop_y(x, y) = -x/y^2;
///
/// The multiplication in the grad * backprop_x itself is also broadcasting
/// following the same rule.
class BCast : public BCastList<2> {
public:
  /// Constructs all helper shapes, following the aforementioned rules.
  ///
  /// If "fewer_dims_optimization" is set to true (the default), the
  /// implementation tries to reduce intermediate dimensions needed to be more
  /// efficient. This is transparent to the caller.
  ///
  /// If false, all intermediate shapes (except for grad_{x,y}_reduce_idx())
  /// have the same number of dimensions as the larger of the two inputs.
  using Vec = SmallVector<int64_t, 4>;

  BCast(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
        bool fewer_dims_optimization = true,
        bool return_flattened_batch_indices = false);

  ~BCast() = default;

  /// If and only if IsValid(), the following fields can be used in implementing
  /// a broadcasted binary tensor operation according to the broadcasting rule.
  ArrayRef<int64_t> x_reshape() const { return reshape_[0]; }
  ArrayRef<int64_t> x_bcast() const { return bcast_[0]; }
  ArrayRef<int64_t> y_reshape() const { return reshape_[1]; }
  ArrayRef<int64_t> y_bcast() const { return bcast_[1]; }
  ArrayRef<int64_t> result_shape() const { return result_; }
  ArrayRef<int64_t> output_shape() const { return output_; }
  ArrayRef<int64_t> grad_x_reduce_idx() const { return grad_reduce_idx_[0]; }
  ArrayRef<int64_t> grad_y_reduce_idx() const { return grad_reduce_idx_[1]; }

  /// Returns the mapping from the flattened output batch indices to x's
  /// flattened batch indices. The result is a vector of length
  /// output_batch_size(). To compute the i'th batch output, a binary
  /// matmul-like operation should use the `x_batch_indices()[i]`th batch index
  /// of `x`. Note: Returns an empty vector if broadcasting is not required.
  /// Callers should only use this when IsBroadcastingRequired() returns true.

  ArrayRef<int64_t> x_batch_indices() const { return batch_indices_[0]; }
  /// Returns the mapping from the flattened output batch indices to y's
  /// flattened batch indices. Similar to x_batch_indices().
  /// Note: Returns an empty vector if broadcasting is not required. Callers
  /// should only use this when IsBroadcastingRequired() returns true.
  ArrayRef<int64_t> y_batch_indices() const { return batch_indices_[1]; }

private:
  BCast(const BCast &) = delete;
  void operator=(const BCast &) = delete;
};

} // namespace M

#endif // SUPPORT_ML_BCAST_H
