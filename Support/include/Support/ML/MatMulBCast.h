//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains broadcast helpers for batched matmul-like operations,
// following the broadcasting logic of numpy and TensorFlow. This file should
// not depend on either of these libraries, but since their broadcasting logic
// is widely used, it is made available for all modules.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// File originates from:
//   Repo:   git@github.com:tensorflow/tensorflow.git
//   Commit: 0b7349102db619105fb282c2340a64c44e4adbe6
//   Path:   tensorflow/core/util/matmul_bcast.h
//
//===----------------------------------------------------------------------===//

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SUPPORT_ML_MATMULBCAST_H
#define SUPPORT_ML_MATMULBCAST_H

#include "Support/ML/BCast.h"
#include "Support/ML/TensorShape.h"

namespace M {

/// Simple wrapper over BCast specialized for MatMul. Provides utilities for
/// broadcasting across batch dimensions for binary MatMul-like operations. If
/// neither argument has batch dimensions (rank <= 2) then no broadcasting is
/// needed and the operation MatMul operation is considered valid.
class MatMulBCast {
public:
  using Vec = BCast::Vec;

  MatMulBCast(ArrayRef<int64_t> x, ArrayRef<int64_t> y);

  bool IsValid() const {
    return !broadcasting_required_ || (batch_bcast_ && batch_bcast_->IsValid());
  }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  int64_t output_batch_size() const { return output_batch_size_; }
  int64_t x_batch_size() const { return x_batch_size_; }
  int64_t y_batch_size() const { return y_batch_size_; }
  const TensorShape &output_batch_shape() const { return output_batch_shape_; }

  /// Returns the mapping from the flattened output batch indices to x's
  /// flattened batch indices. The result is a vector of length
  /// output_batch_size(). To compute the i'th batch output, a binary
  /// matmul-like operation should use the `x_batch_indices()[i]`th batch index
  /// of `x`. Note: Returns an empty vector if broadcasting is not required.
  /// Callers should only use this when IsBroadcastingRequired() returns true.
  ArrayRef<int64_t> x_batch_indices() const { return x_batch_indices_; }

  /// Returns the mapping from the flattened output batch indices to y's
  /// flattened batch indices. Similar to x_batch_indices(). Note: Returns an
  /// empty vector if broadcasting is not required. Callers should only use this
  /// when IsBroadcastingRequired() returns true.
  ArrayRef<int64_t> y_batch_indices() const { return y_batch_indices_; }

private:
  std::unique_ptr<BCast> batch_bcast_;
  bool broadcasting_required_ = false;
  int64_t x_batch_size_ = 1;
  int64_t y_batch_size_ = 1;
  TensorShape output_batch_shape_;
  int64_t output_batch_size_ = 1;
  SmallVector<int64_t> x_batch_indices_;
  SmallVector<int64_t> y_batch_indices_;
};

} // namespace M

#endif // SUPPORT_ML_MATMULBCAST_H
