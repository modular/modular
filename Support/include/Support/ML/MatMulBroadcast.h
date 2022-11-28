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

#ifndef SUPPORT_ML_MATMULBROADCAST_H
#define SUPPORT_ML_MATMULBROADCAST_H

#include "Support/ML/BroadcastShape.h"
#include "Support/ML/TensorShape.h"

namespace M {

/// Simple wrapper over BroadcastShape specialized for MatMul. Provides
/// utilities for broadcasting across batch dimensions for binary MatMul-like
/// operations. If neither argument has batch dimensions (rank <= 2) then no
/// broadcasting is needed and the operation MatMul operation is considered
/// valid.
class MatMulBroadcast {
public:
  using Vec = BroadcastShape::Vec;

  MatMulBroadcast(ArrayRef<int64_t> x, ArrayRef<int64_t> y);

  bool isValid() const {
    return !broadcastingRequired || (batchBCast && batchBCast->isValid());
  }
  bool isBroadcastingRequired() const { return broadcastingRequired; }

  int64_t getOutputBatchSize() const { return outputBatchSize; }
  int64_t getXBatchSize() const { return xBatchSize; }
  int64_t getYBatchSize() const { return yBatchSize; }
  const TensorShape &getOutputBatchShape() const { return outputBatchShape; }

  /// Returns the mapping from the flattened output batch indices to x's
  /// flattened batch indices. The result is a vector of length
  /// getOutputBatchSize(). To compute the i'th batch output, a binary
  /// matmul-like operation should use the `getXBatchIndices()[i]`th batch index
  /// of `x`. Note: Returns an empty vector if broadcasting is not required.
  /// Callers should only use this when isBroadcastingRequired() returns true.
  ArrayRef<int64_t> getXBatchIndices() const { return xBatchIndices; }

  /// Returns the mapping from the flattened output batch indices to y's
  /// flattened batch indices. Similar to getXBatchIndices(). Note: Returns an
  /// empty vector if broadcasting is not required. Callers should only use this
  /// when isBroadcastingRequired() returns true.
  ArrayRef<int64_t> getYBatchIndices() const { return yBatchIndices; }

private:
  std::unique_ptr<BroadcastShape> batchBCast;
  bool broadcastingRequired = false;
  int64_t xBatchSize = 1;
  int64_t yBatchSize = 1;
  TensorShape outputBatchShape;
  int64_t outputBatchSize = 1;
  SmallVector<int64_t> xBatchIndices;
  SmallVector<int64_t> yBatchIndices;
};

} // namespace M

#endif // SUPPORT_ML_MATMULBROADCAST_H
