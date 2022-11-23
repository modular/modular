//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/MatMulBCast.h"

using namespace M;

M::MatMulBCast::MatMulBCast(ArrayRef<int64_t> x, ArrayRef<int64_t> y) {
  if (std::max(x.size(), y.size()) == 2)
    return;
  const Vec x_resized(x.begin(), x.end() - 2);
  const Vec y_resized(y.begin(), y.end() - 2);

  batch_bcast_ = std::make_unique<BCast>(x_resized, y_resized);
  if (!batch_bcast_->IsValid()) {
    // Set broadcasting_required_ to true to make IsValid() return false;
    broadcasting_required_ = true;
    return;
  }

  x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).getNumElements();
  y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).getNumElements();
  output_batch_shape_ = TensorShape(batch_bcast_->output_shape());
  output_batch_size_ = output_batch_shape_.getNumElements();
  broadcasting_required_ =
      std::min(x_batch_size_, y_batch_size_) != output_batch_size_;

  if (broadcasting_required_) {
    ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                        batch_bcast_->x_bcast(), x_batch_indices_);
    ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                        batch_bcast_->y_bcast(), y_batch_indices_);
  }
}
