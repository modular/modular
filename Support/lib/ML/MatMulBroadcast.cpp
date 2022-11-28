//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/MatMulBroadcast.h"

using namespace M;

M::MatMulBroadcast::MatMulBroadcast(ArrayRef<int64_t> x, ArrayRef<int64_t> y) {
  if (std::max(x.size(), y.size()) == 2)
    return;
  const Vec xResized(x.begin(), x.end() - 2);
  const Vec yResized(y.begin(), y.end() - 2);

  batchBCast = std::make_unique<BroadcastShape>(xResized, yResized,
                                                /*fewerDimsOptimization=*/true);
  if (!batchBCast->isValid()) {
    // Set broadcastingRequired to true to make isValid() return false;
    broadcastingRequired = true;
    return;
  }

  xBatchSize = TensorShape(batchBCast->getXReshape()).getNumElements();
  yBatchSize = TensorShape(batchBCast->getYReshape()).getNumElements();
  outputBatchShape = llvm::to_vector(batchBCast->getOutputShape());
  outputBatchSize = TensorShape(outputBatchShape).getNumElements();
  broadcastingRequired = std::min(xBatchSize, yBatchSize) != outputBatchSize;

  if (broadcastingRequired) {
    computeBatchIndices(outputBatchSize, batchBCast->getXReshape(),
                        batchBCast->getXBCast(), xBatchIndices);
    computeBatchIndices(outputBatchSize, batchBCast->getYReshape(),
                        batchBCast->getYBCast(), yBatchIndices);
  }
}
