//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/MatMulBroadcast.h"

using namespace M;

/// Return the total number of elements in this tensor, which is the product of
/// all the dimension sizes.
static int64_t getNumElements(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (auto dim : shape)
    result = multiplySymDims(result, dim);
  return result;
}

M::MatMulBroadcast::MatMulBroadcast(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                                    bool fewerDimsOptimization) {
  if (std::max(x.size(), y.size()) == 2)
    return;
  const Vec xResized(x.begin(), x.end() - 2);
  const Vec yResized(y.begin(), y.end() - 2);

  batchBCast = std::make_unique<BroadcastShape>(xResized, yResized,
                                                fewerDimsOptimization);
  if (!batchBCast->isValid()) {
    // Set broadcastingRequired to true to make isValid() return false;
    broadcastingRequired = true;
    return;
  }

  xBatchSize = getNumElements(batchBCast->getXReshape());
  yBatchSize = getNumElements(batchBCast->getYReshape());
  outputBatchShape = llvm::to_vector(batchBCast->getOutputShape());
  outputBatchSize = getNumElements(outputBatchShape);

  // When fewerDimsOptimization is false, we don't calculate batch indices.
  if (!fewerDimsOptimization) {
    broadcastingRequired =
        !batchBCast->getXReshape().equals(batchBCast->getYReshape());
    return;
  }

  broadcastingRequired = std::min(xBatchSize, yBatchSize) != outputBatchSize;
  if (broadcastingRequired) {
    computeBatchIndices(outputBatchSize, batchBCast->getXReshape(),
                        batchBCast->getXBCast(), xBatchIndices);
    computeBatchIndices(outputBatchSize, batchBCast->getYReshape(),
                        batchBCast->getYBCast(), yBatchIndices);
  }
}
