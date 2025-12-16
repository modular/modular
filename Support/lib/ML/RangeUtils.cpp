//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/RangeUtils.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <cstdlib>

using namespace M;

ErrorOr<int64_t> M::getRangeNumElements(int64_t start, int64_t end,
                                        int64_t step) {
  if (step == 0)
    return Error("step must not be zero");

  bool stepSign = step > 0;
  int64_t intervalLen = (stepSign ? 1 : -1) * (end - start);
  if (intervalLen < 0) {
    return Error(Twine("limit must be ") + (stepSign ? "greater" : "less") +
                 " than or equal to start when step is " +
                 (stepSign ? "positive" : "negative"));
  }

  return llvm::divideCeil(intervalLen, std::abs(step));
}
