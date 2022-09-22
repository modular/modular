//===- MLIRDenseAttrStorage.cpp -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/MLIRDenseAttrStorage.h"

using namespace M;

bool M::shouldUseOutOfLineAttrStorage(size_t numElements) {
  // A sufficiently large element threshold is used to avoid treating large
  // arrays as "free". The storage, constant folding, etc. of large arrays
  // should be treated specially to ensure we don't bloat generated code, memory
  // use, and more.
  static constexpr size_t kLargeDataThreshold = 512;

  return numElements > kLargeDataThreshold;
}
