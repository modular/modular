//===- MLSupport/TensorShape.cpp ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLSupport/TensorShape.h"
using namespace M;

template <typename Collection>
static void printShapeInternal(const Collection &dimensions, raw_ostream &os) {
  os << '[';
  llvm::interleaveComma(dimensions, os);
  os << ']';
}

/// This is used by the FixedRankTensorShape template so we don't have to
/// instantiate this code for every rank.
void M::printShape(ArrayRef<ssize_t> dimensions, raw_ostream &os) {
  printShapeInternal(dimensions, os);
}

void TensorShape::print(raw_ostream &os) const { printShape(storage, os); }
void TensorShape::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// CompactTensorShape
//===----------------------------------------------------------------------===//

void CompactTensorShape::print(raw_ostream &os) const {
  printShapeInternal(*this, os);
}

void CompactTensorShape::dump() const { print(llvm::errs()); }
