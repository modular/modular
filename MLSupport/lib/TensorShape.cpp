//===- MLSupport/TensorShape.cpp
//-------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLSupport/TensorShape.h"
using namespace M;

void M::printShape(ArrayRef<ssize_t> dimensions, raw_ostream &os) {
  os << '[';
  llvm::interleaveComma(dimensions, os);
  os << ']';
}

void TensorShape::print(raw_ostream &os) const { printShape(storage, os); }
void TensorShape::dump() const { print(llvm::errs()); }

