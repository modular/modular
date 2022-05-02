//===- MLSupport/TensorShape.cpp
//-------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLSupport/TensorShape.h"
#include "llvm/Support/raw_ostream.h"
using namespace M;

void TensorShape::print(raw_ostream &os) const {
  os << '[';
  llvm::interleaveComma(*this, os);
  os << ']';
}

void TensorShape::dump() const { print(llvm::errs()); }
