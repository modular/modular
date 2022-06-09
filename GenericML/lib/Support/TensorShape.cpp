//===- TensorShape.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/TensorShape.h"
#include "llvm/Support/raw_ostream.h"
using namespace M;

void TensorShape::print(raw_ostream &os) const {
  llvm::interleave(getDims(), os, "x");
}

std::string TensorShape::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void TensorShape::dump() const { print(llvm::errs()); }
