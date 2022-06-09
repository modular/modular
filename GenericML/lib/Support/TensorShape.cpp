//===- TensorShape.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/TensorShape.h"
#include "llvm/Support/raw_ostream.h"
using namespace M;

void M::printTensorShape(ArrayRef<ssize_t> dims, raw_ostream &os) {
  llvm::interleave(dims, os, "x");
}

std::string M::getTensorShapeAsString(ArrayRef<ssize_t> dims) {
  std::string str;
  llvm::raw_string_ostream os(str);
  printTensorShape(dims, os);
  return os.str();
}

//===----------------------------------------------------------------------===//
// Dump methods
//===----------------------------------------------------------------------===//

void CompactTensorShape::dump() const { print(llvm::errs()); }
