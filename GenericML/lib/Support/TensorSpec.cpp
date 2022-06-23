//===- TensorSpec.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/TensorSpec.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Stringification and printing methods
//===----------------------------------------------------------------------===//

void TensorSpec::print(raw_ostream &os) const {
  llvm::interleave(getDims(), os, "x");
  if (getRank() != 0)
    os << 'x';
  os << getEltType();
}

std::string TensorSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void TensorSpec::dump() const { print(llvm::errs()); }
