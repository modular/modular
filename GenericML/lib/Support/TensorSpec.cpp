//===- TensorSpec.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/TensorSpec.h"
#include "llvm/Support/raw_ostream.h"
using namespace M;

void M::printTensorSpec(ArrayRef<ssize_t> dims, TensorEltType eltType,
                        raw_ostream &os) {
  llvm::interleave(dims, os, "x");
  if (!dims.empty())
    os << 'x';
  os << eltType;
}

std::string M::getTensorSpecAsString(ArrayRef<ssize_t> dims,
                                     TensorEltType eltType) {
  std::string str;
  llvm::raw_string_ostream os(str);
  printTensorSpec(dims, eltType, os);
  return os.str();
}

//===----------------------------------------------------------------------===//
// Dump methods
//===----------------------------------------------------------------------===//

void TensorSpec::dump() const { print(llvm::errs()); }
void CompactTensorSpec::dump() const { print(llvm::errs()); }
