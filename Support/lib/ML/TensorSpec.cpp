//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/TensorSpec.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Stringification and printing methods
//===----------------------------------------------------------------------===//

void TensorSpec::print(raw_ostream &os) const {
  llvm::interleave(
      *this, os,
      [&](ssize_t dim) {
        if (mlir::ShapedType::isDynamic(dim)) {
          os << "?";
          return;
        }
        os << dim;
      },
      "x");
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
