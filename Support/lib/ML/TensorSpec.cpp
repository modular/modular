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
// Parsing methods
//===----------------------------------------------------------------------===//

/// Parses a string of the form dim0xdim1x...xDType into a TensorSpec.
ErrorOr<TensorSpec> TensorSpec::parseFromString(StringRef str) {
  // dtype is the portion after the last 'x'.  Shape is the portion before
  // that.  If there is no 'x', the whole string is the dtype.  (rsplit almost
  // does this, but the no-'x' case would put the string in shape instead of
  // dtype.)
  auto lastXIndex = str.rfind('x');
  StringRef shapeStr, dtypeStr;
  if (lastXIndex == StringRef::npos) {
    shapeStr = "";
    dtypeStr = str;
  } else {
    shapeStr = str.slice(0, lastXIndex);
    dtypeStr = str.slice(lastXIndex + 1, StringRef::npos);
  }

  auto shape = TensorShape::parseFromString(shapeStr);
  if (failed(shape))
    return Error(Twine("could not parse shape from string: ") + str + ": " +
                 shape.getError());

  auto dtype = DType::getFromString(dtypeStr);
  if (failed(dtype))
    return Error(Twine("could not parse dtype from string: ") + str +
                 " because " + dtypeStr + " is not a valid DType");

  // Create the tensor spec from the shape and dtype information.
  return TensorSpec(*shape, *dtype);
}

//===----------------------------------------------------------------------===//
// Stringification and printing methods
//===----------------------------------------------------------------------===//

void TensorSpec::print(raw_ostream &os) const {
  TensorShape::print(os);
  if (!(hasRank() && getRank() == 0))
    os << "x";
  os << getEltType();
}

std::string TensorSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}
