//===- TensorSpec.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/QuantSpec.h"
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

//===----------------------------------------------------------------------===//
// Static helpers
//===----------------------------------------------------------------------===//

ErrorOr<QuantTensorSpec> QuantTensorSpec::get(BEFType type) {
  SmallVector<int64_t> dims;
  auto tensorType = dyn_cast<BEFTensorType>(type);
  if (!tensorType)
    return Error("BEFType cannot be cast to BEFTensorType");

  auto befType = tensorType->decode(dims);
  if (auto qType = dyn_cast<BEFQuantizedType>(befType)) {
    float scale;
    int64_t zeroPoint;
    auto [storageType, expressedType] = qType->decode(scale, zeroPoint);
    return QuantTensorSpec(TensorSpec(dims, storageType.getKind()),
                           QuantSpec::getUniform(scale, zeroPoint));
  } else if (auto qType = dyn_cast<BEFQuantizedPerAxisType>(befType)) {
    SmallVector<float> scales;
    SmallVector<int64_t> zeroPoints;
    uint8_t quantDim;
    auto [storageType, expressedType] =
        qType->decode(scales, zeroPoints, quantDim);

    return QuantTensorSpec(
        TensorSpec(dims, storageType.getKind()),
        QuantSpec::getUniformPerAxis(quantDim, scales, zeroPoints));
  }
  // BEFType are GenericML TensorEltType kinds.
  return QuantTensorSpec(TensorSpec(dims, befType.getKind()), {});
}
