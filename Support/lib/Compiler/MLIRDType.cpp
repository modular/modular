//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/MLIRDType.h"
#include "Support/ML/DType.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;

bool M::areEquivalentFloatTypes(DType dtype, mlir::FloatType fpType) {
  assert(dtype.isFloat() && "expected a float dtype");
  switch (dtype.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  case DType::SHORT_NAME:                                                      \
    return isa<MLIR_TYPE>(fpType);
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  default:
    return false;
  }
}

FloatType M::getEquivalentFloatType(MLIRContext *ctx, DType dtype) {
  switch (dtype.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  case DType::SHORT_NAME:                                                      \
    return MLIR_TYPE::get(ctx);
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  default:
    return {}; // null denotes failure
  }
}

bool M::hasEquivalentFloatType(DType dtype) {
  switch (dtype.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, ...) case DType::SHORT_NAME:
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
    return true;
  default:
    return false;
  }
}

IntegerType M::getEquivalentIntegerType(MLIRContext *ctx, DType dtype) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1, IntegerType::Signless);
  if (dtype.isInt())
    return IntegerType::get(ctx, dtype.getWidthInBits(),
                            dtype.isSInt() ? IntegerType::Signed
                                           : IntegerType::Unsigned);
  return {}; // null denotes failure
}

bool M::hasEquivalentIntegerType(DType dtype) {
  return dtype.isInt() || dtype.isBool();
}

DType M::getEquivalentDType(FloatType fpType) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  if (llvm::isa<MLIR_TYPE>(fpType))                                            \
    return DType::SHORT_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  return {}; // invalid denotes failure
}

DType M::getEquivalentDType(IntegerType intType) {
  if (intType.isSignless()) {
    if (intType.getWidth() == 1)
      return DType::kBool;
    else
      return {}; // invalid denotes failure
  }
  FailureOr<DType> optDType =
      DType::getInt(intType.getIntOrFloatBitWidth(), intType.isSignedInteger());
  if (failed(optDType))
    return {}; // invalid denotes failure
  return *optDType;
}
