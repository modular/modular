//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/MLIRDType.h"
#include "Support/ML/DType.h"

using namespace M;

bool M::areEquivalentFloatTypes(DType dtype, FloatType fpType) {
  assert(dtype.isFloat() && "expected a float dtype");
  switch (dtype.getValue()) {
  case DType::f16:
    return fpType.isa<Float16Type>();
  case DType::f32:
    return fpType.isa<Float32Type>();
  case DType::f64:
    return fpType.isa<Float64Type>();
  case DType::f80:
    return fpType.isa<Float80Type>();
  case DType::f128:
    return fpType.isa<Float128Type>();
  case DType::bf16:
    return fpType.isa<BFloat16Type>();
  default:
    return false;
  }
}

FloatType M::getEquivalentFloatType(MLIRContext *ctx, DType dtype) {
  switch (dtype.getValue()) {
  default:
    return nullptr;
  case DType::f16:
    return FloatType::getF16(ctx);
  case DType::bf16:
    return FloatType::getBF16(ctx);
  case DType::f32:
    return FloatType::getF32(ctx);
  case DType::f64:
    return FloatType::getF64(ctx);
  case DType::f80:
    return FloatType::getF80(ctx);
  case DType::f128:
    return FloatType::getF128(ctx);
  }
}

IntegerType M::getEquivalentIntegerType(MLIRContext *ctx, DType dtype) {
  return IntegerType::get(ctx, dtype.getWidthInBits(),
                          dtype.isSInt() ? IntegerType::Signed
                                         : IntegerType::Unsigned);
}

DType M::getEquivalentDType(FloatType fpType) {
  if (fpType.isF16())
    return DType(DType::f16);
  if (fpType.isBF16())
    return DType(DType::bf16);
  if (fpType.isF32())
    return DType(DType::f32);
  if (fpType.isF64())
    return DType(DType::f64);
  if (fpType.isF80())
    return DType(DType::f80);
  if (fpType.isF128())
    return DType(DType::f128);
  return {}; // unrepresentable
}

DType M::getEquivalentDType(IntegerType intType) {
  if (intType.isSignless())
    return {}; // unrepresentable
  FailureOr<DType> optDType =
      DType::getInt(intType.getIntOrFloatBitWidth(), intType.isSignedInteger());
  if (failed(optDType))
    return {}; // unrepresentable
  return *optDType;
}