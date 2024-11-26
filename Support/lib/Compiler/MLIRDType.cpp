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
  case DType::f8e5m2:
    return llvm::isa<Float8E5M2Type>(fpType);
  case DType::f8e5m2fnuz:
    return llvm::isa<Float8E5M2FNUZType>(fpType);
  case DType::f8e4m3:
    return llvm::isa<Float8E4M3Type>(fpType);
  case DType::f8e4m3fnuz:
    return llvm::isa<Float8E4M3FNUZType>(fpType);
  case DType::f8e3m4:
    return llvm::isa<Float8E3M4Type>(fpType);
  case DType::f16:
    return llvm::isa<Float16Type>(fpType);
  case DType::bf16:
    return llvm::isa<BFloat16Type>(fpType);
  case DType::f32:
    return llvm::isa<Float32Type>(fpType);
  case DType::f64:
    return llvm::isa<Float64Type>(fpType);
  case DType::f80:
    return llvm::isa<Float80Type>(fpType);
  case DType::f128:
    return llvm::isa<Float128Type>(fpType);
  default:
    return false;
  }
}

FloatType M::getEquivalentFloatType(MLIRContext *ctx, DType dtype) {
  switch (dtype.getValue()) {
  case DType::f8e5m2:
    return FloatType::getFloat8E5M2(ctx);
  case DType::f8e5m2fnuz:
    return FloatType::getFloat8E5M2FNUZ(ctx);
  case DType::f8e4m3:
    return FloatType::getFloat8E4M3(ctx);
  case DType::f8e4m3fnuz:
    return FloatType::getFloat8E4M3FNUZ(ctx);
  case DType::f8e3m4:
    return FloatType::getFloat8E3M4(ctx);
  case DType::f16:
    return FloatType::getF16(ctx);
  case DType::bf16:
    return FloatType::getBF16(ctx);
  case DType::f32:
    return FloatType::getF32(ctx);
  case DType::tf32:
    return FloatType::getTF32(ctx);
  case DType::f64:
    return FloatType::getF64(ctx);
  case DType::f80:
    return FloatType::getF80(ctx);
  case DType::f128:
    return FloatType::getF128(ctx);
  default:
    return {}; // null denotes failure
  }
}

bool M::hasEquivalentFloatType(DType dtype) {
  switch (dtype.getValue()) {
  case DType::f8e5m2:
  case DType::f8e5m2fnuz:
  case DType::f8e4m3:
  case DType::f8e4m3fnuz:
  case DType::f8e3m4:
  case DType::f16:
  case DType::bf16:
  case DType::f32:
  case DType::f64:
  case DType::f80:
  case DType::f128:
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
  if (fpType.isFloat8E5M2())
    return DType::f8e5m2;
  if (fpType.isFloat8E5M2FNUZ())
    return DType::f8e5m2fnuz;
  if (fpType.isFloat8E4M3())
    return DType::f8e4m3;
  if (fpType.isFloat8E4M3FNUZ())
    return DType::f8e4m3fnuz;
  if (fpType.isFloat8E3M4())
    return DType::f8e3m4;
  if (fpType.isF16())
    return DType::f16;
  if (fpType.isBF16())
    return DType::bf16;
  if (fpType.isF32())
    return DType::f32;
  if (fpType.isF64())
    return DType::f64;
  if (fpType.isF80())
    return DType::f80;
  if (fpType.isF128())
    return DType::f128;
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
