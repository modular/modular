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
