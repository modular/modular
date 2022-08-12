//===- POPOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the POP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static const llvm::fltSemantics &getFloatSemantics(DType dtype) {
  switch (dtype.getValue()) {
  default:
    return llvm::APFloat::Bogus();
  case DType::bf16:
    return llvm::APFloat::BFloat();
  case DType::f16:
    return llvm::APFloat::IEEEhalf();
  case DType::f32:
    return llvm::APFloat::IEEEsingle();
  case DType::f64:
    return llvm::APFloat::IEEEdouble();
  }

  llvm_unreachable("unhandled floating point semantics for dtype");
}

/// Checks if the DType constant can be cast from the MLIR type.
static bool isCastableFrom(Attribute value, DTypeConstantAttr dtypeAttr) {
  auto inputTy = value.cast<TypedAttr>().getType();
  // The types are compatible, so we have nothing else to do.
  if (dtypeAttr.isCompatibleWith(inputTy))
    return true;

  // We now check if we can cast the value to the dtype without loss of
  // accuracy.

  auto dtype = dtypeAttr.getDType();

  // Just reject if the dtype in an integer type, but the input is floating
  // point.
  if (dtype.isInt() && inputTy.isa<FloatType>())
    return false;

  // We now check if we can cast the input (which can be either floating
  // point or integer) to the floating point dtype.

  // If the input is a floating point value, then we just assume the input can
  // be coerced to the dtype.
  if (auto val = value.dyn_cast<FloatAttr>())
    return true;

  // Otherwise, the input is an integer value. We check if we can convert it to
  // the target type.
  assert(value.isa<IntegerAttr>() && "expected integer attribute");

  auto intVal = value.cast<IntegerAttr>().getValue();
  auto truncVal = intVal.trunc(dtype.getWidthInBits());

  // We check if we can truncate the input to the target type without loss of
  // accuracy.
  if (!APInt::isSameValue(intVal, truncVal))
    return false;

  // We now check if we can roundtrip the conversion to floating point and back.

  // First, convert the input to floating point for the given dtype floating
  // point semantics.
  auto floatVal = llvm::APFloat(getFloatSemantics(dtype));
  floatVal.convertFromAPInt(truncVal, false,
                            llvm::APFloat::rmNearestTiesToEven);

  // Then, convert the floating point value to an integer value.
  llvm::APSInt convertedVal(truncVal);
  bool isExact = false;
  floatVal.convertToInteger(convertedVal, llvm::APFloat::rmTowardZero,
                            &isExact);

  // We then check if the converted value is the same value as the one we
  // started with.
  return isExact && APInt::isSameValue(convertedVal, truncVal);
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

LogicalResult ConstantOp::verify() {
  auto valueType = getValue().dyn_cast<TypedAttr>().getType();
  // The value type should be a scalar type.
  if (!isBuildableWith(getValue(), valueType))
    return emitError(
        "constant has to be either an integer, float or index type");

  // The return type should castable from the input type or can be
  // parameterized.
  auto resultDType = getType().cast<ScalarType>().getDType();

  if (auto resultConstantDType = resultDType.dyn_cast<DTypeConstantAttr>()) {
    if (!isCastableFrom(getValue(), resultConstantDType))
      return emitOpError()
             << "expected the type of the constant input value (" << valueType
             << ") to be compatible with the dtype of the return value ('"
             << resultConstantDType.getDType().getAsString() << "').";
    return success();
  } else if (resultDType.isa<ParamDeclRefAttr>()) {
    return success();
  }

  llvm::report_fatal_error("unhandled output type");
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  return type.isa<IntegerType, FloatType, IndexType>();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

static Type getBoolOfSameParentType(Type type) {
  auto boolType = DTypeConstantAttr::get(type.getContext(), DType::kBool);
  if (auto scalar = type.dyn_cast<ScalarType>())
    return ScalarType::get(boolType);
  else if (auto simd = type.dyn_cast<SIMDType>())
    return SIMDType::get(simd.getSize(), boolType);
  return nullptr;
}

LogicalResult CmpOp::inferReturnTypes(MLIRContext *ctx, Optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      RegionRange regions,
                                      SmallVectorImpl<Type> &types) {
  Type argType = operands[0].getType();
  types.push_back(getBoolOfSameParentType(argType));
  if (types.back())
    return success();
  return mlir::emitError(loc.value_or(operands[0].getLoc()),
                         "expected a scalar or simd operand type");
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
