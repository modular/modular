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
#include "mlir/IR/TypeRange.h"
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

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

/// Checks if the DType constant can be cast from the MLIR type.
static bool isCastableFrom(Attribute value, DTypeConstantAttr dtypeAttr) {
  auto inputTy = value.cast<TypedAttr>().getType();
  if (!inputTy.isa<IndexType, IntegerType, FloatType>())
    return false;

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

  // If the target type is an integer, check if the integer value can be
  // truncated without loss.
  if (dtype.isInt()) {
    if (dtype.getWidthInBits() > inputTy.getIntOrFloatBitWidth())
      return true;
    return APInt::isSameValue(intVal, intVal.trunc(dtype.getWidthInBits()));
  }

  // We now check if we can roundtrip the conversion to floating point and back.

  // First, convert the input to floating point for the given dtype floating
  // point semantics.
  auto floatVal = llvm::APFloat(getFloatSemantics(dtype));
  floatVal.convertFromAPInt(intVal, false, llvm::APFloat::rmNearestTiesToEven);

  // Then, convert the floating point value to an integer value.
  llvm::APSInt convertedVal(intVal);
  bool isExact = false;
  floatVal.convertToInteger(convertedVal, llvm::APFloat::rmTowardZero,
                            &isExact);

  // We then check if the converted value is the same value as the one we
  // started with.
  return isExact && APInt::isSameValue(convertedVal, intVal);
}

/// Check whether an attribute can be materialized by a constant of the given
/// result type.
static LogicalResult
canMaterializeConstant(TypedAttr attr, Type type,
                       function_ref<InFlightDiagnostic(StringRef)> emitError) {
  if (type.isa<ScalarType>()) {
    auto checkDType = [&](DTypeConstantAttr dtype) -> LogicalResult {
      if (isCastableFrom(attr, dtype))
        return mlir::success();
      return emitError("expected the type of the constant input value (")
             << attr.getType()
             << ") to be compatible with the dtype of the return value ('"
             << dtype.getDType().getAsString() << "').";
    };
    return checkMetaCastedTypes(emitError, type, attr.getType(), checkDType);
  }

  auto elements = attr.dyn_cast<DenseElementsAttr>();
  if (!elements)
    return emitError(
        "expected vector constant to be a dense elements attribute");
  auto checkDType = [&](DTypeConstantAttr dtype) -> LogicalResult {
    for (Attribute element : elements.getValues<Attribute>()) {
      if (!isCastableFrom(element, dtype))
        return emitError("cannot cast from vector element to ")
               << dtype.getDType().getAsString() << ": " << element;
    }
    return mlir::success();
  };
  return checkMetaCastedTypes(emitError, type, attr.getType(), checkDType);
}

LogicalResult ConstantOp::verify() {
  return canMaterializeConstant(getValue(), getType(), [this](StringRef msg) {
    return emitOpError(msg);
  });
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  return succeeded(canMaterializeConstant(value, type, [](StringRef msg) {
    InFlightDiagnostic diag;
    diag.abandon();
    return diag;
  }));
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
  if (type.isa<ScalarType>())
    return ScalarType::get(boolType);
  if (auto simd = type.dyn_cast<SIMDType>())
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
// BitcastOp
//===----------------------------------------------------------------------===//

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  auto inputType = inputs.front().cast<DTypeInterface>();
  auto outputType = outputs.front().cast<DTypeInterface>();
  // The input and output types must be of the same kind.
  // TODO: In theory we can support casting a scalar type to a vector type (e.g.
  // f64 to a 2xf32) or vice versa. We should support this when the use case
  // arises.
  if (inputType.isa<ScalarType>() != outputType.isa<ScalarType>())
    return false;

  auto inputDType = inputType.resolveDType();
  auto outputDType = outputType.resolveDType();

  // If we cannot resolve the dtype, then we cannot cast.
  if (inputDType.isInvalid() || outputDType.isInvalid())
    return false;

  auto inputDTypeWidth = inputDType.getWidthInBits();
  auto outputDTypeWidth = outputDType.getWidthInBits();

  // If we have a scalar type, then the bitwidths must match.
  if (auto inputSimd = inputType.dyn_cast<SIMDType>()) {
    auto outputSimd = outputType.cast<SIMDType>();
    auto inputSimdSize = inputSimd.resolveSize();
    auto outputSimdSize = outputSimd.resolveSize();
    // If we cannot resolve the sizes, then we cannot verify the cast.
    if (!inputSimdSize || !outputSimdSize)
      return false;
    // If the sizes do not match, then we cannot cast.
    return inputSimdSize.value() * inputDTypeWidth ==
           outputSimdSize.value() * outputDTypeWidth;
  }

  // Otherwise, we have a scalar type. So the bitwidths must match.
  return inputDTypeWidth == outputDTypeWidth;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  if (getInput().getType() == getOutput().getType())
    return getInput();
  return {};
}

LogicalResult CastOp::verify() {
  auto inputType = getInput().getType().cast<DTypeInterface>();
  auto outputType = getOutput().getType().cast<DTypeInterface>();
  if (inputType.isa<ScalarType>() != outputType.isa<ScalarType>())
    return emitOpError("cannot cast between a scalar type and SIMD type");

  if (auto inputSimd = inputType.dyn_cast<SIMDType>();
      inputSimd && inputSimd.getSize() != outputType.cast<SIMDType>().getSize())
    return emitOpError("cannot cast between SIMD types of different sizes");

  return success();
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

static Type getScalarOfSameDType(Type type) {
  return ScalarType::get(type.getContext(),
                         type.cast<DTypeInterface>().getDType());
}

//===----------------------------------------------------------------------===//
// BufferStackAllocationOp
//===----------------------------------------------------------------------===//

LogicalResult BufferStackAllocationOp::verify() {
  if (!getType().cast<BufferType>().getSize())
    return emitOpError("cannot stack allocate a buffer of unknown size");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
