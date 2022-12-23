//===----------------------------------------------------------------------===//
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
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Reify an integer to another integer.
static ErrorOr<APSInt> reifyIntToInt(const APInt &value, IntegerType type,
                                     KGENDType dtype) {
  // Integer to integer conversion. Check that this isn't converting between
  // signed or unsigned integers.
  if (!dtype.isBool() && !type.isSignlessInteger() &&
      type.isSignedInteger() != dtype.isSInt()) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    os << "cannot change signfulness when converting from " << type << " to "
       << dtype.getAsString();
    return Error(std::move(os.str()));
  }

  // Truncate or extend the value depending on the result width.
  // If casting to a bool, do c-style downcast on value first
  APSInt origInt(dtype.isBool() ? APInt(1, value.getBoolValue()) : value,
                 dtype.isUInt() || dtype.isBool());
  APSInt intValue =
      origInt.extOrTrunc(dtype.isIndex()  ? IndexType::kInternalStorageBitWidth
                         : dtype.isBool() ? 1
                                          : dtype.getWidthInBits());
  if (intValue.extOrTrunc(origInt.getBitWidth()) != origInt)
    return Error("integer constant does not fit into " + dtype.getAsString());

  return intValue;
}

/// Reify an integer to a float.
static ErrorOr<APFloat> reifyIntToFloat(const APInt &value, IntegerType type,
                                        FloatType fpType, KGENDType dtype) {
  // Roundtrip the integer value through float.
  APFloat apFp(fpType.getFloatSemantics());
  apFp.convertFromAPInt(value, !type.isUnsigned(),
                        APFloat::rmNearestTiesToEven);
  APSInt apInt(type.getIntOrFloatBitWidth(), type.isUnsigned());
  bool exact;
  apFp.convertToInteger(apInt, APFloat::rmTowardZero, &exact);

  // Fail if the roundtrip was lossy.
  if (!exact || !APInt::isSameValue(apInt, value))
    return Error("integer constant could not be exactly converted to " +
                 dtype.getAsString());
  return apFp;
}

/// Reify a float to an integer.
static ErrorOr<APSInt> reifyFloatToInt(const APFloat &value, DType dtype) {
  // If casting to a bool, do c-style downcast on value first
  APFloat newValue = value;
  if (dtype.isBool())
    newValue = value.isZero() ? APFloat(0.0) : APFloat(1.0);
  // Float to integer conversion. Only exact integers can be converted.
  if (!newValue.isInteger())
    return Error("only exact integer floats can be converted to integers");

  // Convert the float to an integer.
  APSInt apInt(dtype.isBool() ? 1 : dtype.getWidthInBits(),
               dtype.isUInt() || dtype.isBool());
  bool exact;
  newValue.convertToInteger(apInt, APFloat::rmTowardZero, &exact);
  assert(exact && "expected an exact integer");
  return apInt;
}

/// Reify a float to a float.
static ErrorOr<APFloat> reifyFloatToFloat(APFloat apFp, FloatType fpType) {
  // Coerce the floating point type, regardless of lossiness.
  bool lossy;
  apFp.convert(fpType.getFloatSemantics(), APFloat::rmTowardZero, &lossy);
  return apFp;
}

/// Reify a single bool, integer, integer, or float attribute to an attribute
/// that fits the given dtype.
static ErrorOr<TypedAttr> reifyOneAttribute(Attribute attr, KGENDType dtype) {
  if (auto value = dyn_cast<IntegerAttr>(attr)) {
    auto type = value.getType().cast<IntegerType>();

    if (!dtype.isIndex() && !dtype.isInt() && !dtype.isFloat() &&
        !dtype.isBool())
      return Error("cannot coerce constant value to " + dtype.getAsString());

    if (dtype.isBool() || dtype.isInt() || dtype.isIndex()) {
      UNWRAP_ERROR(intValue, reifyIntToInt(value.getValue(), type, dtype));
      return IntegerAttr::get(attr.getContext(), intValue);
    }

    // Integer to float conversion. Check for a valid floating point type.
    FloatType fpType = getEquivalentFloatType(type.getContext(), dtype);
    if (!fpType)
      return Error("unsupported floating point type: " + dtype.getAsString());
    UNWRAP_ERROR(apFp, reifyIntToFloat(value.getValue(), type, fpType, dtype));
    return FloatAttr::get(fpType, apFp);
  }

  auto value = attr.cast<FloatAttr>();
  if (dtype.isInt() || dtype.isBool()) {
    UNWRAP_ERROR(apInt, reifyFloatToInt(value.getValue(), dtype));
    return IntegerAttr::get(attr.getContext(), apInt);
  }

  // Float to float conversion. Check for a valid floating point type.
  FloatType fpType = getEquivalentFloatType(attr.getContext(), dtype);
  if (!fpType)
    return Error("unsupported floating point type: " + dtype.getAsString());

  return FloatAttr::get(
      fpType, reifyFloatToFloat(value.getValue(), fpType).takeValue());
}

/// Reify a range of floats or integers.
template <typename OutAttrT, typename InAttrT, typename ConvertValueFn>
static ErrorOr<TypedAttr> reifyArray(InAttrT attr, ConvertValueFn &&convert,
                                     Type newElementType) {
  using T = decltype(*attr.getValues().begin());
  SmallVector<decltype(convert(std::declval<T>()).takeValue())> values;
  values.reserve(attr.size());
  for (T value : attr.getValues()) {
    UNWRAP_ERROR(newValue, convert(value));
    values.push_back(std::move(newValue));
  }
  ShapedType newShapedType = attr.getType().cloneWith({}, newElementType);
  return OutAttrT::get(newShapedType, values);
}

/// Reify a primitive constant attribute (index, integer, float, or vector
/// thereof) to an attribute that fits the given type.
static ErrorOr<TypedAttr> reifyConstant(TypedAttr attr, DType dtype,
                                        Type type) {
  // If the value is an integer or float attribute, reify it to according to the
  // result dtype.
  if (attr.isa<IntegerAttr, FloatAttr>()) {
    UNWRAP_ERROR(result, reifyOneAttribute(attr, dtype));
    // If the result is an array or vector, splat the constant.
    ShapedType shapedType;
    if (auto simd = dyn_cast<SIMDType>(type)) {
      shapedType = VectorType::get(*simd.getResolvedSize(), result.getType());
    } else if (auto array = dyn_cast<POP::ArrayType>(type)) {
      shapedType =
          M::ArrayType::get(*array.getResolvedSize(), result.getType());
    }
    if (shapedType) {
      if (auto fpVal = dyn_cast<FloatAttr>(result)) {
        SmallVector<APFloat> values(shapedType.getNumElements(),
                                    fpVal.getValue());
        result = FloatArrayElementsAttr::get(shapedType, values);
      } else {
        SmallVector<APInt> values(shapedType.getNumElements(),
                                  result.cast<IntegerAttr>().getValue());
        result = IntArrayElementsAttr::get(shapedType, values);
      }
    }
    return result;
  }

  // If the value is an elements attribute, reify each element according to the
  // result dtype.
  if (auto fpValues = dyn_cast<FloatArrayElementsAttr>(attr)) {
    if (dtype.isInt() || dtype.isBool()) {
      return reifyArray<IntArrayElementsAttr>(
          fpValues,
          [&](const APFloat &val) { return reifyFloatToInt(val, dtype); },
          getEquivalentIntegerType(attr.getContext(), dtype));
    }

    FloatType fpType = getEquivalentFloatType(attr.getContext(), dtype);
    if (!fpType)
      return Error("unsupported floating point type: " + dtype.getAsString());
    return reifyArray<FloatArrayElementsAttr>(
        fpValues,
        [&](const APFloat &val) { return reifyFloatToFloat(val, fpType); },
        fpType);
  }

  auto intValues = attr.cast<IntArrayElementsAttr>();
  IntegerType intType = intValues.getElementType().cast<IntegerType>();
  if (dtype.isInt() || dtype.isBool()) {
    return reifyArray<IntArrayElementsAttr>(
        intValues,
        [&](const APInt &val) { return reifyIntToInt(val, intType, dtype); },
        getEquivalentIntegerType(attr.getContext(), dtype));
  }

  FloatType fpType = getEquivalentFloatType(attr.getContext(), dtype);
  if (!fpType)
    return Error("unsupported floating point type: " + dtype.getAsString());
  return reifyArray<FloatArrayElementsAttr>(
      intValues,
      [&](const APInt &val) {
        return reifyIntToFloat(val, intType, fpType, dtype);
      },
      fpType);
}

/// Verify a scalar or SIMD constant value.
static LogicalResult
verifyConstant(function_ref<InFlightDiagnostic(StringRef)> emitError,
               TypedAttr value, Type type) {
  // If the type is unresolved, allow only scalar constants or parameter
  // expressions..
  if (type.isa<ParamRefType>()) {
    if (!value.isa<IntegerAttr, FloatAttr, ParamDeclRefAttr,
                   ParamOperatorAttr>())
      return emitError(
          "expected integer or float attribute for unspecified result type");
    return success();
  }

  auto checkDType = [&](SIMDType type) -> LogicalResult {
    Type valueType = mlir::getElementTypeOrSelf(value);
    if (auto dtype = dyn_cast<DTypeConstantAttr>(type.getDType()))
      if (!dtype.isConvertibleFrom(valueType))
        return emitError("cannot convert from attribute type ")
               << valueType << " to dtype " << dtype.getDType().getAsString();
    return success();
  };

  // Verify array constant.
  if (auto array = dyn_cast<POP::ArrayType>(type)) {
    // If the size is known, require an elements attribute of the same shape.
    if (Optional<int64_t> size = array.getResolvedSize()) {
      auto elements = dyn_cast<ArrayElementsAttr>(value);
      if (!elements)
        return emitError("expected array elements attribute for array "
                         "constant with known size");
      auto type = dyn_cast<M::ArrayType>(elements.getType());
      if (!type || type.getSize() != *size)
        return emitError("expected attribute type to be !M.array<")
               << *size << "xT>";
    } else if (!value.isa<IntegerAttr, FloatAttr, ParamDeclRefAttr,
                          ParamOperatorAttr>()) {
      return emitError("expected integer or float attribute for array "
                       "constant of unspecified size");
    }
    // Only scalar arrays can be created.
    if (isSIMDSizeOneType(array.getResolvedElementType()))
      return checkDType(array.getResolvedElementType().cast<SIMDType>());
    return emitError("array constant must have scalar elements");
  }

  // Verify vector constant.
  auto simd = type.cast<SIMDType>();
  // If the attribute is scalar, we only need to check its dtype.
  if (value.isa<IntegerAttr, FloatAttr, ParamDeclRefAttr, ParamOperatorAttr>())
    return checkDType(simd);

  // The attribute is array, and its size needs to match the simd size.
  Optional<int64_t> size = simd.getResolvedSize();
  if (!size)
    return emitError("expected integer or float attribute for vector "
                     "constant of unspecified size");
  auto elements = dyn_cast<ArrayElementsAttr>(value);
  if (!elements)
    return emitError("expected array elements attribute for vector constant "
                     "with known size");
  auto vtype = dyn_cast<VectorType>(elements.getType());
  if (!vtype || vtype.getRank() != 1 || vtype.getShape().front() != *size)
    return emitError("expected attribute type to be vector<") << *size << "xT>";
  return checkDType(simd);
}

ErrorOrSuccess ConstantOp::finalizeElaboration() {
  auto simd = dyn_cast<SIMDType>(getType());
  if (!simd)
    simd = cast<SIMDType>(
        cast<POP::ArrayType>(getType()).getResolvedElementType());
  UNWRAP_ERROR(value,
               reifyConstant(getValue(), *simd.getResolvedDType(), getType()));
  setValueAttr(value);
  return success();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

LogicalResult ConstantOp::verify() {
  return verifyConstant([this](StringRef msg) { return emitOpError(msg); },
                        getValue(), getType());
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
  if (auto simd = dyn_cast<SIMDType>(type))
    return SIMDType::get(simd.getSize(), boolType);
  return nullptr;
}

LogicalResult CmpOp::inferReturnTypes(MLIRContext *ctx,
                                      std::optional<Location> loc,
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

  auto inputType = cast<SIMDType>(inputs.front());
  auto outputType = cast<SIMDType>(outputs.front());

  // First, check the input and output types must be of the same kind.
  // TODO: In theory we can support casting a scalar type to a vector type (e.g.
  // f64 to a 2xf32) or vice versa. We should support this when the use case
  // arises.
  Optional<KGENDType> inputDType = inputType.getResolvedDType();
  Optional<KGENDType> outputDType = outputType.getResolvedDType();

  // If neither dtype could be resolved, allow the cast.
  if (!inputDType || !outputDType)
    return true;

  ssize_t inputDTypeWidth = inputDType->getWidthInBits();
  ssize_t outputDTypeWidth = outputDType->getWidthInBits();

  // If we have a simd type, then the bitwidths must match.
  Optional<int64_t> inputSize = 1, outputSize = 1;
  if (auto inputSimd = dyn_cast<SIMDType>(inputType)) {
    auto outputSimd = outputType.cast<SIMDType>();
    inputSize = inputSimd.getResolvedSize();
    outputSize = outputSimd.getResolvedSize();
    // If neither size could be resolved, allow the cast.
    if (!inputSize || !outputSize)
      return true;
  }

  // If the sizes do not match, then we cannot cast.
  return inputDTypeWidth * *inputSize == outputDTypeWidth * *outputSize;
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

bool PointerBitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return inputs.front().isa<ParamRefType, PointerType, FunctionType>() &&
         outputs.front().isa<ParamRefType, PointerType, FunctionType>();
}

OpFoldResult PointerBitcastOp::fold(ArrayRef<Attribute> operands) {
  auto cast = getInput().getDefiningOp<PointerBitcastOp>();
  if (cast && cast.getInput().getType() == getType())
    return cast.getInput();
  return {};
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
  if (getInput().getType().getSize() != getOutput().getType().getSize())
    return emitOpError("cannot cast between SIMD types of different sizes");
  return success();
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDShuffleOp::verify() {
  Optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();
  if (static_cast<size_t>(*size) != getMask().size())
    return emitOpError("expected result to be a vector of ")
           << getMask().size() << " elements";

  auto lhsType = getLhs().getType().cast<SIMDType>();
  if (lhsType.getDType() != getType().getDType())
    return emitOpError("expected result dtype to match operand dtypes");

  if (Optional<int64_t> size = lhsType.getResolvedSize()) {
    for (int32_t index : getMask())
      if (index >= *size * 2)
        return emitOpError("mask element ") << index << " is out of bounds";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   Optional<unsigned> alignment) {
  build(b, state, ptr, alignment ? b.getIndexAttr(*alignment) : TypedAttr());
}

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   TypedAttr alignment) {
  auto type =
      ParamRefType::get(ptr.getType().cast<PointerType>().getElementType());
  build(b, state, type, ptr, alignment);
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(OpBuilder &b, OperationState &state, Value arg, Value ptr,
                    Optional<unsigned> alignment) {
  TypedAttr alignmentAttr;
  if (alignment)
    alignmentAttr = b.getIndexAttr(*alignment);
  build(b, state, arg, ptr, alignmentAttr);
}

//===----------------------------------------------------------------------===//
// StructGetOp
//===----------------------------------------------------------------------===//

/// Verify the value type matches the struct element type at the given index.
static LogicalResult verifyStructValueType(Operation *op, StructType container,
                                           IntegerAttr indexAttr,
                                           Type valueType,
                                           StringRef valueKind) {
  ArrayRef<TypedAttr> elementTypes = container.getElementTypes();
  size_t index = indexAttr.getInt();
  if (index >= elementTypes.size())
    return op->emitOpError("element index ")
           << index << " out of bounds (>=" << elementTypes.size() << ")";
  if (ParamRefType::get(elementTypes[index]) != valueType)
    return op->emitOpError(valueKind)
           << " type " << valueType
           << " does not match struct element type at index " << index << ": "
           << elementTypes[index];
  return success();
}

LogicalResult StructGetOp::verify() {
  return verifyStructValueType(*this, getContainer().getType(), getIndexAttr(),
                               getType(), "result");
}

template <typename OpT>
static FailureOr<TypedAttr>
inferStructElementType(function_ref<LogicalResult(const Twine &)> emitError,
                       StructType structType, DictionaryAttr attrs) {
  if (!structType)
    return emitError("expected struct operand");
  mlir::OperationName name(OpT::getOperationName(), attrs.getContext());
  auto indexAttr =
      dyn_cast_if_present<IntegerAttr>(attrs.get(OpT::getIndexAttrName(name)));
  if (!indexAttr)
    return emitError("expected an integer index attribute");
  size_t index = indexAttr.getInt();
  if (index >= structType.getNumElements())
    return emitError("struct element index out of bounds");
  return structType.getElementTypes()[index];
}

LogicalResult StructGetOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &types) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto structType = dyn_cast<StructType>(operands.front().getType());
  FailureOr<TypedAttr> type =
      inferStructElementType<StructGetOp>(emitError, structType, attrs);
  if (succeeded(type))
    types.push_back(ParamRefType::get(*type));
  return type;
}

void StructGetOp::build(OpBuilder &b, OperationState &state, Value container,
                        int64_t index) {
  return build(b, state, container, b.getIndexAttr(index));
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructValueType(AsmParser &p, Type &valueType,
                                        Type structType, IntegerAttr index) {
  ArrayRef<TypedAttr> elementTypes =
      structType.cast<StructType>().getElementTypes();
  if (index.getInt() > static_cast<int64_t>(elementTypes.size()))
    return p.emitError(p.getCurrentLocation(), "element index out of bounds (")
           << index.getInt() << " >= " << elementTypes.size() << ")";
  // Infer the value type from the struct type and index.
  valueType = ParamRefType::get(elementTypes[index.getInt()]);
  return success();
}

static void printStructValueType(AsmPrinter &p, Operation *op, Type valueType,
                                 Type structType, IntegerAttr index) {}

LogicalResult StructReplaceOp::verify() {
  return verifyStructValueType(*this, getContainer().getType(), getIndexAttr(),
                               getValue().getType(), "operand");
}

void StructReplaceOp::build(OpBuilder &b, OperationState &state, Value value,
                            Value container, int64_t index) {
  build(b, state, value, container, b.getIndexAttr(index));
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult StructGEPOp::verify() {
  return verifyStructValueType(
      *this,
      cast<StructType>(getContainer().getType().getResolvedElementType()),
      getIndexAttr(), ParamRefType::get(getType().getElementType()), "result");
}

LogicalResult StructGEPOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &types) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto pointerType = dyn_cast<PointerType>(operands.front().getType());
  if (!pointerType)
    return emitError("expected pointer operand");
  auto structType = dyn_cast<StructType>(pointerType.getResolvedElementType());
  FailureOr<TypedAttr> type =
      inferStructElementType<StructGetOp>(emitError, structType, attrs);
  if (succeeded(type))
    types.push_back(PointerType::get(*type));
  return type;
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

/// This is used by the `ArrayElementType` constraint to match a type range
/// against a single type.
static bool typeRangeMatches(Type type, TypeRange range) {
  return llvm::all_of(range, [&](Type e) { return type == e; });
}

LogicalResult ArrayCreateOp::verify() {
  int64_t size = *getType().getResolvedSize();
  if (size != getNumOperands())
    return emitOpError("expected ")
           << size << " operands to create array but got " << getNumOperands();
  return success();
}

void ArrayCreateOp::build(OpBuilder &b, OperationState &state,
                          ValueRange elements) {
  return build(b, state, ArrayType::get(elements), elements);
}

//===----------------------------------------------------------------------===//
// ArrayRepeatOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayRepeatOp::verify() {
  Optional<int64_t> size = getType().getResolvedSize();
  if (size && *size != 0 && getNumOperands() == 0)
    return emitOpError("requires at least one operand to create an array whose "
                       "size is non-zero");
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

// If the array has a concrete size, do a bounds check.
static LogicalResult verifyArrayIndex(Operation *op, TypedAttr indexExpr,
                                      POP::ArrayType arrayType) {
  Optional<int64_t> size = arrayType.getResolvedSize();
  auto indexAttr = dyn_cast<IntegerAttr>(indexExpr);
  if (!size || !indexAttr)
    return success();

  int64_t index = indexAttr.getInt();
  if (index < 0 || index >= *size)
    return op->emitOpError("array index out of bounds: ") << index;
  return success();
}

void ArrayGetOp::build(OpBuilder &b, OperationState &state, Value array,
                       int64_t index) {
  return build(
      b, state,
      ParamRefType::get(array.getType().cast<ArrayType>().getElementType()),
      array, b.getIndexAttr(index));
}

LogicalResult ArrayGetOp::verify() {
  return verifyArrayIndex(*this, getIndex(), getArray().getType());
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayReplaceOp::verify() {
  return verifyArrayIndex(*this, getIndex(), getArray().getType());
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

static Type getPointerToArrayElementType(Type arrayPtr) {
  return PointerType::get(
      cast<POP::ArrayType>(cast<PointerType>(arrayPtr).getResolvedElementType())
          .getElementType());
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

/// Verify that the type is one of the variant types.
static LogicalResult verifyVariantElementType(Operation *op, Type type,
                                              VariantType variantType) {
  if (!variantType.getTypeIndex(type))
    return op->emitOpError("operand type ")
           << type << " is not a variant type of " << variantType;
  return success();
}

LogicalResult VariantCreateOp::verify() {
  return verifyVariantElementType(*this, getOperand().getType(), getType());
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

LogicalResult VariantIsOp::verify() {
  return verifyVariantElementType(*this, getTestType(), getVariant().getType());
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

LogicalResult VariantGetOp::verify() {
  return verifyVariantElementType(*this, getType(), getVariant().getType());
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

/// Parse the element type of the allocated pointer type.
static ParseResult parsePointerOf(AsmParser &p, Type &result) {
  FailureOr<TypedAttr> elementType;
  if (parseTypeParamValue(p, elementType))
    return failure();
  result = PointerType::get(*elementType);
  return success();
}

/// Print the element type of the allocated pointer type.
static void printPointerOf(AsmPrinter &p, Operation *op, Type result) {
  printTypeParamValue(p, result.cast<PointerType>().getElementType());
}

void StackAllocationOp::build(OpBuilder &b, OperationState &state, Type result,
                              TypedAttr count) {
  build(b, state, result, count, TypedAttr());
}

void StackAllocationOp::build(OpBuilder &b, OperationState &state, Type result,
                              int64_t count) {
  auto countAttr = b.getIndexAttr(count);
  build(b, state, result, countAttr);
}

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess GlobalConstantOp::finalizeElaboration() {
  Type type = getType().getResolvedElementType();
  auto simd = dyn_cast<SIMDType>(type);
  if (!simd)
    simd = cast<SIMDType>(cast<POP::ArrayType>(type).getResolvedElementType());
  UNWRAP_ERROR(value,
               reifyConstant(getValue(), *simd.getResolvedDType(), type));
  setValueAttr(value);
  return success();
}

LogicalResult GlobalConstantOp::verify() {
  return verifyConstant([this](StringRef msg) { return emitOpError(msg); },
                        getValue(),
                        ParamRefType::get(getType().getElementType()));
}

//===----------------------------------------------------------------------===//
// IndexToPointerOp
//===----------------------------------------------------------------------===//

/// Checks the the input pointer type is catabolic to the output address type.
static bool isPointerToAddressCastCompatible(TypeRange inputs,
                                             TypeRange outputs) {
  if (inputs.size() != 1 || inputs.size() != outputs.size())
    return false;
  Type pointerType = inputs.front();
  Type addressType = outputs.front();

  // The input and output must be either both scalar or both SIMD. And so,
  // implement the DTypeInterface.
  auto pointerDType = dyn_cast<SIMDType>(pointerType);
  auto addressDType = dyn_cast<SIMDType>(addressType);

  // If the address type does not implement the dtype interface, then the lhs
  // must be a pointer or address type and the rhs must be an index type.
  if (!addressDType) {
    if (!isa<IndexType>(addressType))
      return false;
    if (pointerType.isa<PointerType>())
      return true;
    // If the pointer type is unresolved, then we are ok.
    if (!pointerDType || !pointerDType.getResolvedDType())
      return true;
    // If the pointer type is of the form !pop.simd<1, address> then that's
    // ok.
    if (pointerDType.getResolvedDType()->isAddress() &&
        pointerType.cast<SIMDType>().getResolvedSize().value_or(0) == 1)
      return true;
  }

  // If the pointer type is unresolved, then we we are ok.
  if (!pointerDType)
    return false;

  auto isUnboundOrIndexDType = [](SIMDType type) {
    Optional<KGENDType> dtype = type.getResolvedDType();
    return !dtype || dtype->isIndex();
  };
  auto isUnboundOrAddressDType = [](SIMDType type) {
    Optional<KGENDType> dtype = type.getResolvedDType();
    return !dtype || dtype->isAddress();
  };

  // Otherwise, the lhs type must be an address dtype and the rhs type must be
  // an index dtype.
  if (!isUnboundOrAddressDType(pointerDType) ||
      !isUnboundOrIndexDType(addressDType))
    return false;

  // Finally, the simd width must match.
  return pointerType.cast<SIMDType>().getResolvedSize() ==
         addressType.cast<SIMDType>().getResolvedSize();
}

bool IndexToPointerOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return isPointerToAddressCastCompatible(outputs, inputs);
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

bool PointerToIndexOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return isPointerToAddressCastCompatible(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// VariantVisitOp
//===----------------------------------------------------------------------===//

static ParseResult
parseVariantVisitRegions(OpAsmParser &p, TypeArrayAttr &cases,
                         SmallVectorImpl<std::unique_ptr<Region>> &regions) {
  SmallVector<Type> caseTypes;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    OpAsmParser::Argument arg;
    if (p.parseLParen() || p.parseArgument(arg, /*allowType=*/true) ||
        p.parseRParen() ||
        p.parseRegion(*regions.emplace_back(std::make_unique<Region>()), arg))
      return failure();
    caseTypes.push_back(arg.type);
  }
  cases = TypeArrayAttr::get(p.getContext(), caseTypes);

  if (succeeded(p.parseOptionalKeyword("default")))
    if (p.parseRegion(*regions.emplace_back(std::make_unique<Region>())))
      return failure();

  return success();
}

static void printVariantVisitRegions(OpAsmPrinter &p, Operation *op,
                                     TypeArrayAttr cases, RegionRange regions) {
  for (auto [caseType, region] : llvm::zip(cases, regions)) {
    p.printNewline();
    p << "case (";
    p.printRegionArgument(region->getArgument(0));
    p << ") ";
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
  if (cases.size() == regions.size())
    return;
  p.printNewline();
  p << "default ";
  p.printRegion(*regions.back());
}

LogicalResult VariantVisitOp::verify() {
  SmallPtrSet<Type, 4> typeSet, seenTypes;
  VariantType variant = getVariant().getType();
  for (Type type : variant.getParameterizedElementTypes())
    typeSet.insert(type);
  for (Type caseType : getCases()) {
    if (!typeSet.contains(caseType))
      return emitOpError("type case ")
             << caseType << " is not a possible variant type of " << variant;
    if (!seenTypes.insert(caseType).second)
      return emitOpError("duplicate type case ") << caseType;
  }
  if (seenTypes.size() == variant.getTypes().size()) {
    if (getNumRegions() != seenTypes.size())
      return emitOpError("expected ")
             << seenTypes.size() << " regions when all type cases are present";
  } else {
    if (getNumRegions() != seenTypes.size() + 1) {
      return emitOpError("expected ") << seenTypes.size()
                                      << " regions plus a default region when "
                                         "not all case types are present";
    }
    if (getRegions().back().getNumArguments())
      return emitOpError("expected default region to have zero arguments");
  }
  for (Region &region : getRegions()) {
    Operation *terminator = region.front().getTerminator();
    auto yield = dyn_cast<YieldOp>(terminator);
    if (!yield) {
      return (emitOpError("region #")
              << region.getRegionNumber() << " expected `pop.yield` terminator")
                 .attachNote(terminator->getLoc())
             << "see invalid terminator here";
    }
    if (yield.getOperandTypes() != getResultTypes()) {
      return (emitOpError("operand types of region #")
              << region.getRegionNumber() << " yield do not match result types")
                 .attachNote(yield.getLoc())
             << "see terminator here";
    }
  }
  for (auto [type, region] : llvm::zip(getCases(), getRegions())) {
    if (region.getNumArguments() != 1)
      return emitOpError("expected region #")
             << region.getRegionNumber() << " to have one argument";
    if (region.getArgumentTypes().front() != type)
      return emitOpError("expected region #")
             << region.getRegionNumber() << " argument type to be " << type;
  }
  return success();
}

bool VariantVisitOp::hasDefaultRegion() {
  return getCases().size() != getNumRegions();
}

Region *VariantVisitOp::getDefaultRegion() {
  assert(hasDefaultRegion());
  return &getRegions().back();
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface implementation

bool VariantVisitOp::areTypesCompatible(Type lhs, Type rhs) {
  if (lhs == rhs)
    return true;

  // The variant operand maps to the case value.
  if (auto variant = dyn_cast<VariantType>(lhs))
    return variant.getTypeIndex(rhs).has_value();

  return false;
}

void VariantVisitOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &successors) {
  // All regions branch back to the parent op.
  if (index) {
    successors.emplace_back(getResults());
    return;
  }

  // The known variant type of the operand can be used to narrow the successor
  // regions of the parent op to just one, but we can't do that here.
  for (Region &region : getRegions())
    successors.emplace_back(&region, region.getArguments());
}

OperandRange
VariantVisitOp::getSuccessorEntryOperands(std::optional<unsigned> index) {
  assert(index);
  if (hasDefaultRegion() && *index == getNumRegions() - 1)
    return {(*this)->operand_end(), (*this)->operand_end()};
  return (*this)->getOperands();
}

/// Each region is invoked at most once per op.
void VariantVisitOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::InvocationBounds> &bounds) {
  bounds.append(getNumRegions(), mlir::InvocationBounds(/*lb=*/0, /*ub=*/1));
}

//===----------------------------------------------------------------------===//
// ListGetOp
//===----------------------------------------------------------------------===//

LogicalResult ListGetOp::verify() {
  auto index = dyn_cast<IntegerAttr>(getIndex());
  Optional<int64_t> length = getList().getType().getResolvedLength();
  if (!index || !length)
    return success();
  if (index.getInt() < 0 || index.getInt() >= *length)
    return emitOpError("list index out-of-range");
  return success();
}

//===----------------------------------------------------------------------===//
// ListCreateOp
//===----------------------------------------------------------------------===//

LogicalResult ListCreateOp::verify() {
  if (getResult().getType().getLength() !=
      Builder(getContext()).getIndexAttr(getNumOperands()))
    return emitOpError("expected result list to have ")
           << getNumOperands() << "elements";
  return success();
}

//===----------------------------------------------------------------------===//
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Verify the conversion between the higher-level type and lower-level type.
static LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     Type popType, Type builtinType) {
  auto simd = dyn_cast<SIMDType>(popType);
  if (!simd)
    return emitError("cannot convert type ") << popType;
  // Verify the SIMD size matches the vector size and the dtypes match.

  auto size = simd.getResolvedSize();
  if (size && *size == 1) {
    // Scalar case
    auto vector = dyn_cast<VectorType>(builtinType);
    if (vector) {
      builtinType = vector.getElementType();
      return verifyConversionCast(emitError, popType, builtinType);
    }
    auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
    if (dtype && !dtype.isConvertibleTo(builtinType))
      return emitError("cannot convert from scalar dtype ")
             << dtype.getDType().getAsString() << " to " << builtinType;
    return success();
  }

  auto vector = dyn_cast<VectorType>(builtinType);
  if (!vector || vector.getRank() != 1 || vector.getNumScalableDims() != 0)
    return emitError("expected a rank 1 non-scalable vector");

  if (size && *size != vector.getShape().front())
    return emitError("expected vector<") << *size << "xT>";

  if (auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
      dtype && !dtype.isConvertibleTo(vector.getElementType()))
    return emitError("cannot convert from SIMD dtype ")
           << dtype.getDType().getAsString() << " to vector element "
           << vector.getElementType();
  return success();
}

LogicalResult CastToBuiltinOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getInput().getType(),
      getType());
}

OpFoldResult CastToBuiltinOp::fold(ArrayRef<Attribute> operands) {
  // Fold A->B->A cast.
  if (auto parent = getInput().getDefiningOp<CastFromBuiltinOp>();
      parent && parent.getInput().getType() == getType())
    return parent.getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

LogicalResult CastFromBuiltinOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getType(),
      getInput().getType());
}

OpFoldResult CastFromBuiltinOp::fold(ArrayRef<Attribute> operands) {
  // Fold A->B->A cast.
  if (auto parent = getInput().getDefiningOp<CastToBuiltinOp>();
      parent && parent.getInput().getType() == getType())
    return parent.getInput();
  return {};
}
//===----------------------------------------------------------------------===//
// PartialApplyOp
//===----------------------------------------------------------------------===//

static Type computePartialApplyResultType(std::optional<Location> loc,
                                          FunctionType callee,
                                          ValueRange inputs,
                                          ArrayRef<int64_t> boundInputs) {
  auto emitError = [&](const Twine &msg) -> Type {
    (void)mlir::emitOptionalError(loc, "'pop.partial_apply' op " + msg);
    return {};
  };
  // Ensure the indices are sorted.
  if (!llvm::is_sorted(boundInputs))
    return emitError("expected indices to be sorted ascending");
  if (boundInputs.size() != inputs.size())
    return emitError("mismatch between number of indices and inputs: " +
                     Twine(boundInputs.size()) + " vs " + Twine(inputs.size()));

  DenseSet<int64_t> seenInputs;
  seenInputs.reserve(boundInputs.size());
  ArrayRef<Type> argumentTypes = callee.getInputs();
  SmallVector<Type> newInputTypes;
  SmallVector<ValueInputConvention> newInputConventions;
  unsigned lastIdx = 0;
  for (auto [input, index] : llvm::zip(inputs, boundInputs)) {
    if (index >= static_cast<int64_t>(argumentTypes.size()))
      return emitError("bound input index is out of range: " + Twine(index));
    if (!seenInputs.insert(index).second)
      return emitError("duplicate bound input index: " + Twine(index));
    if (input.getType() != argumentTypes[index])
      return emitError("input bound to argument #" + Twine(index) +
                       " is incorrect");
    // Pick the types of arguments that aren't bound.
    while (lastIdx++ < index)
      newInputTypes.push_back(argumentTypes[lastIdx - 1]);
  }
  for (; lastIdx < argumentTypes.size(); ++lastIdx)
    newInputTypes.push_back(argumentTypes[lastIdx]);

  assert(newInputTypes.size() == argumentTypes.size() - boundInputs.size());

  MLIRContext *context = callee.getContext();
  auto resultFnType =
      FunctionType::get(context, newInputTypes, callee.getResults());

  return ClosureType::get(context, resultFnType);
}

LogicalResult PartialApplyOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &types) {
  mlir::OperationName name(getOperationName(), attrs.getContext());
  auto boundInputs =
      attrs.getAs<mlir::DenseI64ArrayAttr>(getBoundInputsAttrName(name));
  if (!boundInputs || operands.empty() ||
      (!isa<FunctionType>(operands[0].getType()) &&
       !isa<ClosureType>(operands[0].getType())))
    return mlir::emitOptionalError(loc, "missing required attributes");

  if (auto closure = dyn_cast<ClosureType>(operands[0].getType())) {
    types.push_back(computePartialApplyResultType(
        loc, closure.getFunc(), operands.drop_front(), boundInputs));
  } else {
    types.push_back(computePartialApplyResultType(
        loc, cast<FunctionType>(operands[0].getType()), operands.drop_front(),
        boundInputs));
  }
  return success(types.back() != Type());
}

/// Verify the operation is well-formed. It is not possible to get an
/// ill-formed operation using the pretty syntax, but it is possible from C++.
LogicalResult PartialApplyOp::verify() {
  Type resultType;
  if (auto closureType = dyn_cast<ClosureType>(getCallee().getType()))
    resultType = computePartialApplyResultType(getLoc(), closureType.getFunc(),
                                               getInputs(), getBoundInputs());
  else
    resultType = computePartialApplyResultType(
        getLoc(), cast<FunctionType>(getCallee().getType()), getInputs(),
        getBoundInputs());

  if (!resultType)
    return failure();
  if (resultType != getType())
    return emitOpError("result signature does not match");
  return success();
}

/// Canonicalize `partial_apply(partial_apply))` by folding the bound operands
/// into the same operation.
LogicalResult PartialApplyOp::canonicalize(PartialApplyOp op,
                                           PatternRewriter &rewriter) {
  auto bind = dyn_cast_or_null<PartialApplyOp>(op.getCallee().getDefiningOp());
  if (!bind)
    return failure();
  // Merge the values and indices together.
  SmallVector<Value> newInputs;
  SmallVector<int64_t> newIndices;
  size_t totalInputs = op.getInputs().size() + bind.getInputs().size();
  newInputs.reserve(totalInputs);
  newIndices.reserve(totalInputs);
  auto lhsRange = llvm::zip(op.getInputs(), op.getBoundInputs());
  auto rhsRange = llvm::zip(bind.getInputs(), bind.getBoundInputs());
  auto lhs = lhsRange.begin(), rhs = rhsRange.begin(), lhsEnd = lhsRange.end(),
       rhsEnd = rhsRange.end();
  while (lhs != lhsEnd && rhs != rhsEnd) {
    auto [lhsInput, lhsIndex] = *lhs;
    auto [rhsInput, rhsIndex] = *rhs;
    if (lhsIndex < rhsIndex) {
      ++lhs;
      newInputs.push_back(lhsInput);
      newIndices.push_back(lhsIndex);
    } else {
      ++rhs;
      newInputs.push_back(rhsInput);
      newIndices.push_back(rhsIndex);
    }
  }
  auto pushTheRest = [&](auto it, auto end) {
    for (; it != end; ++it) {
      auto [input, index] = *it;
      newInputs.push_back(input);
      newIndices.push_back(index);
    }
  };
  pushTheRest(lhs, lhsEnd);
  pushTheRest(rhs, rhsEnd);
  rewriter.replaceOpWithNewOp<PartialApplyOp>(
      op, op.getType(), bind.getCallee(), newInputs, newIndices);
  return success();
}

/// Parse the input operands, using `?` to represent a placeholder value.
static ParseResult parseBoundInputs(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
    mlir::DenseI64ArrayAttr &boundInputs, SmallVectorImpl<Type> &inputTypes,
    Type &resultType, Type &calleeType) {
  // Parse the binding list `(` ((`?` | operand) (`,` (`?` | operand))*)? `)`.
  SmallVector<int64_t> boundInputIndices;
  if (p.parseLParen())
    return failure();
  if (p.parseOptionalRParen()) {
    int64_t index = 0;
    OpAsmParser::UnresolvedOperand input;
    auto parseElt = [&]() -> ParseResult {
      llvm::SMLoc loc = p.getCurrentLocation();
      if (p.parseOptionalQuestion()) {
        mlir::OptionalParseResult result = p.parseOptionalOperand(input);
        if (result.has_value() && failed(*result))
          return failure();
        if (!result.has_value())
          return p.emitError(loc, "expected '?' or an operand in binding list");
        inputs.push_back(input);
        boundInputIndices.push_back(index);
      }
      ++index;
      return success();
    };
    if (p.parseCommaSeparatedList(parseElt) || p.parseRParen())
      return failure();
  }

  // Parse the input function or closure type `:` function-type || closure-type.
  llvm::SMLoc loc = p.getCurrentLocation();
  Type type;
  if (p.parseColonType(type))
    return failure();
  FunctionType funcType;
  if (auto closureType = dyn_cast<ClosureType>(type)) {
    funcType = closureType.getFunc();
    calleeType = closureType;
  } else if (auto ft = dyn_cast<FunctionType>(type)) {
    funcType = ft;
    calleeType = funcType;
  } else {
    return p.emitError(
        loc, "expected callee type to be a function type or closure type.");
  }
  boundInputs = p.getBuilder().getDenseI64ArrayAttr(boundInputIndices);

  // Infer the input types from the function type.
  SmallVector<Type> resultTypes;
  int64_t lastIdx = 0;
  int64_t numInputs = funcType.getNumInputs();
  for (int64_t index : boundInputIndices) {
    if (index >= numInputs)
      return p.emitError(loc, "there are more bound inputs than arguments");
    inputTypes.push_back(funcType.getInputs()[index]);
    while (lastIdx++ < index)
      resultTypes.push_back(funcType.getInputs()[lastIdx - 1]);
  }
  for (; lastIdx < numInputs; ++lastIdx)
    resultTypes.push_back(funcType.getInputs()[lastIdx]);

  // Infer the result signature type.
  resultType = ClosureType::get(
      p.getContext(),
      FunctionType::get(p.getContext(), resultTypes, funcType.getResults()));
  return success();
}

static void printBoundInputs(OpAsmPrinter &p, Operation *op, ValueRange inputs,
                             mlir::DenseI64ArrayAttr boundInputs,
                             TypeRange inputTypes, Type resultType,
                             Type calleeType) {
  int64_t numInputs = 0;
  if (auto closure = dyn_cast<ClosureType>(calleeType))
    numInputs = closure.getFunc().getNumInputs();
  else
    numInputs = cast<FunctionType>(calleeType).getNumInputs();

  p << '(';
  auto idxIt = boundInputs.asArrayRef().begin();
  int64_t index = 0;
  auto eachFn = [&](int64_t i) {
    if (idxIt == boundInputs.asArrayRef().end() || i < *idxIt) {
      p << '?';
    } else {
      ++idxIt;
      p << inputs[index++];
    }
  };
  llvm::interleaveComma(llvm::seq<int64_t>(0, numInputs), p, eachFn);
  p << ") : " << calleeType;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

/// Infer the input and result types from the callee type
static ParseResult parseCallIndirectCalleeAndInputResultTypes(
    AsmParser &p, SmallVectorImpl<Type> &inputTypes,
    SmallVectorImpl<Type> &resultTypes, Type &calleeType) {
  auto loc = p.getCurrentLocation();

  if (p.parseColonType(calleeType))
    return failure();

  FunctionType calleeFuncType;

  if (auto closureType = dyn_cast<ClosureType>(calleeType))
    calleeFuncType = closureType.getFunc();
  else if (auto functionType = dyn_cast<FunctionType>(calleeType))
    calleeFuncType = functionType;
  else
    return p.emitError(
        loc, "the callee type must be a function type or a closure type.");

  for (unsigned i = 0, e = calleeFuncType.getNumInputs(); i < e; i++)
    inputTypes.emplace_back(calleeFuncType.getInput(i));

  for (unsigned i = 0, e = calleeFuncType.getNumResults(); i < e; i++)
    resultTypes.emplace_back(calleeFuncType.getResult(i));

  return success();
}

static void printCallIndirectCalleeAndInputResultTypes(AsmPrinter &p,
                                                       Operation *op,
                                                       TypeRange inputTypes,
                                                       TypeRange resultTypes,
                                                       Type calleeType) {
  p << " : ";
  p.printType(calleeType);
}

/// Canonicalize `call_indirect(partial_apply) -> call_indirect` by folding the
/// bound arguments into the call, and canonicalize `call_indirect(constant)`
/// into `call_param`.
LogicalResult CallIndirectOp::canonicalize(CallIndirectOp op,
                                           PatternRewriter &rewriter) {
  Operation *calleeOp = op.getCallee().getDefiningOp();

  if (auto bind = dyn_cast_or_null<PartialApplyOp>(calleeOp)) {
    SmallVector<Value> newInputs;
    int64_t totalInputs = op.getInputs().size() + bind.getInputs().size();
    newInputs.reserve(totalInputs);
    auto boundIt = bind.getBoundInputs().begin();
    auto curInputsIt = op.getInputs().begin();
    auto boundInputsIt = bind.getInputs().begin();
    for (int64_t i = 0; i < totalInputs; ++i) {
      if (boundIt == bind.getBoundInputs().end() || i < *boundIt) {
        newInputs.push_back(*curInputsIt++);
      } else {
        ++boundIt;
        newInputs.push_back(*boundInputsIt++);
      }
    }
    rewriter.replaceOpWithNewOp<CallIndirectOp>(op, op.getResultTypes(),
                                                bind.getCallee(), newInputs);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
