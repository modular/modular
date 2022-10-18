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
                                     DType dtype) {
  // Integer to integer conversion. Check that this isn't converting between
  // signed or unsigned integers.
  if (!type.isSignlessInteger() && type.isSignedInteger() != dtype.isSInt()) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    os << "cannot change signfulness when converting from " << type << " to "
       << dtype.getAsString();
    return Error(std::move(os.str()));
  }

  // Truncate or extend the value depending on the result width.
  APSInt origInt(value, dtype.isUInt());
  APSInt intValue = origInt.extOrTrunc(dtype.getWidthInBits());
  if (intValue.extOrTrunc(origInt.getBitWidth()) != origInt)
    return Error("integer constant does not fit into " + dtype.getAsString());

  return intValue;
}

/// Reify an integer to a float.
static ErrorOr<APFloat> reifyIntToFloat(const APInt &value, IntegerType type,
                                        FloatType fpType, DType dtype) {
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
  // Float to integer conversion. Only exact integers can be converted.
  if (!value.isInteger())
    return Error("only exact integer floats can be converted to integers");

  // Convert the float to an integer.
  APSInt apInt(dtype.getWidthInBits(), dtype.isUInt());
  bool exact;
  value.convertToInteger(apInt, APFloat::rmTowardZero, &exact);
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

/// Reify a single integer or float attribute to an attribute that fits the
/// given dtype.
static ErrorOr<TypedAttr> reifyOneAttribute(Attribute attr, DType dtype) {
  if (auto value = dyn_cast<IntegerAttr>(attr)) {
    auto type = value.getType().cast<IntegerType>();

    if (!dtype.isInt() && !dtype.isFloat())
      return Error("cannot coerce constant value to " + dtype.getAsString());

    if (dtype.isInt()) {
      ErrorOr<APSInt> intValue = reifyIntToInt(value.getValue(), type, dtype);
      if (intValue.isError())
        return intValue.takeError();
      return IntegerAttr::get(attr.getContext(), intValue.takeValue());
    }

    // Integer to float conversion. Check for a valid floating point type.
    FloatType fpType = getEquivalentFloatType(type.getContext(), dtype);
    if (!fpType)
      return Error("unsupported floating point type: " + dtype.getAsString());
    ErrorOr<APFloat> apFp =
        reifyIntToFloat(value.getValue(), type, fpType, dtype);
    if (apFp.isError())
      return apFp.takeError();
    return FloatAttr::get(fpType, apFp.takeValue());
  }

  auto value = attr.cast<FloatAttr>();
  if (dtype.isInt()) {
    ErrorOr<APSInt> apInt = reifyFloatToInt(value.getValue(), dtype);
    if (apInt.isError())
      return apInt.takeError();
    return IntegerAttr::get(attr.getContext(), apInt.takeValue());
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
    auto newValue = convert(value);
    if (newValue.isError())
      return newValue.takeError();
    values.push_back(newValue.takeValue());
  }
  ShapedType newShapedType = attr.getType().cloneWith({}, newElementType);
  return OutAttrT::get(newShapedType, values);
}

/// Reify a primitive constant attribute (integer, float, or vector thereof)
/// to an attribute that fits the given type.
static ErrorOr<TypedAttr> reifyConstant(TypedAttr attr, DType dtype,
                                        Type type) {
  // If the value is an integer or float attribute, reify it to according to the
  // result dtype.
  if (attr.isa<IntegerAttr, FloatAttr>()) {
    ErrorOr<TypedAttr> result = reifyOneAttribute(attr, dtype);
    if (result.isError())
      return result.takeError();
    // If the result is an array or vector, splat the constant.
    ShapedType shapedType;
    if (auto simd = dyn_cast<SIMDType>(type))
      shapedType = VectorType::get(*simd.getResolvedSize(), result->getType());
    else if (auto array = type.dyn_cast<POP::ArrayType>())
      shapedType =
          M::ArrayType::get(*array.getResolvedSize(), result->getType());
    if (shapedType) {
      if (auto fpVal = dyn_cast<FloatAttr>(result.get())) {
        SmallVector<APFloat> values(shapedType.getNumElements(),
                                    fpVal.getValue());
        result = FloatArrayElementsAttr::get(shapedType, values);
      } else {
        SmallVector<APInt> values(shapedType.getNumElements(),
                                  result->cast<IntegerAttr>().getValue());
        result = IntArrayElementsAttr::get(shapedType, values);
      }
    }
    return result;
  }

  // If the value is an elements attribute, reify each element according to the
  // result dtype.
  if (auto fpValues = dyn_cast<FloatArrayElementsAttr>(attr)) {
    if (dtype.isInt()) {
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
  if (dtype.isInt()) {
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
  // If the type is unresolved, allow only scalar constants.
  if (type.isa<ParamRefType>()) {
    if (!value.isa<IntegerAttr, FloatAttr>())
      return emitError(
          "expected integer or float attribute for unspecified result type");
    return success();
  }

  auto checkDType = [&](DTypeInterface type) -> LogicalResult {
    Type valueType = mlir::getElementTypeOrSelf(value);
    if (auto dtype = dyn_cast<DTypeConstantAttr>(type.getDType()))
      if (!dtype.isConvertibleFrom(valueType))
        return emitError("cannot convert from attribute type ")
               << valueType << " to dtype " << dtype.getDType().getAsString();
    return success();
  };

  // If the type is a scalar, allow only scalar constant attributes.
  if (auto scalar = dyn_cast<ScalarType>(type)) {
    if (!value.isa<IntegerAttr, FloatAttr>())
      return emitError("scalar constant expected integer or float "
                       "attribute for constant value");
    // If the dtype is specified, ensure it matches the attribute type.
    return checkDType(scalar.cast<DTypeInterface>());
  }

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
    } else if (!value.isa<IntegerAttr, FloatAttr>()) {
      return emitError("expected integer or float attribute for array "
                       "constant of unspecified size");
    }
    // Only scalar arrays can be created.
    auto scalar =
        dyn_cast_if_present<ScalarType>(array.getResolvedElementType());
    if (!scalar)
      return emitError("array constant must have scalar elements");
    return checkDType(scalar.cast<DTypeInterface>());
  }

  // Verify vector constant.
  auto simd = type.cast<SIMDType>();
  // If the size is specified, require an attribute of the same shape.
  if (Optional<int64_t> size = simd.getResolvedSize()) {
    auto elements = dyn_cast<ArrayElementsAttr>(value);
    if (!elements)
      return emitError("expected array elements attribute for vector "
                       "constant with known size");
    auto type = dyn_cast<VectorType>(elements.getType());
    if (!type || type.getRank() != 1 || type.getShape().front() != *size)
      return emitError("expected attribute type to be vector<")
             << *size << "xT>";
  } else if (!value.isa<IntegerAttr, FloatAttr>()) {
    return emitError("expected integer or float attribute for vector "
                     "constant of unspecified size");
  }
  // If the dtype is specified, ensure it matches the attribute type.
  return checkDType(simd.cast<DTypeInterface>());
}

ErrorOrSuccess ConstantOp::finalizeElaboration() {
  ErrorOr<TypedAttr> value = reifyConstant(
      getValue(), *getType().cast<DTypeInterface>().getResolvedDType(),
      getType());
  if (value.isError())
    return value.takeError();
  setValueAttr(value.takeValue());
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

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  auto attr = dyn_cast<TypedAttr>(value);
  if (!attr)
    return false;
  // Call the verify function without emitting any errors.
  return succeeded(verifyConstant(
      [](StringRef msg) {
        InFlightDiagnostic diag;
        diag.abandon();
        return diag;
      },
      attr, type));
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
  if (auto simd = dyn_cast<SIMDType>(type))
    return SIMDType::get(simd.getSize(), boolType);
  return nullptr;
}

LogicalResult CmpOp::inferReturnTypes(MLIRContext *ctx, Optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::RegionRange regions,
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

  // The input and output must be either both scalar or both SIMD. And so,
  // implement the DTypeInterface.
  auto inputType = inputs.front().cast<DTypeInterface>();
  auto outputType = outputs.front().cast<DTypeInterface>();

  // First, check the input and output types must be of the same kind.
  // TODO: In theory we can support casting a scalar type to a vector type (e.g.
  // f64 to a 2xf32) or vice versa. We should support this when the use case
  // arises.
  if (inputType.isa<ScalarType>() != outputType.isa<ScalarType>())
    return false;

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
  return inputDTypeWidth * inputSize.value() ==
         outputDTypeWidth * outputSize.value();
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

bool PointerBitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return inputs.front().isa<PointerType>() &&
         outputs.front().isa<PointerType>();
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

  if (auto inputSimd = dyn_cast<SIMDType>(inputType);
      inputSimd && inputSimd.getSize() != outputType.cast<SIMDType>().getSize())
    return emitOpError("cannot cast between SIMD types of different sizes");

  return success();
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDShuffleOp::verify() {
  Optional<int64_t> size = getType().getResolvedSize();
  if (!size || static_cast<size_t>(*size) != getMask().size())
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
// OptionalAlignmentParamValue
//===----------------------------------------------------------------------===//

static void printOptionalAlignmentParamValue(AsmPrinter &p, Operation *op,
                                             TypedAttr alignment) {
  if (!alignment)
    return;
  p << "align ";
  printParamValue(p, alignment);
  p << " ";
}

/// Parse a parameter value that is known to be an alignment type.
ParseResult parseOptionalAlignmentParamValue(AsmParser &p, TypedAttr &result) {
  if (p.parseOptionalKeyword("align")) {
    result = TypedAttr();
    return success();
  }

  if (parseIndexParamValue(p, result))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   Optional<unsigned> alignment) {
  auto type =
      ParamRefType::get(ptr.getType().cast<PointerType>().getElementType());
  TypedAttr alignmentAttr;
  if (alignment)
    alignmentAttr = b.getIndexAttr(*alignment);
  build(b, state, type, ptr, alignmentAttr);
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
static LogicalResult
verifyStructValueType(Operation *op, mlir::TypedValue<StructType> container,
                      IntegerAttr indexAttr, Type valueType,
                      StringRef valueKind) {
  ArrayRef<TypedAttr> elementTypes = container.getType().getElementTypes();
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
  return verifyStructValueType(*this, getContainer(), getIndexAttr(), getType(),
                               "result");
}

LogicalResult StructGetOp::inferReturnTypes(MLIRContext *context,
                                            Optional<Location> loc,
                                            ValueRange operands,
                                            DictionaryAttr attrs,
                                            mlir::RegionRange regions,
                                            SmallVectorImpl<Type> &types) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    if (loc)
      return mlir::emitError(*loc, msg);
    return failure();
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto structType = dyn_cast<StructType>(operands.front().getType());
  if (!structType)
    return emitError("expected struct operand");
  mlir::OperationName name(getOperationName(), context);
  auto indexAttr =
      dyn_cast_if_present<IntegerAttr>(attrs.get(getIndexAttrName(name)));
  if (!indexAttr)
    return emitError("expected an integer index attribute");
  size_t index = indexAttr.getInt();
  if (index >= structType.getNumElements())
    return emitError("struct element index out of bounds");
  types.push_back(ParamRefType::get(structType.getElementTypes()[index]));
  return success();
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
  return verifyStructValueType(*this, getContainer(), getIndexAttr(),
                               getValue().getType(), "operand");
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
static LogicalResult verifyArrayIndex(Operation *op, IntegerAttr indexAttr,
                                      POP::ArrayType arrayType) {
  if (Optional<int64_t> size = arrayType.getResolvedSize()) {
    int64_t index = indexAttr.getInt();
    if (index >= *size)
      return op->emitOpError("array index out of bounds (")
             << index << " >= " << *size << ")";
  }
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
  return verifyArrayIndex(*this, getIndexAttr(), getArray().getType());
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayReplaceOp::verify() {
  return verifyArrayIndex(*this, getIndexAttr(), getArray().getType());
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

/// Canonicalize `pop.variant.get(pop.variant.create(x)) -> x`.
OpFoldResult VariantGetOp::fold(ArrayRef<Attribute> operands) {
  auto create = getVariant().getDefiningOp<VariantCreateOp>();
  if (!create || create.getOperand().getType() != getType())
    return {};
  return create.getOperand();
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

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess GlobalConstantOp::finalizeElaboration() {
  Type type = getType().getResolvedElementType();
  DType dtype;
  if (auto array = dyn_cast<ArrayType>(type))
    dtype =
        *array.getResolvedElementType().cast<ScalarType>().getResolvedDType();
  else
    dtype = *type.cast<DTypeInterface>().getResolvedDType();

  ErrorOr<TypedAttr> value = reifyConstant(getValue(), dtype, type);
  if (value.isError())
    return value.takeError();
  setValueAttr(value.takeValue());
  return success();
}

LogicalResult GlobalConstantOp::verify() {
  return verifyConstant([this](StringRef msg) { return emitOpError(msg); },
                        getValue(),
                        ParamRefType::get(getType().getElementType()));
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
                                     TypeArrayAttr cases,
                                     mlir::RegionRange regions) {
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
    if (getRegions().back()->getNumArguments())
      return emitOpError("expected default region to have zero arguments");
  }
  for (Region *region : getRegions()) {
    auto yield = cast<YieldOp>(region->front().getTerminator());
    if (yield.getOperandTypes() != getResultTypes()) {
      return (emitOpError("operand types of region #")
              << region->getRegionNumber()
              << " yield do not match result types")
                 .attachNote(yield.getLoc())
             << "see terminator here";
    }
  }
  for (auto [type, region] : llvm::zip(getCases(), getRegions())) {
    if (region->getNumArguments() != 1)
      return emitOpError("expected region #")
             << region->getRegionNumber() << " to have one argument";
    if (region->getArgumentTypes().front() != type)
      return emitOpError("expected region #")
             << region->getRegionNumber() << " argument type to be " << type;
  }
  return success();
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
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &successors) {
  // All regions branch back to the parent op.
  if (index) {
    successors.emplace_back(getResults());
    return;
  }

  // The known variant type of the operand can be used to narrow the successor
  // regions of the parent op to just one, but we can't do that here.
  for (Region *region : getRegions())
    successors.emplace_back(region, region->getArguments());
}

OperandRange
VariantVisitOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index);
  if (getCases().size() != getVariant().getType().getTypes().size() &&
      *index == getNumRegions() - 1)
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
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Verify the conversion between the higher-level type and lower-level type.
static LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     Type popType, Type builtinType) {
  // Verify the scalar dtype matches the MLIR type.
  if (auto scalar = dyn_cast<ScalarType>(popType)) {
    if (!builtinType.isa<IntegerType, FloatType>())
      return emitError("expected an integer or float type");

    if (auto dtype = dyn_cast<DTypeConstantAttr>(scalar.getDType());
        dtype && !dtype.isConvertibleTo(builtinType))
      return emitError("cannot convert from scalar dtype ")
             << dtype.getDType().getAsString() << " to " << builtinType;
    return success();
  }

  // Verify the SIMD size matches the vector size and the dtypes match.
  if (auto simd = dyn_cast<SIMDType>(popType)) {
    auto vector = dyn_cast<VectorType>(builtinType);
    if (!vector || vector.getRank() != 1 || vector.getNumScalableDims() != 0)
      return emitError("expected a rank 1 non-scalable vector");

    auto size = dyn_cast<IntegerAttr>(simd.getSize());
    if (size && size.getInt() != vector.getShape().front())
      return emitError("expected vector<") << size.getInt() << "xT>";

    if (auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
        dtype && !dtype.isConvertibleTo(vector.getElementType()))
      return emitError("cannot convert from SIMD dtype ")
             << dtype.getDType().getAsString() << " to vector element "
             << vector.getElementType();
    return success();
  }

  return emitError("cannot convert type ") << popType;
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
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
