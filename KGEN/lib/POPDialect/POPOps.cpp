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
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Reify a single integer or float attribute to an attribute that fits the
/// given dtype.
static ErrorOr<TypedAttr> reifyOneAttribute(Attribute attr, DType dtype) {
  if (auto value = attr.dyn_cast<IntegerAttr>()) {
    auto type = value.getType().cast<IntegerType>();

    if (!dtype.isInt() && !dtype.isFloat())
      return Error("cannot coerce constant value to " + dtype.getAsString());

    if (dtype.isInt()) {
      // Integer to integer conversion. Check that this isn't converting between
      // signed or unsigned integers.
      if (!type.isSignlessInteger() &&
          type.isSignedInteger() != dtype.isSInt()) {
        std::string errorMessage;
        llvm::raw_string_ostream os(errorMessage);
        os << "cannot change signfulness when converting from " << type
           << " to " << dtype.getAsString();
        return Error(std::move(os.str()));
      }

      // Truncate or extend the value depending on the result width.
      APSInt origInt(value.getValue(), dtype.isUInt());
      APSInt intValue = origInt.extOrTrunc(dtype.getWidthInBits());
      if (intValue.extOrTrunc(origInt.getBitWidth()) != origInt)
        return Error("integer constant does not fit into " +
                     dtype.getAsString());

      // Update the integer type and replace the value.
      return IntegerAttr::get(attr.getContext(), intValue);
    }

    // Integer to float conversion. Check for a valid floating point type.
    FloatType fpType = getEquivalentFloatType(attr.getContext(), dtype);
    if (!fpType)
      return Error("unsupported floating point type: " + dtype.getAsString());

    // Roundtrip the integer value through float.
    APFloat apFp(fpType.getFloatSemantics());
    apFp.convertFromAPInt(value.getValue(), !type.isUnsigned(),
                          APFloat::rmNearestTiesToEven);
    APSInt apInt(type.getIntOrFloatBitWidth(), type.isUnsigned());
    bool exact;
    apFp.convertToInteger(apInt, APFloat::rmTowardZero, &exact);

    // Fail if the roundtrip was lossy.
    if (!exact || !APInt::isSameValue(apInt, value.getValue()))
      return Error("integer constant could not be exactly converted to " +
                   dtype.getAsString());
    return FloatAttr::get(fpType, apFp);
  }

  auto value = attr.cast<FloatAttr>();
  if (dtype.isInt()) {
    // Float to integer conversion. Only exact integers can be converted.
    if (!value.getValue().isInteger())
      return Error("only exact integer floats can be converted to integers");

    // Convert the float to an integer.
    APSInt apInt(dtype.getWidthInBits(), dtype.isUInt());
    bool exact;
    value.getValue().convertToInteger(apInt, APFloat::rmTowardZero, &exact);
    assert(exact && "expected an exact integer");

    return IntegerAttr::get(attr.getContext(), apInt);
  }

  // Float to float conversion. Check for a valid floating point type.
  FloatType fpType = getEquivalentFloatType(attr.getContext(), dtype);
  if (!fpType)
    return Error("unsupported floating point type: " + dtype.getAsString());

  // Coerce the floating point type, regardless of lossiness.
  APFloat apFp = value.getValue();
  bool lossy;
  apFp.convert(fpType.getFloatSemantics(), APFloat::rmTowardZero, &lossy);
  return FloatAttr::get(fpType, apFp);
}

/// Reify a primitive constant attribute (integer, float, or vector thereof)
/// to an attribute that fits the given type.
static ErrorOr<TypedAttr> reifyContant(TypedAttr attr, DType dtype, Type type) {
  // If the value is an integer or float attribute, reify it to according to the
  // result dtype.
  if (attr.isa<IntegerAttr, FloatAttr>()) {
    ErrorOr<TypedAttr> result = reifyOneAttribute(attr, dtype);
    if (result.isError())
      return result.takeError();
    // If the result is an array or vector, splat the constant.
    ShapedType shapedType;
    if (auto simd = type.dyn_cast<SIMDType>())
      shapedType = VectorType::get(*simd.resolveSize(), result->getType());
    else if (auto array = type.dyn_cast<ArrayType>())
      shapedType =
          RankedTensorType::get(*array.resolveSize(), result->getType());
    if (shapedType)
      result = DenseElementsAttr::get(shapedType, result.takeValue());
    return result;
  }

  // If the value is an elements attribute, reify each element according to the
  // result dtype.
  auto value = attr.cast<mlir::DenseIntOrFPElementsAttr>();
  SmallVector<Attribute> values;
  values.reserve(value.size());
  for (Attribute attr : value.getValues<Attribute>()) {
    ErrorOr<TypedAttr> result = reifyOneAttribute(attr, dtype);
    if (result.isError())
      return result.takeError();
    values.push_back(result.takeValue());
  }

  // Convert the dtype to an element type.
  Type elType;
  if (dtype.isInt()) {
    elType = IntegerType::get(attr.getContext(), dtype.getWidthInBits(),
                              dtype.isSInt()
                                  ? IntegerType::SignednessSemantics::Signed
                                  : IntegerType::SignednessSemantics::Unsigned);
  } else {
    elType = getEquivalentFloatType(attr.getContext(), dtype);
  }
  return DenseElementsAttr::get(value.getType()
                                    .cast<mlir::SubElementTypeInterface>()
                                    .replaceImmediateSubElements({}, {elType})
                                    .cast<ShapedType>(),
                                values);
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
    if (auto dtype = type.getDType().dyn_cast<DTypeConstantAttr>())
      if (!dtype.isConvertibleFrom(valueType))
        return emitError("cannot convert from attribute type ")
               << valueType << " to dtype " << dtype.getDType().getAsString();
    return success();
  };

  // If the type is a scalar, allow only scalar constant attributes.
  if (auto scalar = type.dyn_cast<ScalarType>()) {
    if (!value.isa<IntegerAttr, FloatAttr>())
      return emitError("scalar constant expected integer or float "
                       "attribute for constant value");
    // If the dtype is specified, ensure it matches the attribute type.
    return checkDType(scalar.cast<DTypeInterface>());
  }

  // Verify array constant.
  if (auto array = type.dyn_cast<ArrayType>()) {
    // If the size is known, require an elements attribute of the same shape.
    if (Optional<int64_t> size = array.resolveSize()) {
      auto elements = value.dyn_cast<mlir::DenseIntOrFPElementsAttr>();
      if (!elements)
        return emitError("expected dense elements attribute for array "
                         "constant with known size");
      auto type = elements.getType().dyn_cast<RankedTensorType>();
      if (!type || type.getRank() != 1 || type.getShape().front() != *size)
        return emitError("expected attribute type to be tensor<")
               << *size << "xT>";
    } else if (!value.isa<IntegerAttr, FloatAttr>()) {
      return emitError("expected integer or float attribute for array "
                       "constant of unspecified size");
    }
    // Only scalar arrays can be created.
    auto scalar = array.resolveElementType().dyn_cast_or_null<ScalarType>();
    if (!scalar)
      return emitError("array constant must have scalar elements");
    return checkDType(scalar.cast<DTypeInterface>());
  }

  // Verify vector constant.
  auto simd = type.cast<SIMDType>();
  // If the size is specified, require an attribute of the same shape.
  if (Optional<int64_t> size = simd.resolveSize()) {
    auto elements = value.dyn_cast<mlir::DenseIntOrFPElementsAttr>();
    if (!elements)
      return emitError("expected dense elements attribute for vector "
                       "constant with known size");
    auto type = elements.getType().dyn_cast<VectorType>();
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
  ErrorOr<TypedAttr> value = reifyContant(
      getValue(), getType().cast<DTypeInterface>().resolveDType(), getType());
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
  auto attr = value.dyn_cast<TypedAttr>();
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
  if (auto simd = type.dyn_cast<SIMDType>())
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

  DType inputDType = inputType.resolveDType();
  DType outputDType = outputType.resolveDType();

  // If neither dtype could be resolved, allow the cast.
  if (inputDType.isInvalid() || outputDType.isInvalid())
    return true;

  ssize_t inputDTypeWidth = inputDType.getWidthInBits();
  ssize_t outputDTypeWidth = outputDType.getWidthInBits();

  // If we have a simd type, then the bitwidths must match.
  Optional<int64_t> inputSize = 1, outputSize = 1;
  if (auto inputSimd = inputType.dyn_cast<SIMDType>()) {
    auto outputSimd = outputType.cast<SIMDType>();
    inputSize = inputSimd.resolveSize();
    outputSize = outputSimd.resolveSize();
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

  if (auto inputSimd = inputType.dyn_cast<SIMDType>();
      inputSimd && inputSimd.getSize() != outputType.cast<SIMDType>().getSize())
    return emitOpError("cannot cast between SIMD types of different sizes");

  return success();
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDShuffleOp::verify() {
  Optional<int64_t> size = getType().resolveSize();
  if (!size || static_cast<size_t>(*size) != getMask().size())
    return emitOpError("expected result to be a vector of ")
           << getMask().size() << " elements";

  auto lhsType = getLhs().getType().cast<SIMDType>();
  if (lhsType.getDType() != getType().getDType())
    return emitOpError("expected result dtype to match operand dtypes");

  if (Optional<int64_t> size = lhsType.resolveSize()) {
    for (int32_t index : getMask())
      if (index >= *size * 2)
        return emitOpError("mask element ") << index << " is out of bounds";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr) {
  auto type = ptr.getType().cast<PointerType>().getElementType();
  build(b, state, ParamRefType::get(type), ptr);
}

//===----------------------------------------------------------------------===//
// GetElementOp
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

LogicalResult GetElementOp::verify() {
  return verifyStructValueType(*this, getContainer(), getIndexAttr(), getType(),
                               "result");
}

LogicalResult GetElementOp::inferReturnTypes(MLIRContext *context,
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
  auto structType = operands.front().getType().dyn_cast<StructType>();
  if (!structType)
    return emitError("expected struct operand");
  mlir::OperationName name(getOperationName(), context);
  auto indexAttr =
      attrs.get(getIndexAttrName(name)).dyn_cast_or_null<IntegerAttr>();
  if (!indexAttr)
    return emitError("expected an integer index attribute");
  size_t index = indexAttr.getInt();
  if (index >= structType.getNumElements())
    return emitError("struct element index out of bounds");
  types.push_back(ParamRefType::get(structType.getElementTypes()[index]));
  return success();
}

void GetElementOp::build(OpBuilder &b, OperationState &state, Value container,
                         int64_t index) {
  return build(b, state, container, b.getIndexAttr(index));
}

//===----------------------------------------------------------------------===//
// ReplaceElementOp
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

LogicalResult ReplaceElementOp::verify() {
  return verifyStructValueType(*this, getContainer(), getIndexAttr(),
                               getValue().getType(), "operand");
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
  Type type = getType().resolveElementType();
  DType dtype;
  if (auto array = type.dyn_cast<ArrayType>())
    dtype = array.resolveElementType().cast<ScalarType>().resolveDType();
  else
    dtype = type.cast<DTypeInterface>().resolveDType();

  ErrorOr<TypedAttr> value = reifyContant(getValue(), dtype, type);
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
// TypeLowerOp
//===----------------------------------------------------------------------===//

/// Verify the conversion between the higher-level type and lower-level type.
static LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     Type high, Type low) {
  // Verify the scalar dtype matches the MLIR type.
  if (auto scalar = high.dyn_cast<ScalarType>()) {
    if (!low.isa<IntegerType, FloatType>())
      return emitError("expected an integer or float type");

    if (auto dtype = scalar.getDType().dyn_cast<DTypeConstantAttr>();
        dtype && !dtype.isConvertibleTo(low))
      return emitError("cannot convert from scalar dtype ")
             << dtype.getDType().getAsString() << " to " << low;
    return success();
  }

  // Verify the SIMD size matches the vector size and the dtypes match.
  if (auto simd = high.dyn_cast<SIMDType>()) {
    auto vector = low.dyn_cast<VectorType>();
    if (!vector || vector.getRank() != 1 || vector.getNumScalableDims() != 0)
      return emitError("expected a rank 1 non-scalable vector");

    auto size = simd.getSize().dyn_cast<IntegerAttr>();
    if (size && size.getInt() != vector.getShape().front())
      return emitError("expected vector<") << size.getInt() << "xT>";

    if (auto dtype = simd.getDType().dyn_cast<DTypeConstantAttr>();
        dtype && !dtype.isConvertibleTo(vector.getElementType()))
      return emitError("cannot convert from SIMD dtype ")
             << dtype.getDType().getAsString() << " to vector element "
             << vector.getElementType();
    return success();
  }

  // TODO: Verify other types through an interface.
  return success();
}

LogicalResult TypeLowerOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getInput().getType(),
      getType());
}

OpFoldResult TypeLowerOp::fold(ArrayRef<Attribute> operands) {
  // Fold A->B->A cast.
  if (auto parent = getInput().getDefiningOp<TypeRaiseOp>();
      parent && parent.getInput().getType() == getType())
    return parent.getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// TypeRaiseOp
//===----------------------------------------------------------------------===//

LogicalResult TypeRaiseOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getType(),
      getInput().getType());
}

OpFoldResult TypeRaiseOp::fold(ArrayRef<Attribute> operands) {
  // Fold A->B->A cast.
  if (auto parent = getInput().getDefiningOp<TypeLowerOp>();
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
