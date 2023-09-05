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
#include "KGEN/KGENDialect/KGENOps.h"
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

/// This is used by the `ArrayElementType` and `VariadicElementType`
/// constraints to match a type range against a single type.
static bool typeRangeMatches(Type type, TypeRange range) {
  return llvm::all_of(range, [&](Type e) { return type == e; });
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

/// Return a SIMD type whose dtype is bool with the same size as the given type.
static Type getBoolOfSameParentType(Type type) {
  auto boolType = DTypeConstantAttr::get(type.getContext(), DType::kBool);
  if (auto simd = dyn_cast<SIMDType>(type))
    return SIMDType::get(simd.getSize(), boolType);
  return nullptr;
}

LogicalResult CmpOp::inferReturnTypes(MLIRContext *ctx,
                                      std::optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
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
  std::optional<KGENDType> inputDType = inputType.getResolvedDType();
  std::optional<KGENDType> outputDType = outputType.getResolvedDType();

  // If neither dtype could be resolved, allow the cast.
  if (!inputDType || !outputDType)
    return true;

  ssize_t inputDTypeWidth = inputDType->getWidthInBits();
  ssize_t outputDTypeWidth = outputDType->getWidthInBits();

  // If we have a simd type, then the bitwidths must match.
  std::optional<int64_t> inputSize = 1, outputSize = 1;
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

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  if (getInput().getType().getSize() != getOutput().getType().getSize())
    return emitOpError("cannot cast between SIMD types of different sizes");
  return success();
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

static ParseResult parseShuffleMask(AsmParser &p, TypedAttr &mask,
                                    Type resultType) {
  return parseParamValue(p, mask,
                         VariadicType::get(p.getBuilder().getIndexType()));
}

static void printShuffleMask(AsmPrinter &p, Operation *op, TypedAttr mask,
                             Type resultType) {
  printParamValue(p, mask);
}

LogicalResult SIMDShuffleOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();
  auto maskType = cast<VariadicType>(getMask().getType());
  if (!isa<IndexType>(maskType.getElementAsType()))
    return emitOpError("expected mask to be a list of indices");
  auto mask = dyn_cast_or_null<VariadicAttr>(getMask());
  if (!mask)
    return success();

  if (*size != static_cast<int64_t>(mask.getValues().size()))
    return emitOpError("expected result to be a vector of ")
           << mask.getValues().size() << " elements";

  auto lhsType = getLhs().getType().cast<SIMDType>();
  if (lhsType.getDType() != getType().getDType())
    return emitOpError("expected result dtype to match operand dtypes");

  if (std::optional<int64_t> size = lhsType.getResolvedSize()) {
    for (TypedAttr indexAttr : mask.getValues()) {
      auto index = dyn_cast<IntegerAttr>(indexAttr);
      if (!index)
        continue;
      if (index.getInt() >= *size * 2)
        return emitOpError("mask element ")
               << index.getInt() << " is out of bounds";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   std::optional<unsigned> alignment) {
  build(b, state, ptr, alignment ? b.getIndexAttr(*alignment) : TypedAttr());
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(OpBuilder &b, OperationState &state, Value arg, Value ptr,
                    std::optional<unsigned> alignment) {
  TypedAttr alignmentAttr;
  if (alignment)
    alignmentAttr = b.getIndexAttr(*alignment);
  build(b, state, arg, ptr, alignmentAttr, {});
}

//===----------------------------------------------------------------------===//
// StructExtractOp
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
           << ParamRefType::get(elementTypes[index]);
  return success();
}

LogicalResult StructExtractOp::verify() {
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

LogicalResult StructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &types) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto structType = dyn_cast<StructType>(operands.front().getType());
  FailureOr<TypedAttr> type =
      inferStructElementType<StructExtractOp>(emitError, structType, attrs);
  if (succeeded(type))
    types.push_back(ParamRefType::get(*type));
  return type;
}

void StructExtractOp::build(OpBuilder &b, OperationState &state,
                            Value container, int64_t index) {
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
      *this, cast<StructType>(getContainer().getType().getElementAsType()),
      getIndexAttr(), getType().getElementAsType(), "result");
}

LogicalResult StructGEPOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &types) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto pointerType = dyn_cast<PointerType>(operands.front().getType());
  if (!pointerType)
    return emitError("expected pointer operand");
  auto structType = dyn_cast<StructType>(pointerType.getElementAsType());
  FailureOr<TypedAttr> type =
      inferStructElementType<StructExtractOp>(emitError, structType, attrs);
  if (succeeded(type))
    types.push_back(PointerType::get(*type));
  return type;
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

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
  std::optional<int64_t> size = getType().getResolvedSize();
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
  std::optional<int64_t> size = arrayType.getResolvedSize();
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
  auto ptr = dyn_cast<PointerType>(arrayPtr);
  if (!ptr)
    return Type();
  auto array = dyn_cast<POP::ArrayType>(ptr.getElementAsType());
  return array ? PointerType::get(array.getElementType()) : Type();
}

//===----------------------------------------------------------------------===//
// PackCreateOp
//===----------------------------------------------------------------------===//

/// Parses a pop.pack.create op.
///
/// operation ::=
///   `pop.pack.create` `(` operands `)` attr-dict `:` result-type
///
/// This is custom because we need to match operands at each index to the
/// resulting pack type element at that index.
static ParseResult parsePackCreateType(AsmParser &p, Type &resultType,
                                       SmallVectorImpl<Type> &elementTypes) {
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseType(resultType))
    return failure();
  auto type = dyn_cast<PackType>(resultType);
  if (!type)
    return p.emitError(loc, "expected a pack type");

  auto variadic = type.getVariadicAttr();
  if (!variadic) {
    // We can only infer if we know the elements of the pack type (i.e.: it is
    // backed by a variadic attribute).
    return p.emitError(loc) << "operand types cannot be "
                               "inferred for resulting pack type "
                            << type;
  }

  ArrayRef<TypedAttr> values = variadic.getValues();
  for (TypedAttr value : values)
    elementTypes.push_back(ParamRefType::get(value));
  return success();
}

static void printPackCreateType(OpAsmPrinter &p, Operation *op, Type resultType,
                                TypeRange elementTypes) {
  p << resultType;
}

//===----------------------------------------------------------------------===//
// PackGetOp
//===----------------------------------------------------------------------===//

/// Given a concrete pack type, such as `<[i32, f32]>`, and an index attribute,
/// such as `1 : index`, we can infer the return type (`f32`). However, if the
/// pack type is not concrete, such as `<Ts>`, or the index is a parametric
/// expression, such as `add(I, 1)`, then we need to parse the return type:
/// "`->` type($result)".
static ParseResult
parsePackGetResultType(AsmParser &p, OpAsmParser::UnresolvedOperand packOperand,
                       Type type, TypedAttr indexAttr, Type &resultType) {
  // Use the pack operand's location for errors. Otherwise, any errors emitted
  // appear on the line following the `pop.pack.get` op, since we're attempting
  // to parse a trailing `-> type($result)` that isn't there.
  llvm::SMLoc loc = packOperand.location;

  auto packType = dyn_cast<PackType>(type);
  if (!packType)
    return p.emitError(loc, "expected a pack type");

  auto index = dyn_cast<IntegerAttr>(indexAttr);
  if (index && index.getInt() < 0)
    return p.emitError(loc) << "pack element index must not be negative";

  auto variadic = packType.getVariadicAttr();
  if (variadic && index) {
    if (index.getInt() >= static_cast<int64_t>(variadic.getValues().size()))
      return p.emitError(loc) << "pack element index out of bounds";

    resultType = ParamRefType::get(variadic.getValues()[index.getInt()]);
    return success();
  }

  if (p.parseArrow())
    return p.emitError(loc) << "could not infer return type and none provided";
  return p.parseType(resultType);
}

/// Only print "`->` type($result)" in cases where the return type cannot be
/// inferred (see `parsePackGetResultType` above).
static void printPackGetResultType(OpAsmPrinter &p, Operation *op,
                                   TypedValue<PackType> pack, PackType type,
                                   TypedAttr indexAttr, Type resultType) {
  if (!pack.getType().getVariadicAttr() || !isa<IntegerAttr>(indexAttr)) {
    p << " -> ";
    p.printType(resultType);
  }
}

LogicalResult PackGetOp::verify() {
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (index && index.getInt() < 0)
    return emitOpError("index ") << index << " must not be negative";

  // If we have a pack backed by an attribute, check that the provided index
  // attribute is within bounds.
  auto variadic = getPack().getType().getVariadicAttr();
  if (!variadic || !index)
    return success();

  ArrayRef<TypedAttr> values = variadic.getValues();
  if (index.getInt() >= static_cast<int64_t>(values.size()))
    return emitOpError("index ")
           << index << " is out of bounds (>=" << values.size() << ")";
  TypedAttr value = values[index.getInt()];
  if (ParamRefType::get(value) != getType()) {
    return emitOpError("result")
           << " type " << getType()
           << " does not match pack element type at index " << index << ": "
           << value;
  }
  return success();
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
static ParseResult parsePointerOf(AsmParser &p, Type &result,
                                  TypedAttr &addressSpace) {
  TypedAttr elementType;
  if (parseTypeParamValue(p, elementType) ||
      KGEN::parseOptionalAddressSpaceParamValue(p, addressSpace))
    return failure();

  result = PointerType::get(elementType, addressSpace);
  return success();
}

/// Print the element type of the allocated pointer type.
static void printPointerOf(AsmPrinter &p, Operation *op, Type result,
                           TypedAttr addressSpace) {
  printTypeParamValue(p, cast<PointerType>(result).getElementType());
  KGEN::printOptionalAddressSpaceParamValue(p, op, addressSpace);
}

void StackAllocationOp::build(OpBuilder &b, OperationState &state, Type result,
                              TypedAttr count, TypedAttr alignment,
                              unsigned addressSpace) {
  build(b, state, result, count, alignment, b.getIndexAttr(addressSpace));
}

void StackAllocationOp::build(OpBuilder &b, OperationState &state, Type result,
                              int64_t count, TypedAttr alignment,
                              unsigned addressSpace) {
  build(b, state, result, b.getIndexAttr(count), alignment, addressSpace);
}

//===----------------------------------------------------------------------===//
// ExternalCallOp
//===---------------------------------------------------------------------===//

static ParseResult parseExternalCallee(AsmParser &p, TypedAttr &callee) {
  StringAttr concreteCallee;
  // Try `@foo`.
  if (succeeded(p.parseOptionalSymbolName(concreteCallee))) {
    callee = StringAttr::get(concreteCallee.getValue(),
                             StringType::get(p.getContext()));
    return success();
  }
  // Otherwise, parse a string expression inside square brackets.
  if (p.parseLSquare() ||
      parseParamValue(p, callee, StringType::get(p.getContext())) ||
      p.parseRSquare())
    return failure();
  return success();
}

static void printExternalCallee(AsmPrinter &p, Operation *op,
                                TypedAttr callee) {
  // Print a symbol name if the callee is concrete.
  if (auto concrete = dyn_cast<StringAttr>(callee)) {
    p.printSymbolName(concrete);
    return;
  }
  // Otherwise, print the string expression in square brackets to disambiguate
  // `callee(` as a parameter operator.
  p << '[';
  printParamValue(p, callee);
  p << ']';
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state, StringRef func,
                           ValueRange operands) {
  build(b, state, {}, func, operands);
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state,
                           TypeRange results, StringRef func,
                           ValueRange operands) {
  build(b, state, results,
        StringAttr::get(func, StringType::get(b.getContext())), operands,
        TypeAttr(), mlir::ArrayAttr(), mlir::ArrayAttr(), mlir::ArrayAttr(),
        Attribute());
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state,
                           TypeRange results, StringRef func,
                           ValueRange operands, FunctionType variadicType) {
  build(b, state, results,
        StringAttr::get(func, StringType::get(b.getContext())), operands,
        TypeAttr::get(variadicType), mlir::ArrayAttr(), mlir::ArrayAttr(),
        mlir::ArrayAttr(), Attribute());
}

LogicalResult
ExternalCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return success();
}

LogicalResult ExternalCallOp::verify() {
  if (mlir::ArrayAttr argAttrs = getArgAttrsAttr()) {
    size_t numArgs;
    if (std::optional<FunctionType> fnType = getVariadicType())
      numArgs = fnType->getNumInputs();
    else
      numArgs = getNumOperands();
    if (argAttrs.size() != numArgs) {
      return mlir::emitError(getLoc(), "external callee has ")
             << numArgs << " arguments but " << argAttrs.size()
             << " argument attributes specified";
    }
  }
  if (mlir::ArrayAttr resAttrs = getResAttrsAttr()) {
    if (getNumResults() != resAttrs.size())
      return mlir::emitError(getLoc(), "external callee has ")
             << getNumResults() << " results but " << resAttrs.size()
             << " result attributes specified";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalAddressOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalAddressOp::verifySymbolUses(SymbolTableCollection &symtab) {
  auto global = symtab.lookupSymbolIn<GlobalOp>(
      (*this)->getParentOfType<ModuleOp>(), getGlobal());
  if (!global)
    return emitOpError("does not reference a `pop.global` operation");
  if (global.getType() != getResult().getType().getElementAsType())
    return emitOpError("result type does not match global type ")
           << global.getType();
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//

void GlobalConstantOp::build(OpBuilder &b, OperationState &state, Type result,
                             TypedAttr value) {
  build(b, state, result, value, TypedAttr{});
}

static ParseResult parseGlobalConstantOpValue(OpAsmParser &p, TypedAttr &value,
                                              Type &resultType) {
  Type elementType;
  if (parseColonTypeOrIndex(p, elementType) || p.parseEqual() ||
      p.parseLess() || parseParamValue(p, value, elementType) ||
      p.parseGreater())
    return failure();
  resultType = PointerType::get(elementType);
  return success();
}

static void printGlobalConstantOpValue(OpAsmPrinter &p, Operation *,
                                       TypedAttr value, Type type) {
  printColonTypeOrIndex(p, cast<PointerType>(type).getElementAsType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

LogicalResult GlobalConstantOp::verify() {
  if (!isa<ParamRefType>(getResult().getType().getElementAsType()))
    return success();
  return emitOpError("must have a concrete element type");
}

//===----------------------------------------------------------------------===//
// CompilerGlobalLoadOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess CompilerGlobalLoadOp::interpret(ArrayRef<Attribute> operands,
                                                   InterpreterState &state) {
  Attribute value = state.getNamedGlobal(getNameAttr());
  if (!value)
    return ErrorTree(getLoc(), "internal error: missing named global '" +
                                   getName() + "'");
  state.mapResults(value);
  return success();
}

//===----------------------------------------------------------------------===//
// CompilerGlobalStoreOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
CompilerGlobalStoreOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state) {
  state.setNamedGlobal(getNameAttr(), operands.front());
  return success();
}

//===----------------------------------------------------------------------===//
// IndexToPointerOp
//===----------------------------------------------------------------------===//

/// Checks the the input pointer type is catabolic to the output address type.
static bool isPointerToAddressCastCompatible(TypeRange inputs,
                                             TypeRange outputs) {
  if (inputs.size() != 1 || inputs.size() != outputs.size())
    return false;

  // The output type must be a vector of indices.
  auto outputType = dyn_cast<SIMDType>(outputs.front());
  if (!outputType || outputType.getResolvedDType().value_or(DType::invalid) !=
                         KGENDType::index)
    return false;

  // The input type can be a vector, in which case its dtype must be address and
  // its size must match the output size.
  if (auto inputType = dyn_cast<SIMDType>(inputs.front())) {
    if (inputType.getResolvedDType().value_or(DType::invalid) !=
        KGENDType::address)
      return false;
    return inputType.getSize() == outputType.getSize();
  }

  // Otherwise, the input type must be a pointer and the output type must be an
  // index scalar.
  return isa<PointerType>(inputs.front()) &&
         outputType.getResolvedSize().value_or(0) == 1;
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
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Verify the conversion between the higher-level type and lower-level type.
static LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     SIMDType simd, Type builtinType) {
  // Verify the SIMD size matches the vector size and the dtypes match.
  auto size = simd.getResolvedSize();
  if (size && *size == 1) {
    // Scalar case
    auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
    if (dtype && !dtype.isConvertibleTo(builtinType))
      return emitError("cannot convert from scalar dtype ")
             << dtype.getDType().getAsString() << " to " << builtinType;
    return success();
  }

  auto vector = dyn_cast<VectorType>(builtinType);
  if (!vector || vector.getRank() != 1 || vector.isScalable())
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

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

LogicalResult CastFromBuiltinOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getType(),
      getInput().getType());
}

//===----------------------------------------------------------------------===//
// CoroutineHandleOp
//===----------------------------------------------------------------------===//

LogicalResult CoroutineHandleOp::verify() {
  if (auto func = (*this)->getParentOfType<FuncOp>()) {
    if (func.getNumResults() != 1) {
      return emitOpError("surrounding function must have 1 result")
                 .attachNote(func.getLoc())
             << "see function here";
    }
    Type resultType = func.getResultTypes().front();
    if (resultType != getType()) {
      return emitOpError("surrounding function result type does not match "
                         "coroutine handle type")
                 .attachNote(func.getLoc())
             << "surrounding function returns " << resultType;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CoroutinePromiseOp
//===----------------------------------------------------------------------===//

static PointerType getCoroutinePromiseType(Type type) {
  return PointerType::get(StructType::get(
      type.getContext(), cast<CoroutineType>(type).getResultTypes()));
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

/// Return an KGEN struct type with any integer or pointer followed by a
/// boolean.
static Type getCmpXChgResultType(Type type) {
  auto pointerType = dyn_cast<PointerType>(type);
  if (!pointerType)
    return nullptr;
  auto eltType = pointerType.getElementAsType();
  auto boolType =
      SIMDType::get(1, DTypeConstantAttr::get(type.getContext(), DType::kBool));
  return POP::StructType::get({eltType, boolType});
}

//===----------------------------------------------------------------------===//
// FenceOp
//===----------------------------------------------------------------------===//

LogicalResult FenceOp::verify() {
  if (llvm::is_contained({AtomicOrdering::NOT_ATOMIC, AtomicOrdering::UNORDERED,
                          AtomicOrdering::MONOTONIC},
                         getOrdering()))
    return emitOpError("can be given only acquire, release, acq_rel, "
                       "and seq_cst orderings");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
