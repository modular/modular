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
                         ListType::get(p.getBuilder().getIndexType(),
                                       cast<SIMDType>(resultType).getSize()));
}

static void printShuffleMask(AsmPrinter &p, Operation *op, TypedAttr mask,
                             Type resultType) {
  printParamValue(p, mask);
}

LogicalResult SIMDShuffleOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();
  auto maskType = cast<ListType>(getMask().getType());
  if (maskType.getResolvedElementType() != Builder(getContext()).getIndexType())
    return emitOpError("expected mask to be a list of indices");
  auto mask = dyn_cast_or_null<ListAttr>(getMask());
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
  build(b, state, arg, ptr, alignmentAttr);
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
           << elementTypes[index];
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
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &types) {
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
  auto array = dyn_cast<POP::ArrayType>(ptr.getResolvedElementType());
  return array ? PointerType::get(array.getElementType()) : Type();
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
  TypedAttr elementType;
  if (parseTypeParamValue(p, elementType))
    return failure();
  result = PointerType::get(elementType);
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
// ExternalCallOp
//===---------------------------------------------------------------------===//

void ExternalCallOp::build(OpBuilder &b, OperationState &state, StringRef func,
                           ValueRange operands) {
  build(b, state, {}, func, operands);
}

void ExternalCallOp::build(OpBuilder &b, OperationState &state,
                           TypeRange results, StringRef func,
                           ValueRange operands) {
  build(b, state, results, func, operands, /*variadicType=*/nullptr);
}

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//

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
  printColonTypeOrIndex(p, cast<PointerType>(type).getResolvedElementType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

LogicalResult GlobalConstantOp::verify() {
  if (getResult().getType().getResolvedElementType())
    return success();
  return emitOpError("must have a concrete element type");
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
  std::optional<int64_t> length = getList().getType().getResolvedLength();
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
           << getNumOperands() << " elements";
  return success();
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

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

LogicalResult CastFromBuiltinOp::verify() {
  return verifyConversionCast(
      [this](StringRef msg) { return emitOpError(msg); }, getType(),
      getInput().getType());
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
  auto eltType = pointerType.getResolvedElementType();
  auto boolType =
      SIMDType::get(1, DTypeConstantAttr::get(type.getContext(), DType::kBool));
  return POP::StructType::get({eltType, boolType});
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
