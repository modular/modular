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
    auto outputSimd = cast<SIMDType>(outputType);
    inputSize = inputSimd.getResolvedSize();
    outputSize = outputSimd.getResolvedSize();
    // If neither size could be resolved, allow the cast.
    if (!inputSize || !outputSize)
      return true;
  }

  if (inputDType->isBool() || outputDType->isBool())
    return *inputSize == outputDTypeWidth * *outputSize ||
           *outputSize == inputDTypeWidth * *inputSize;

  // If the sizes do not match, then we cannot cast.
  return inputDTypeWidth * *inputSize == outputDTypeWidth * *outputSize;
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

bool PointerBitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return isa<ParamRefType, PointerType, SignatureType>(inputs.front()) &&
         isa<ParamRefType, PointerType, SignatureType>(outputs.front());
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

  return parseColonTypeParamValue(p, mask);
}

static void printShuffleMask(AsmPrinter &p, Operation *op, TypedAttr mask,
                             Type resultType) {
  printColonTypeParamValue(p, mask);
}

LogicalResult SIMDShuffleOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();
  auto maskType = cast<ArrayType>(getMask().getType());
  if (!isa<IndexType>(maskType.getElementType()))
    return emitOpError("expected mask to be a list of indices");
  auto mask = dyn_cast_or_null<ArrayAttr>(getMask());
  if (!mask)
    return success();

  if (*size != static_cast<int64_t>(mask.getValues().size()))
    return emitOpError("expected result to be a vector of ")
           << mask.getValues().size() << " elements";

  auto lhsType = cast<SIMDType>(getLhs().getType());
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
// SIMDSplatOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDSplatOp::verify() {
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return success();

  if (*size <= 0)
    return emitOpError("requires a non-negative size");

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
  build(b, state, arg, ptr,
        alignment ? b.getIndexAttr(*alignment) : TypedAttr());
}

//===----------------------------------------------------------------------===//
// PackLoadOp
//===----------------------------------------------------------------------===//

LogicalResult PackLoadOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &types) {
  if (!isa<PackType>(operands[0].getType()))
    return mlir::emitError(loc.value_or(operands[0].getLoc()),
                           "expected one !kgen.pack operand, not ")
           << operands[0].getType();
  auto packType = cast<PackType>(operands[0].getType());
  // The result type is the same as the input type, but with a layer of pointers
  // stripped off.
  auto mappedTypes =
      ParamOperatorAttr::get(POC::VariadicPtrRemoveMap, packType.getVariadic());
  types.push_back(PackType::get(mappedTypes));
  return success();
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
  // Can only verify with concrete size.
  if (!size)
    return success();
  if (*size < 0)
    return emitOpError("requires a non-negative size");
  if (*size != 0 && getNumOperands() == 0)
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
  return build(b, state, cast<ArrayType>(array.getType()).getElementType(),
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
  auto array = dyn_cast<POP::ArrayType>(ptr.getElementType());
  return array ? PointerType::get(array.getElementType()) : Type();
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

/// Parse the element type of the allocated pointer type.
static ParseResult parsePointerOf(AsmParser &p, Type &result,
                                  TypedAttr &addressSpace) {
  Type elementType;
  if (parseParamType(p, elementType) ||
      KGEN::parseOptionalAddressSpaceParamValue(p, addressSpace))
    return failure();

  result = PointerType::get(p.getContext(), elementType, addressSpace);
  return success();
}

/// Print the element type of the allocated pointer type.
static void printPointerOf(AsmPrinter &p, Operation *op, Type result,
                           TypedAttr addressSpace) {
  printParamType(p, cast<PointerType>(result).getElementType());
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
// StackAllocLifetimeStartOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyLifetimeMarker(Operation *op) {
  for (auto [idx, value] : llvm::enumerate(op->getOperands())) {
    if (auto alloc = value.getDefiningOp<StackAllocationOp>()) {
      continue;
    }
    InFlightDiagnostic diag = op->emitOpError()
                              << "operand #" << idx
                              << " is not defined by a stack allocation op";
    diag.attachNote(value.getLoc()) << "value is defined here";
    return diag;
  }
  return success();
}

LogicalResult StackAllocLifetimeStartOp::verify() {
  return verifyLifetimeMarker(*this);
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeEndOp
//===----------------------------------------------------------------------===//

LogicalResult StackAllocLifetimeEndOp::verify() {
  return verifyLifetimeMarker(*this);
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
  printColonTypeOrIndex(p, cast<PointerType>(type).getElementType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

LogicalResult GlobalConstantOp::verify() {
  if (!isa<ParamRefType>(getResult().getType().getElementType()))
    return success();
  return emitOpError("must have a concrete element type");
}

//===----------------------------------------------------------------------===//
// CallLLVMIntrinsicOp
//===----------------------------------------------------------------------===//

void CallLLVMIntrinsicOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (getHasSideEffects())
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

mlir::Speculation::Speculatability CallLLVMIntrinsicOp::getSpeculatability() {
  if (getHasSideEffects())
    return mlir::Speculation::NotSpeculatable;
  return mlir::Speculation::Speculatable;
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
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

/// Return an KGEN struct type with any integer or pointer followed by a
/// boolean.
static Type getCmpXChgResultType(Type type) {
  auto pointerType = dyn_cast<PointerType>(type);
  if (!pointerType)
    return nullptr;
  Type eltType = pointerType.getElementType();
  auto boolType =
      SIMDType::get(1, DTypeConstantAttr::get(type.getContext(), DType::kBool));
  return StructType::get(type.getContext(), {eltType, boolType});
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
// VariadicSplatOp
//===----------------------------------------------------------------------===//

void VariadicSplatOp::build(OpBuilder &b, OperationState &state,
                            Type resultType, Value element,
                            size_t numElements) {
  assert(isa<VariadicType>(resultType) && "invalid result type");
  build(b, state, resultType, element, b.getIndexAttr(numElements));
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.cpp.inc"
