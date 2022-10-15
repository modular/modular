//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "Support/ForwardDecls.h"
#include "Support/MDialect/MTypes.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace M;
using namespace KGEN;
using namespace ZAP;

/// Returns success if the memory value and SIMD value have the same dtype,
/// otherwise emit an Op error and return failure.
static LogicalResult
verifyHasSameUnderlyingDType(Operation *op, Value memory,
                             mlir::TypedValue<POP::SIMDType> simd) {
  TypedAttr aDType = TypeSwitch<Type, TypedAttr>(memory.getType())
                         .Case<BufferType, TensorType>(
                             [](auto type) { return type.getDType(); })
                         .Default([](Type) { return TypedAttr(); });
  TypedAttr bDType = simd.getType().getDType();
  if (aDType == bDType)
    return success();
  return op->emitOpError("the type (")
         << memory.getType()
         << ") must have the same element type as the simd type ("
         << simd.getType() << ")";
}

//===----------------------------------------------------------------------===//
// ZAPDialect
//===----------------------------------------------------------------------===//

void ZAPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/ZAPDialect/ZAP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// BufferConstructOp
//===----------------------------------------------------------------------===//

LogicalResult BufferConstructOp::verify() {
  BufferType type = getType();
  if (!type.getSize() == !getSize())
    return emitOpError(
        "requires either a size operand or a buffer type with static size");
  if (!type.getDType() == !getDType())
    return emitOpError(
        "requires either a dtype operand or a buffer type with static dtype");
  return success();
}

void BufferConstructOp::build(OpBuilder &b, OperationState &state, Type type,
                              Value ptr) {
  build(b, state, type, ptr, /*size=*/Value(), /*dtype=*/Value());
}

//===----------------------------------------------------------------------===//
// BufferSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferSizeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "zap.buffer.size has one operand");
  // A null size indicates ? size (unknown size). Since returning null
  // indicates that we don't fold anything, we don't need to check if
  // size is null.
  return getBuffer().getType().getSize();
}

//===----------------------------------------------------------------------===//
// BufferDTypeOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferDTypeOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.size() == 1 && "zap.buffer.dtype has one operand");
  // A null dtype indicates ? dtype (unknown dtype). Since returning null
  // indicates that we don't fold anything, we don't need to check if dtype is
  // null.
  return getBuffer().getType().getDType();
}

//===----------------------------------------------------------------------===//
// BufferConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferConvertOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "buffer convert expected 1 operand");
  // Fold cast x to same type.
  if (getOperand().getType() == getType())
    return getOperand();

  // Fold A->B->C casts into a cast of the original cast's operand.
  if (auto castOperand = getOperand().getDefiningOp<BufferConvertOp>()) {
    // A->B->A doesn't need a cast at all.
    if (castOperand.getOperand().getType() == getType())
      return castOperand.getOperand();
    setOperand(castOperand.getOperand());
    return getResult();
  }

  return {};
}

bool BufferConvertOp::areCastCompatible(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != 1 || rhs.size() != 1)
    return false;
  return lhs.front().isa<BufferType>() && rhs.front().isa<BufferType>();
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
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Parse the dtype of a constant buffer.
static ParseResult
parseConstantBufferDType(AsmParser &p, ArrayElementsAttr values, Type &result) {
  TypedAttr dtype;
  if (parseParamValue(p, dtype, DTypeType::get(p.getContext())))
    return failure();
  result = BufferType::get(p.getBuilder().getIndexAttr(values.size()), dtype);
  return success();
}

/// Print the dtype of a constant buffer.
static void printConstantBufferDType(AsmPrinter &p, Operation *op,
                                     ArrayElementsAttr values, Type result) {
  printParamValue(p, result.cast<BufferType>().getDType());
}

LogicalResult BufferConstantOp::verify() {
  auto type = getValues().getType().dyn_cast<ArrayType>();
  if (!type)
    return emitOpError("expected an '!M.array' type");
  return success();
}

//===----------------------------------------------------------------------===//
// SIMDLoadOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSIMDLoadOp::verify() {
  return verifyHasSameUnderlyingDType(*this, getBuffer(), getResult());
}

//===----------------------------------------------------------------------===//
// SIMDStoreOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSIMDStoreOp::verify() {
  return verifyHasSameUnderlyingDType(*this, getBuffer(), getValue());
}

//===----------------------------------------------------------------------===//
// TensorConstructOp
//===----------------------------------------------------------------------===//

LogicalResult TensorConstructOp::verify() {
  TensorType type = getType();
  size_t numUnknownDims =
      std::count_if(type.getShape().begin(), type.getShape().end(),
                    [](auto shape) { return !shape; });
  size_t numShapeParams = getShape().size();
  if (numShapeParams != numUnknownDims)
    return emitOpError("requires the shape operand to match the non-static "
                       "dimensions of the tensor type");
  if (!type.getDType() == !getDType())
    return emitOpError(
        "requires either a dtype operand or a tensor type with static dtype");
  return success();
}

//===----------------------------------------------------------------------===//
// TensorLoadOp
//===----------------------------------------------------------------------===//

template <typename Operation>
static LogicalResult verifyTensorLoadStoreOp(Operation op) {
  size_t positionsSize = op.getPositions().size();
  size_t tensorRank =
      op.getTensor().getType().template cast<TensorType>().getRank();
  if (positionsSize == tensorRank)
    return success();
  return op.emitOpError("requires the number of input positions (")
         << positionsSize << ") to match the rank of the tensor type ("
         << tensorRank << ")";
}

LogicalResult TensorLoadOp::verify() { return verifyTensorLoadStoreOp(*this); }

//===----------------------------------------------------------------------===//
// TensorStoreOp
//===----------------------------------------------------------------------===//

LogicalResult TensorStoreOp::verify() { return verifyTensorLoadStoreOp(*this); }

//===----------------------------------------------------------------------===//
// TensorSIMDLoadOp
//===----------------------------------------------------------------------===//

LogicalResult TensorSIMDLoadOp::verify() {
  if (failed(verifyHasSameUnderlyingDType(*this, getTensor(), getResult())))
    return failure();
  return verifyTensorLoadStoreOp(*this);
}

//===----------------------------------------------------------------------===//
// TensorStoreOp
//===----------------------------------------------------------------------===//

LogicalResult TensorSIMDStoreOp::verify() {
  if (failed(verifyHasSameUnderlyingDType(*this, getTensor(), getValue())))
    return failure();
  return verifyTensorLoadStoreOp(*this);
}

//===----------------------------------------------------------------------===//
// TensorDimOp
//===----------------------------------------------------------------------===//

LogicalResult TensorDimOp::verify() {
  size_t index = getIndexAttr().getInt();
  size_t rank = getTensor().getType().getRank();
  if (index < rank)
    return success();

  return emitOpError("requires the '")
         << index
         << "' index to be less than the rank of the "
            "tensor's rank of '"
         << rank << "'";
}

OpFoldResult TensorDimOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "zap.tensor.dim has one operand");
  // A null size indicates ? size (unknown size). Since returning null
  // indicates that we don't fold anything, we don't need to check if
  // size is null.
  return getTensor().getType().getShape()[getIndexAttr().getInt()];
}

//===----------------------------------------------------------------------===//
// TensorDTypeOp
//===----------------------------------------------------------------------===//

OpFoldResult TensorDTypeOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.size() == 1 && "zap.tensor.dtype has one operand");
  // A null dtype indicates ? dtype (unknown dtype). Since returning null
  // indicates that we don't fold anything, we don't need to check if dtype is
  // null.
  return getTensor().getType().getDType();
}

//===----------------------------------------------------------------------===//
// TensorRankOp
//===----------------------------------------------------------------------===//

OpFoldResult TensorRankOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.size() == 1 && "zap.tensor.dtype has one operand");
  // The rank is always known for a tensor.
  return IntegerAttr::get(IndexType::get(getContext()),
                          getTensor().getType().getRank());
}

//===----------------------------------------------------------------------===//
// TensorSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult TensorSizeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "zap.tensor.size has a single operand");
  if (auto size = getTensor().getType().getResolvedSize())
    return IntegerAttr::get(IndexType::get(getContext()), *size);
  return {};
}

//===----------------------------------------------------------------------===//
// GlobalStringOp
//===----------------------------------------------------------------------===//

/// Parse just the size of the array. Infer the element type.
static ParseResult parseStringSizeArray(AsmParser &p, Type &type) {
  int64_t size;
  if (p.parseInteger(size))
    return failure();
  type = POP::PointerType::get(POP::ArrayType::get(
      size, POP::ScalarType::get(p.getContext(), DType::si8)));
  return success();
}

/// Print just the size of the array.
static void printStringSizeArray(AsmPrinter &p, Operation *op, Type type) {
  p << *type.cast<POP::PointerType>()
            .getResolvedElementType()
            .cast<POP::ArrayType>()
            .getResolvedSize();
}

/// Returns true if the type is an array of scalar `si8`.
static bool isSI8Array(Type type) {
  if (auto elementType = type.cast<POP::ArrayType>().getResolvedElementType())
    if (auto scalarType = dyn_cast<POP::ScalarType>(elementType))
      return scalarType.getResolvedDType() == DType(DType::si8);
  return false;
}

LogicalResult GlobalStringOp::verify() {
  auto type = getType().getResolvedElementType().cast<POP::ArrayType>();
  int64_t size = *type.getResolvedSize();
  if (size != static_cast<int64_t>(getValue().size()))
    return emitOpError("expected array result to have ")
           << getValue().size() << " elements but got " << size;
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZAPDialect/ZAP.cpp.inc"
