//===- ZAPOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "Support/ForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace KGEN;
using namespace ZAP;

/// Returns true if the buffer value and SIMD value have the same dtype.
static bool hasSameUnderlyingDType(Value buffer,
                                   mlir::TypedValue<SIMDType> simd) {
  TypedAttr aDType = buffer.getType().cast<BufferType>().getDType();
  TypedAttr bDType = simd.getType().getDType();
  return aDType == bDType;
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
parseConstantBufferDType(AsmParser &p, mlir::DenseIntOrFPElementsAttr values,
                         Type &result) {
  TypedAttr dtype;
  if (parseParamValue(p, dtype, DTypeType::get(p.getContext())))
    return failure();
  result = BufferType::get(p.getBuilder().getIndexAttr(values.size()), dtype);
  return success();
}

/// Print the dtype of a constant buffer.
static void printConstantBufferDType(AsmPrinter &p, Operation *op,
                                     mlir::DenseIntOrFPElementsAttr values,
                                     Type result) {
  printParamValue(p, result.cast<BufferType>().getDType());
}

LogicalResult BufferConstantOp::verify() {
  auto type = getValues().getType().dyn_cast<RankedTensorType>();
  // TODO: We need an "#M.array" attribute and type.
  if (!type || type.getRank() != 1)
    return emitOpError("expected a rank 1 tensor type");
  return success();
}

//===----------------------------------------------------------------------===//
// SIMDLoadOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDLoadOp::verify() {
  if (hasSameUnderlyingDType(getBuffer(), getResult()))
    return success();
  return emitOpError("the buffer type (")
         << getBuffer().getType()
         << ") must have the same element type as the result simd type ("
         << getResult().getType() << ")";
}

//===----------------------------------------------------------------------===//
// SIMDStoreOp
//===----------------------------------------------------------------------===//

LogicalResult SIMDStoreOp::verify() {
  if (hasSameUnderlyingDType(getBuffer(), getValue()))
    return success();
  return emitOpError("the buffer type (")
         << getBuffer().getType()
         << ") must have the same element type as the value simd type ("
         << getValue().getType() << ")";
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZAPDialect/ZAP.cpp.inc"
