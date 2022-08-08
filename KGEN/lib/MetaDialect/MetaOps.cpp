//===- MetaOps.cpp --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Meta dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaOps.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/DType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// BufferSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferSizeOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.size() == 1 && "meta.buffer.size has one operand");
  // A null size indicates ? size (unknown size). Since returning null
  // indicates that we don't fold anything, we don't need to check if
  // size is null.
  return getValue().getType().cast<BufferType>().getSize();
}

//===----------------------------------------------------------------------===//
// BufferDTypeOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferDTypeOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.size() == 1 && "meta.buffer.dtype has one operand");
  // A null dtype indicates ? dtype (unknown dtype). Since returning null
  // indicates that we don't fold anything, we don't need to check if dtype is
  // null.
  return getValue().getType().cast<BufferType>().getDtype();
}

//===----------------------------------------------------------------------===//
// BufferAddressOp
//===----------------------------------------------------------------------===//

LogicalResult BufferAddressOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  BufferAddressOpAdaptor adaptor(operands, attributes);
  Type inferredPointerType = PointerType::get(
      context, adaptor.getValue().getType().cast<BufferType>().getDtype());
  inferredReturnTypes.push_back(inferredPointerType);

  return success();
}

//===----------------------------------------------------------------------===//
// BufferCastOp
//===----------------------------------------------------------------------===//

/// Verifies that casting the input buffer to the result buffer is OK.
/// Allows casting to occur if
/// inpDtype == ? || resDtype == ? || inpDtype == resDtype
/// and
/// inpSize == ? || resSize == ? || inpSize == resSize
LogicalResult BufferCastOp::verify() {
  BufferType inputBufTy = getBuffer().getType().cast<BufferType>();
  BufferType resultBufTy = getResult().getType().cast<BufferType>();

  Attribute inputDtype = inputBufTy.getDtype();
  Attribute resultDtype = resultBufTy.getDtype();

  if (inputDtype != resultDtype)
    if (inputDtype && resultDtype)
      // A null dtype indicates a buffer with unknown dtype.
      // TODO: Print the string version of the dtypes instead of a number.
      return emitOpError()
             << "expected the dtype of the input buffer (" << inputDtype
             << ") to the same as to the dtype you are casting to ("
             << resultDtype << "), or one of them to be unknown.";

  Attribute inputSize = inputBufTy.getSize();
  Attribute resultSize = resultBufTy.getSize();

  if (inputSize != resultSize)
    if (inputSize && resultSize)
      // A null size indicates a buffer with unknown size.
      return emitOpError()
             << "expected the size of the input buffer (" << inputSize
             << ") to be equal to the size you are casting it to ("
             << resultSize << "), or one of them to be unknown.";
  return success();
}

OpFoldResult BufferCastOp::fold(ArrayRef<Attribute> constants) {
  // Fold cast x to same type.
  if (getOperand().getType() == getType())
    return getOperand();
  // Fold A->B->C casts into a cast of the original cast's operand.
  if (auto castOperand = getOperand().getDefiningOp<BufferCastOp>()) {
    // A->B->A doesn't need a cast at all.
    if (castOperand.getOperand().getType() == getType())
      return castOperand.getOperand();
    setOperand(castOperand.getOperand());
    return getResult();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Shared logic for MetaCastToBuiltinOp and MetaCastFromBuiltinOp
//===----------------------------------------------------------------------===//

static LogicalResult checkCastedTypes(Operation *op, Type metaTy,
                                      Type standardTy) {
  auto emitError = [&] {
    return op->emitOpError()
           << "does not support casting " << op->getOperand(0).getType()
           << " to " << op->getResult(0).getType();
  };

  if (auto scalarTy = metaTy.dyn_cast<ScalarType>()) {
    // Check that the data types match.
    if (auto dtype = scalarTy.getDtype().dyn_cast<DTypeConstantAttr>();
        dtype && !dtype.isCompatibleWith(standardTy))
      return emitError();
    return success();
  }

  // Check that the standard type is a rank 1 vector with 1 scalable
  // dimension, the dimensions match, and the data types match.
  auto simdTy = metaTy.cast<SIMDType>();
  auto vectorTy = standardTy.dyn_cast<VectorType>();
  if (!vectorTy)
    return emitError();
  if (vectorTy.getNumScalableDims() != 0)
    return emitError() << ": vector type should not be scalable";
  if (vectorTy.getRank() != 1)
    return emitError() << ": expected a rank 1 vector";
  if (auto size = simdTy.getSize().dyn_cast<IntegerAttr>();
      size.getInt() != vectorTy.getShape().front())
    return emitError() << ": dimensions do not match";
  if (auto dtype = simdTy.getDtype().dyn_cast<DTypeConstantAttr>();
      !dtype.isCompatibleWith(vectorTy.getElementType()))
    return emitError() << ": element types do not match";
  return success();
}

//===----------------------------------------------------------------------===//
// MetaCastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Verifies that casting the input scalar to the corresponding standard
/// type is valid.
LogicalResult MetaCastToBuiltinOp::verify() {
  return checkCastedTypes(*this, getOperand().getType(), getType());
}

/// Folds fixed_type -> !meta.type -> fixed_type (for A->B->A only)
OpFoldResult MetaCastToBuiltinOp::fold(ArrayRef<Attribute> constants) {
  if (auto fromFixedType =
          getOperand().getDefiningOp<MetaCastFromBuiltinOp>()) {
    // Note: The defining op will be a MetaCastFromBuiltinOp, since
    // we have two asymmetric cast ops.
    // Fold A->B->A
    if (fromFixedType.getOperand().getType() == getType())
      return fromFixedType.getOperand();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// MetaCastFromBuiltinOp
//===----------------------------------------------------------------------===//

/// Verifies that casting the standard type to !meta type is valid.
LogicalResult MetaCastFromBuiltinOp::verify() {
  return checkCastedTypes(*this, getType(), getOperand().getType());
}

/// Folds !meta.type -> fixed_type -> !meta.type (for A->B->A only)
OpFoldResult MetaCastFromBuiltinOp::fold(ArrayRef<Attribute> constants) {
  if (auto toFixedType = getOperand().getDefiningOp<MetaCastToBuiltinOp>()) {
    // Note: The defining op will be a MetaCastToBuiltinOp, since
    // we have two asymmetric cast ops.
    // Fold A->B->A
    if (toFixedType.getOperand().getType() == getType())
      return toFixedType.getOperand();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/MetaDialect/Meta.cpp.inc"
