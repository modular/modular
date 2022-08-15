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

OpFoldResult BufferSizeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "meta.buffer.size has one operand");
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
  return getValue().getType().cast<BufferType>().getDType();
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
      context, adaptor.getValue().getType().cast<BufferType>().getDType());
  inferredReturnTypes.push_back(inferredPointerType);

  return success();
}

//===----------------------------------------------------------------------===//
// BufferCastOp
//===----------------------------------------------------------------------===//

/// Verifies that casting the input buffer to the result buffer is okay.
/// Casting is allowed so long as there isn't a statically known problem.
LogicalResult BufferCastOp::verify() {
  BufferType inputBufTy = getBuffer().getType().cast<BufferType>();
  BufferType resultBufTy = getType();

  Attribute inputDtype = inputBufTy.getDType();
  Attribute resultDtype = resultBufTy.getDType();

  // We allow buffer<param> to be cast to buffer<42> because the parameter may
  // be resolved to 42 during elaboration.  Be careful about ?'s which are
  // represented as null (but which are compatible with everything).
  if (inputDtype != resultDtype &&
      inputDtype.isa_and_nonnull<DTypeConstantAttr>() &&
      resultDtype.isa_and_nonnull<DTypeConstantAttr>()) {
    // TODO: Print these attributes prettier.
    return emitError() << "input buffer dtype '" << inputDtype
                       << "' disagrees with result dtype '" << resultDtype
                       << "'";
  }

  Attribute inputSize = inputBufTy.getSize();
  Attribute resultSize = resultBufTy.getSize();
  if (inputSize != resultSize && inputSize.isa_and_nonnull<IntegerAttr>() &&
      resultSize.isa_and_nonnull<IntegerAttr>())
    // TODO: Print these attributes prettier.
    return emitError() << "input buffer size '" << inputSize
                       << "' disagrees with result size '" << resultSize << "'";
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
    if (auto dtype = scalarTy.getDType().dyn_cast<DTypeConstantAttr>();
        dtype && !dtype.isCompatibleWith(standardTy))
      return emitError();
    return success();
  }

  // Check that the standard type is a rank 1 vector with matching dimensions.
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
  if (auto dtype = simdTy.getDType().dyn_cast<DTypeConstantAttr>();
      dtype && !dtype.isCompatibleWith(vectorTy.getElementType()))
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
OpFoldResult MetaCastToBuiltinOp::fold(ArrayRef<Attribute> operands) {
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
OpFoldResult MetaCastFromBuiltinOp::fold(ArrayRef<Attribute> operands) {
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
