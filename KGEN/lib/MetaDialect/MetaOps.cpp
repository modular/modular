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
// RebindOp
//===----------------------------------------------------------------------===//

/// Fold a rebind op if the input and output types are the same or through a
/// transitory rebind.
template <typename RebindOp>
static OpFoldResult foldRebindOp(RebindOp op, ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "rebind op expected 1 operand");
  // Fold cast x to same type.
  if (op.getOperand().getType() == op.getType())
    return op.getOperand();
  // Fold A->B->C casts into a cast of the original cast's operand.
  if (auto castOperand = op.getOperand().template getDefiningOp<RebindOp>()) {
    // A->B->A doesn't need a cast at all.
    if (castOperand.getOperand().getType() == op.getType())
      return castOperand.getOperand();
    op.setOperand(castOperand.getOperand());
    return op.getResult();
  }

  return {};
}

/// Check that two parameterized fields are the same if they are concrete.
template <typename ConcreteType>
static LogicalResult sameIfConcrete(Operation *op, TypedAttr lhs, TypedAttr rhs,
                                    StringRef fieldStr) {
  if (lhs == rhs || !lhs.isa_and_nonnull<ConcreteType>() ||
      !rhs.isa_and_nonnull<ConcreteType>())
    return success();

  // TODO: Print these attributes prettier.
  return op->emitError() << "input " << fieldStr << " '"
                         << getParamAsString(lhs) << "' disagrees with result "
                         << fieldStr << " '" << getParamAsString(rhs) << "'";
}

//===----------------------------------------------------------------------===//
// ScalarRebindOp
//===----------------------------------------------------------------------===//

OpFoldResult ScalarRebindOp::fold(ArrayRef<Attribute> operands) {
  return foldRebindOp(*this, operands);
}

LogicalResult ScalarRebindOp::verify() {
  return sameIfConcrete<DTypeConstantAttr>(
      *this, getInput().getType().cast<ScalarType>().getDType(),
      getType().getDType(), "scalar dtype");
}

//===----------------------------------------------------------------------===//
// SIMDRebindOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDRebindOp::fold(ArrayRef<Attribute> operands) {
  return foldRebindOp(*this, operands);
}

LogicalResult SIMDRebindOp::verify() {
  auto inputTy = getInput().getType().cast<SIMDType>();
  auto outputTy = getOutput().getType().cast<SIMDType>();
  if (failed(sameIfConcrete<DTypeConstantAttr>(
          *this, inputTy.getDType(), outputTy.getDType(), "SIMD dtype")) ||
      failed(sameIfConcrete<IntegerAttr>(*this, inputTy.getSize(),
                                         outputTy.getSize(), "SIMD size")))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// PointerRebindOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerRebindOp::fold(ArrayRef<Attribute> operands) {
  return foldRebindOp(*this, operands);
}

LogicalResult PointerRebindOp::verify() {
  return sameIfConcrete<DTypeConstantAttr>(
      *this, getInput().getType().cast<PointerType>().getDType(),
      getType().getDType(), "pointer dtype");
}

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
// BufferRebindOp
//===----------------------------------------------------------------------===//

/// Verifies that rebinding the input buffer to the result buffer is okay.
/// Rebinding is allowed so long as there isn't a statically known problem.
LogicalResult BufferRebindOp::verify() {
  BufferType inputBufTy = getInput().getType().cast<BufferType>();
  BufferType resultBufTy = getType();

  // We allow buffer<param> to be cast to buffer<42> because the parameter may
  // be resolved to 42 during elaboration.  Be careful about ?'s which are
  // represented as null (but which are compatible with everything).
  if (failed(sameIfConcrete<DTypeConstantAttr>(*this, inputBufTy.getDType(),
                                               resultBufTy.getDType(),
                                               "buffer dtype")) ||
      failed(sameIfConcrete<IntegerAttr>(*this, inputBufTy.getSize(),
                                         resultBufTy.getSize(), "buffer size")))
    return failure();

  return success();
}

OpFoldResult BufferRebindOp::fold(ArrayRef<Attribute> constants) {
  return foldRebindOp(*this, constants);
}

//===----------------------------------------------------------------------===//
// Shared logic for MetaCastToBuiltinOp and MetaCastFromBuiltinOp
//===----------------------------------------------------------------------===//

static LogicalResult checkCastedTypes(Operation *op, Type metaTy,
                                      Type standardTy) {
  // Ignore types that are opaquely casted.
  if (metaTy.isa<PointerType, BufferType>())
    return success();

  if (!metaTy.isa<ScalarType, SIMDType>())
    return op->emitOpError("expected a scalar or SIMD type");

  // Check the builtin types.
  auto emitError = [op](StringRef msg) {
    return op->emitOpError()
           << "does not support casting " << op->getOperand(0).getType()
           << " to " << op->getResult(0).getType() << ": " << msg;
  };
  return checkMetaCastedTypes(emitError, metaTy, standardTy);
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
