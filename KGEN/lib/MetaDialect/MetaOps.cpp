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

#include "GenericML/Support/TensorEltType.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypes.h"
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

//===----------------------------------------------------------------------===//\
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

/// Types are compatible if the types are the same, and the bit widths are the
/// same. Allows conversions from signed to unsigned, but does not allow
/// conversion from bf to fp.
static bool typesAreCompatible(Type builtinTy, TensorEltType eltTy) {
  auto builtinWidth = builtinTy.getIntOrFloatBitWidth();
  auto eltWidth = eltTy.getWidthInBits();
  if (eltTy.isInt())
    return builtinTy.isa<IntegerType>() && (builtinWidth == eltWidth);

  if (eltTy.isFloat()) {
    // bf16 and fp16 are not convertible, but are both floats and have  the same
    // bit width, so we have to make sure we're not converting fp16 to bf16 and
    // vice versa.
    if (eltTy == TensorEltType::bf16) {
      return builtinTy.isa<BFloat16Type>();
    }
    if (eltTy == TensorEltType::f16) {
      return builtinTy.isa<Float16Type>() && (builtinWidth == eltWidth);
    }
    if (eltTy == TensorEltType::tf32) {
      // There is no builtin tf32 type, so we can't do the conversion.
      return false;
    }
    return builtinTy.isa<FloatType>() && (builtinWidth == eltWidth);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// MetaCastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Verifies that casting the input scalar to the corresponding standard
/// type is valid.
LogicalResult MetaCastToBuiltinOp::verify() {
  auto scalarOrSIMDTy = getOperand().getType();
  auto standardTy = getType();

  // TODO: Remove when the op supports !meta.simd, and check that !meta.simd is
  // concrete
  if (auto simdTy = scalarOrSIMDTy.dyn_cast<SIMDType>())
    return emitOpError() << "does not support casting !meta.simd types "
                            "currently. Eventually it "
                         << "should.";

  if (auto scalarTy = scalarOrSIMDTy.dyn_cast<ScalarType>()) {
    if (auto dtype = scalarTy.getDtype().dyn_cast<DTypeConstantAttr>()) {
      TensorEltType eltTy = dtype.getTensorEltType();
      if (!typesAreCompatible(standardTy, eltTy))
        return emitOpError() << "does not support casting " << getOperand()
                             << " to " << standardTy << ".";
    }
  }

  return success();
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
  auto standardTy = getOperand().getType();
  auto scalarOrSIMDTy = getType();

  // TODO: Remove when the op supports !meta.simd, and check that !meta.simd
  // is concrete
  if (auto simdTy = scalarOrSIMDTy.dyn_cast<SIMDType>())
    return emitOpError() << "does not support casting to !meta.simd types "
                            "currently. Eventually it "
                         << "should.";

  if (auto scalarTy = scalarOrSIMDTy.dyn_cast<ScalarType>()) {
    if (auto dtype = scalarTy.getDtype().dyn_cast<DTypeConstantAttr>()) {
      TensorEltType eltTy = dtype.getTensorEltType();
      if (!typesAreCompatible(standardTy, eltTy))
        return emitOpError() << "does not support casting " << getOperand()
                             << " to " << scalarOrSIMDTy << ".";
    }
  }
  return success();
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
