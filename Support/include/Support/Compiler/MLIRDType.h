//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_MLIRDTYPE_H
#define SUPPORT_COMPILER_MLIRDTYPE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"

namespace M {
class DType;

/// Check if the float dtype and the MLIR float type are equivalent. The types
/// are equivalent if they represent a concrete float type with the same
/// semantics. For example, `dtype:f16` is equivalent to `mlir:f16`, whereas
/// `dtype:bf16` is equivalent to `mlir:bf16` because they represent the same
/// float type semantics.
bool areEquivalentFloatTypes(DType dtype, FloatType fpType);

/// Given a float dtype, return the equivalent MLIR float type which represents
/// a concrete float type with the same semantics as the dtype. For example,
/// for `dtype:bf16`, this function returns an instance of `mlir::BFloat16Type`.
FloatType getEquivalentFloatType(MLIRContext *ctx, DType dtype);

/// Given an integer dtype, return the equivalent MLIR integer type.
IntegerType getEquivalentIntegerType(MLIRContext *ctx, DType dtype);

/// Given an MLIR float type, return the equivalent dtype. Returns an
/// invalid DType if the MLIR float type is not representable.
DType getEquivalentDType(FloatType fpType);

/// Given an MLIR integer type, return the equivalent dtype. Returns an
/// invalid DType if the MLIR integer type is not representable.
DType getEquivalentDType(IntegerType intType);

} // namespace M

#endif // SUPPORT_COMPILER_MLIRDTYPE_H
