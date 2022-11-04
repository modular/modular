//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Value.h"

namespace M::KGEN {
class KGENDType;

//===----------------------------------------------------------------------===//
// POPToLLVMTypeConverter
//===----------------------------------------------------------------------===//

/// Get the MLIR type for a data type.
llvm::Optional<mlir::Type> getMLIRTypeForDType(mlir::MLIRContext *ctx,
                                               KGENDType dtype,
                                               size_t indexBitwidth);

/// Get an LLVM pointer to the given dtype. If the dtype is unknown, return an
/// untyped pointer.
mlir::Type getLLVMPointerTo(mlir::MLIRContext *ctx, KGENDType dtype,
                            size_t indexBitwidth);

/// This type converter maps fully-specified pop dialect parametric types and
/// built-in MLIR types to LLVM types.
class POPToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  POPToLLVMTypeConverter(mlir::Location loc,
                         const mlir::LowerToLLVMOptions &options);

  /// Report an error or conversion failure.
  /// TODO: TypeConverter needs an error reporting mechanism.
  mlir::InFlightDiagnostic emitError(llvm::StringRef msg) {
    return mlir::emitError(loc) << msg;
  }

private:
  /// A location used to report conversion failures.
  mlir::Location loc;
  /// TODO: We don't have a model for target-specific data layout. Use MLIR's
  /// default data layout.
  mlir::DataLayout dl;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Insert an alloca at the top of the function body.
Value createAllocaAtEntry(Operation *op, Type type, PatternRewriter &rewriter);

/// Compute the bytecount of a buffer of numElements with specified elementType.
int64_t getByteCount(Type elementType, IntegerAttr numElements = {});

} // namespace M::KGEN

#endif // KGEN_LLVM_LOWERING_UTILS_H
