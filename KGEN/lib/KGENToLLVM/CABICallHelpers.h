//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Shared C ABI call-lowering helpers extracted from ConvertPOPExternalCall.
// Used by ConvertPOPExternalCall (LowerPOPToLLVMExternalCalls.cpp) to apply
// C ABI struct coercion when lowering pop.external_call to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H
#define KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H

#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// CABICallHelper
//===----------------------------------------------------------------------===//

/// Helper struct that holds the C ABI coercion methods shared across call
/// lowering patterns. Constructed with a type converter, context, and parent
/// operation (used to locate the enclosing LLVM function for alloca placement).
struct CABICallHelper {
  const POPToLLVMTypeConverter *tc;
  mlir::MLIRContext *ctx;
  mlir::Operation *parentOp;

  CABICallHelper(const POPToLLVMTypeConverter *tc, mlir::MLIRContext *ctx,
                 mlir::Operation *parentOp)
      : tc(tc), ctx(ctx), parentOp(parentOp) {}

  /// Create a platform-specific C ABI handler for argument/return
  /// classification. Returns a DefaultCABIInfo (pass-through) when C ABI
  /// is disabled or the platform is unsupported.
  std::unique_ptr<CABIInfo> createABIHandler() const;

  /// Classify function arguments according to C ABI rules.
  /// Converts POP types to LLVM types first, then classifies based on the
  /// actual LLVM layout (which may include padding from @align decorators).
  llvm::SmallVector<CoercionInfo> classifyArgs(CABIInfo *handler,
                                               mlir::TypeRange types,
                                               mlir::Location loc,
                                               size_t numFixedArgs) const;

  /// Classify the return value according to C ABI rules.
  /// Converts POP type to LLVM type first to ensure classification uses
  /// the actual LLVM layout. Returns identity (NoClass) for non-ABI path.
  CoercionInfo classifyReturn(CABIInfo *handler, mlir::Type retTy,
                              mlir::Location loc) const;

  /// Build LLVM function type with C ABI type coercion applied.
  /// Returns the function type and whether sret is used.
  /// For variadic functions, only fixed parameters are included in the
  /// function type and isVarArg is set to true.
  std::pair<mlir::LLVM::LLVMFunctionType, bool>
  buildFunctionType(llvm::ArrayRef<CoercionInfo> argClass,
                    const CoercionInfo &retClass, mlir::ValueRange originalArgs,
                    mlir::Type origRetTy, size_t numFixedArgs,
                    bool isVariadic) const;

  /// Create an alloca in the entry block of the enclosing function.
  /// This ensures stack allocations don't grow unboundedly when the
  /// external_call is inside a loop.
  mlir::Value createEntryBlockAlloca(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::Type elemType) const;

  /// Bitcast a value by storing to stack and loading as a different type.
  /// This is the standard LLVM pattern for struct<->scalar bitcasts.
  ///
  /// The allocation is sized to `allocType` which should be the larger of
  /// sourceValue's type and destType to prevent undefined behavior when
  /// coercion rounds up (e.g., 3-byte struct -> i32 = 4 bytes).
  mlir::Value bitcastViaMemory(mlir::Value src, mlir::Type destType,
                               mlir::Type allocType, mlir::Location loc,
                               mlir::ConversionPatternRewriter &rewriter) const;

  /// Create a GEP to access memory at a byte offset from a pointer.
  /// Used for accessing the second register in two-register struct coercion.
  mlir::Value createOffsetGEP(mlir::Value basePtr, int64_t byteOffset,
                              mlir::Location loc,
                              mlir::ConversionPatternRewriter &rewriter) const;

  /// Prepare a two-register argument for C ABI calling convention.
  /// Allocates a struct of both types, stores the original value, then loads
  /// both registers at the correct offsets.
  llvm::SmallVector<mlir::Value>
  prepareTwoRegisterArgument(mlir::Value orig, mlir::Type firstTy,
                             mlir::Type secondTy, mlir::Location loc,
                             mlir::ConversionPatternRewriter &rewriter) const;

  /// Handle a two-register return value from C ABI call.
  /// Extracts both values from the call result, stores them at the correct
  /// offsets, then loads as the original struct type.
  mlir::Value
  handleTwoRegisterReturn(mlir::Value callResult, mlir::Type firstTy,
                          mlir::Type secondTy, mlir::Type origRetTy,
                          mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter) const;

  /// Prepare a single argument with C ABI coercion applied.
  /// Returns the coerced value(s) to pass to the call.
  llvm::SmallVector<mlir::Value>
  prepareArg(const CoercionInfo &coercion, mlir::Value orig, mlir::Location loc,
             mlir::ConversionPatternRewriter &rewriter) const;

  /// Build the actual call arguments with C ABI coercion applied.
  /// Handles sret pointer preparation if needed.
  /// Returns {callArgs, sretPointer}.
  std::pair<llvm::SmallVector<mlir::Value>, mlir::Value>
  buildCallArgs(llvm::ArrayRef<CoercionInfo> argClass,
                const CoercionInfo &retClass, mlir::ValueRange originalArgs,
                mlir::Type origRetTy, mlir::Location loc,
                mlir::ConversionPatternRewriter &rewriter) const;

  /// Handle the return value from C ABI call.
  /// Applies reverse coercion (bitcast from integer back to struct).
  /// For sret, loads from the sret pointer.
  mlir::Value extractReturn(const CoercionInfo &retClass,
                            mlir::Value callResult, mlir::Value sretPtr,
                            mlir::Type origRetTy, mlir::Location loc,
                            mlir::ConversionPatternRewriter &rewriter) const;
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H
