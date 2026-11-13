//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Shared C ABI call-lowering helpers used by ConvertPOPExternalCall
// (LowerPOPToLLVMExternalCalls.cpp) and ConvertKGENCallIndirect /
// ConvertKGENCall (LowerKGENToLLVM.cpp) to apply C ABI struct coercion.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H
#define KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H

#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace M::KGEN {

/// Apply the tail-call kind from a kgen.call / kgen.call_indirect to the
/// already-created llvm.call.  Extracted here so both ConvertKGENCall and
/// ConvertKGENCallIndirect share a single copy.
inline void applyTailKind(mlir::LLVM::CallOp call, TailKind kind) {
  switch (kind) {
  case TailKind::MustTail:
    call.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::MustTail);
    break;
  case TailKind::NoTail:
    call.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::NoTail);
    break;
  case TailKind::Tail:
    call.setTailCallKind(mlir::LLVM::tailcallkind::TailCallKind::Tail);
    break;
  case TailKind::None:
    break;
  }
}

/// Return true if `type` is a KGEN scalar of a signed integer dtype. LLVM
/// integers are signless, so this is the only source of the sign for a C ABI
/// integer extension.
bool isSignedIntegerScalar(mlir::Type type);

//===----------------------------------------------------------------------===//
// Coercion through memory
//===----------------------------------------------------------------------===//

/// Return the type of the one LLVM value that holds a register coercion: the
/// coerced type, or a literal struct of the two registers.
mlir::Type getCoercedValueType(const CoercionInfo &coercion,
                               mlir::MLIRContext *ctx);

/// Create a stack slot at the top of the entry block of `func` that can hold
/// a value of `mojoType` and its C ABI form `coercedType`. A slot in the entry
/// block does not grow the stack when the coercion is in a loop.
mlir::Value createCoercionSlot(mlir::OpBuilder &b, mlir::Location loc,
                               mlir::LLVM::LLVMFuncOp func, mlir::Type mojoType,
                               mlir::Type coercedType,
                               const LLVMDataLayout &dataLayout);

/// Return `value` as `destType`: store it to `slot` and load it back.
mlir::Value coerceThroughSlot(mlir::OpBuilder &b, mlir::Location loc,
                              mlir::Value value, mlir::Type destType,
                              mlir::Value slot);

/// Return the Mojo `value` in its C ABI form `coercedType`, through a new slot
/// in `func`.
mlir::Value coerceToC(mlir::OpBuilder &b, mlir::Location loc,
                      mlir::LLVM::LLVMFuncOp func, mlir::Value value,
                      mlir::Type coercedType, const LLVMDataLayout &dataLayout);

/// Return the C ABI `value` as a value of `mojoType`, through a new slot in
/// `func`.
mlir::Value coerceFromC(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::LLVM::LLVMFuncOp func, mlir::Value value,
                        mlir::Type mojoType, const LLVMDataLayout &dataLayout);

/// Result of CABICallHelper::prepareCall().  Bundles everything computed
/// during the classify/build phase that callers need to emit the actual call.
struct CABICallPrep {
  mlir::LLVM::LLVMFunctionType signature;
  SignatureClassification classification;
  /// The LLVM argument and result types the signature was classified with.
  /// The result type is null for a call without a result.
  mlir::SmallVector<mlir::Type> argTypes;
  mlir::Type retType;
  mlir::SmallVector<mlir::Value> callArgs;
  mlir::Value sretPtr;
};

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

  /// Create an alloca in the entry block of the enclosing function.
  /// This ensures stack allocations don't grow unboundedly when the
  /// external_call is inside a loop.
  mlir::Value createEntryBlockAlloca(M::DirectRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::Type elemType) const;

  /// The LLVM function that contains the call.
  mlir::LLVM::LLVMFuncOp getParentFunc() const;

  /// Coerce a single argument and return the LLVM value(s) to pass.
  llvm::SmallVector<mlir::Value> prepareArg(const CoercionInfo &coercion,
                                            mlir::Value orig,
                                            mlir::Location loc,
                                            M::DirectRewriter &rewriter) const;

  /// Build the actual call arguments with C ABI coercion applied.
  /// Handles sret pointer preparation if needed.
  /// Returns {callArgs, sretPointer}.
  std::pair<llvm::SmallVector<mlir::Value>, mlir::Value>
  buildCallArgs(llvm::ArrayRef<CoercionInfo> argClass,
                const CoercionInfo &retClass, mlir::ValueRange originalArgs,
                mlir::Type origRetTy, mlir::Location loc,
                M::DirectRewriter &rewriter) const;

  /// Handle the return value from C ABI call.
  /// Applies reverse coercion (bitcast from integer back to struct).
  /// For sret, loads from the sret pointer.
  mlir::Value extractReturn(const CoercionInfo &retClass,
                            mlir::Value callResult, mlir::Value sretPtr,
                            mlir::Type origRetTy, mlir::Location loc,
                            M::DirectRewriter &rewriter) const;

  /// Classify arguments and return type, build the coerced LLVM function
  /// signature, and prepare coerced call arguments in one step.
  ///
  /// `argTypes` drives ABI classification; `argValues` supplies the SSA values
  /// for coercion. Pass the op's unconverted operand types as `argTypes`: the
  /// signedness of a narrow integer argument comes from them.
  /// Pass `numFixedArgs` and `isVariadic=true` for variadic callees.
  ///
  /// The caller creates an llvm.call from the returned signature and call
  /// arguments, then passes it to finishCall.
  CABICallPrep prepareCall(mlir::TypeRange argTypes, mlir::ValueRange argValues,
                           mlir::Type origRetTy, mlir::Location loc,
                           M::DirectRewriter &rewriter,
                           size_t numFixedArgs = SIZE_MAX,
                           bool isVariadic = false) const;

  /// Finish `call`, created from `prep`: apply the C ABI attributes of every
  /// call argument (variadic ones too) and of the result, apply `tailKind`
  /// where the return
  /// convention allows a tail call, and reverse-coerce the result. Returns
  /// the result in the original return type, or null for a call without a
  /// result.
  mlir::Value finishCall(mlir::LLVM::CallOp call, const CABICallPrep &prep,
                         TailKind tailKind, mlir::Location loc,
                         M::DirectRewriter &rewriter) const;
};

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_CABICALLHELPERS_H
