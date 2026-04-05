//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LowerPOPToLLVMExternalCalls.h"
#include "CABICallHelpers.h"
#include "CABILowering.h"
#include "KGEN/POPDialect/POPOps.h"
#include "LLVMLoweringUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace M::KGEN;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPExternalCall
//===----------------------------------------------------------------------===//

/// Lower an external call. Add the callee to the symbol table.
struct ConvertPOPExternalCall : public ConvertSymbolOpToLLVM<ExternalCallOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

private:
  //===--------------------------------------------------------------------===//
  // Helpers for matchAndRewrite
  //===--------------------------------------------------------------------===//

  /// Compute the call argument index where variadic args start.
  /// Accounts for sret pointer and two-register expansion of fixed args.
  unsigned
  computeVariadicCallArgStart(ArrayRef<CoercionInfo> argClassifications,
                              size_t numFixedArgs, bool usesSRet) const {
    unsigned start = usesSRet ? 1 : 0;
    for (size_t i = 0; i < numFixedArgs; ++i) {
      if (argClassifications[i].isTwoRegister())
        start += 2;
      else
        start += 1;
    }
    return start;
  }

  /// Add byval attributes to function declaration for indirect args.
  /// On x86-64 System V, MEMORY class structs are passed by pointer with the
  /// byval attribute so the callee knows to copy from stack.
  void addByvalAttrsToFunc(LLVM::LLVMFuncOp func,
                           ArrayRef<CoercionInfo> argClassifications,
                           ExternalCallOp op, size_t numFixedArgs,
                           bool usesSRet, bool isVariadic) const {
    size_t numParams = isVariadic ? numFixedArgs : argClassifications.size();
    unsigned paramIdx = usesSRet ? 1 : 0;
    for (size_t idx = 0; idx < numParams; ++idx) {
      const auto &coercion = argClassifications[idx];
      if (coercion.useIndirect) {
        Type llvmArgType =
            getTypeConverter()->convertType(op.getOperandTypes()[idx]);
        func.setArgAttr(paramIdx, LLVM::LLVMDialect::getByValAttrName(),
                        mlir::TypeAttr::get(llvmArgType));
      }
      if (coercion.isTwoRegister())
        paramIdx += 2;
      else
        paramIdx += 1;
    }
  }

  /// Add byval attributes on call instruction for variadic indirect args.
  /// Variadic args are not part of the function signature, so byval must
  /// be set on the call itself.
  void addByvalAttrsToCall(LLVM::CallOp call,
                           ArrayRef<CoercionInfo> argClassifications,
                           ExternalCallOp op, size_t numFixedArgs,
                           bool usesSRet,
                           ConversionPatternRewriter &rewriter) const {
    bool hasVariadicIndirect = false;
    for (size_t idx = numFixedArgs; idx < argClassifications.size(); ++idx) {
      if (argClassifications[idx].useIndirect) {
        hasVariadicIndirect = true;
        break;
      }
    }
    if (!hasVariadicIndirect)
      return;

    unsigned varArgStart =
        computeVariadicCallArgStart(argClassifications, numFixedArgs, usesSRet);

    unsigned numCallArgs = call.getNumOperands();
    SmallVector<Attribute> argAttrsList;
    if (mlir::ArrayAttr existing = call.getArgAttrsAttr())
      argAttrsList.assign(existing.begin(), existing.end());
    else
      argAttrsList.assign(numCallArgs, rewriter.getDictionaryAttr({}));

    unsigned varCallIdx = varArgStart;
    for (size_t idx = numFixedArgs; idx < argClassifications.size(); ++idx) {
      if (argClassifications[idx].useIndirect && varCallIdx < numCallArgs) {
        Type llvmArgType =
            getTypeConverter()->convertType(op.getOperandTypes()[idx]);
        auto byvalAttr =
            rewriter.getNamedAttr(LLVM::LLVMDialect::getByValAttrName(),
                                  mlir::TypeAttr::get(llvmArgType));
        auto existing = cast<DictionaryAttr>(argAttrsList[varCallIdx]);
        SmallVector<NamedAttribute> merged(existing.begin(), existing.end());
        merged.push_back(byvalAttr);
        argAttrsList[varCallIdx] = rewriter.getDictionaryAttr(merged);
      }
      if (argClassifications[idx].isTwoRegister())
        varCallIdx += 2;
      else
        varCallIdx += 1;
    }
    call.setArgAttrsAttr(rewriter.getArrayAttr(argAttrsList));
  }

  /// On ARM64, bitcast variadic float args < 64 bits to integer to prevent
  /// LLVM's float→double promotion. This ensures raw bits are placed in GPRs
  /// for va_arg struct reads. On x86-64, floats go in XMM registers and
  /// must NOT be bitcast.
  void applyARM64VariadicFloatBitcast(
      SmallVectorImpl<Value> &callArgs,
      ArrayRef<CoercionInfo> argClassifications, size_t numFixedArgs,
      bool usesSRet, Location loc, ConversionPatternRewriter &rewriter) const {
    unsigned varStart =
        computeVariadicCallArgStart(argClassifications, numFixedArgs, usesSRet);
    for (unsigned i = varStart; i < callArgs.size(); ++i) {
      if (auto floatTy = dyn_cast<FloatType>(callArgs[i].getType())) {
        unsigned bitWidth = floatTy.getWidth();
        if (bitWidth < 64) {
          Type intType = IntegerType::get(getContext(), bitWidth);
          callArgs[i] =
              LLVM::BitcastOp::create(rewriter, loc, intType, callArgs[i]);
        }
      }
    }
  }

  /// Remap argument attributes from POP-level indices to LLVM-level indices.
  /// ABI coercion can change the parameter list: sret prepends a hidden
  /// pointer, two-register args expand to two parameters, etc. The original
  /// argAttrs from the POP op use POP-level indexing, so we must remap them
  /// to match the LLVM function's parameter layout.
  mlir::ArrayAttr remapArgAttrs(mlir::ArrayAttr argAttrs,
                                ArrayRef<CoercionInfo> argClassifications,
                                bool usesSRet,
                                ConversionPatternRewriter &rewriter) const {
    if (!argAttrs)
      return nullptr;

    // Count the total LLVM parameters
    unsigned numLLVMParams = usesSRet ? 1 : 0;
    for (const auto &c : argClassifications) {
      numLLVMParams += c.isTwoRegister() ? 2 : 1;
    }

    // Build remapped attrs: empty dict for each LLVM param, then fill in
    auto emptyDict = rewriter.getDictionaryAttr({});
    SmallVector<Attribute> remapped(numLLVMParams, emptyDict);

    unsigned llvmIdx = usesSRet ? 1 : 0;
    for (size_t popIdx = 0; popIdx < argClassifications.size(); ++popIdx) {
      // Copy the original attr if it exists for this POP arg
      if (popIdx < argAttrs.size()) {
        remapped[llvmIdx] = argAttrs[popIdx];
      }
      // Two-register args expand: second LLVM param gets empty attrs
      llvmIdx += argClassifications[popIdx].isTwoRegister() ? 2 : 1;
    }

    return rewriter.getArrayAttr(remapped);
  }

public:
  /// Lower a POP external_call to an LLVM call, applying C ABI struct
  /// coercion when needed.
  ///
  /// **Why C ABI coercion is necessary:**
  ///
  /// When Mojo calls a C function that takes or returns a struct by value,
  /// the struct cannot simply be passed as-is in LLVM IR. The platform's
  /// C calling convention dictates *how* the struct's bytes are delivered
  /// to the callee — in integer registers, floating-point registers, on
  /// the stack, or via a hidden pointer — depending on the struct's size,
  /// field types, and target architecture.
  ///
  /// For example, a C function `struct Pair add(struct Pair p)` where
  /// `Pair` is `{int a, int b}` (8 bytes) expects its argument in a
  /// single 64-bit integer register on both x86-64 and ARM64. Without
  /// coercion, this lowers to an LLVM IR parameter of type `{i32, i32}`,
  /// which LLVM's backend
  /// decomposes into two separate i32 values in two registers (%edi and
  /// %esi on x86-64). The callee, compiled by Clang with ABI coercion,
  /// expects both fields packed into %rdi — so it reads garbage for the
  /// second field. C ABI coercion is the frontend's responsibility;
  /// LLVM's backend does not re-derive it for aggregate types.
  ///
  /// The rules differ by platform:
  /// - **x86-64 System V**: classifies each 8-byte "eightbyte" of the
  ///   struct as INTEGER or SSE based on field types; structs >16 bytes
  ///   are passed by pointer.
  /// - **ARM64 AAPCS**: non-HFA structs always use integer registers;
  ///   HFA (Homogeneous Float Aggregate) structs use SIMD registers;
  ///   structs >16 bytes are passed by pointer.
  /// - Other targets (e.g., Win64, 32-bit x86) use a DefaultCABIInfo
  ///   pass-through and will need dedicated implementations when supported.
  ///
  /// **How this function works (outline):**
  ///
  /// 1. **Classify** each argument and the return value using a
  ///    platform-specific ABI handler (SystemVABIInfo or AAPCSABIInfo).
  ///    Each type gets a CoercionInfo describing how it must be passed:
  ///    identity (no change), coerce to integer/float, split across two
  ///    registers, or pass indirectly via pointer.
  ///
  /// 2. **Build the LLVM function signature** from the classifications.
  ///    Coerced args become their target types (e.g., kgen.struct → i64);
  ///    two-register args expand to two parameters; indirect args become
  ///    pointers; sret returns prepend a hidden pointer parameter.
  ///
  /// 3. **Prepare call arguments** by storing each struct to the stack
  ///    and reloading it as the coerced type (a store/load "bitcast").
  ///    Identity args pass through unchanged.
  ///
  /// 4. **Reverse-coerce the return value**: store the coerced result
  ///    back to the stack and reload it as the original struct type.
  ///
  /// The classification pipeline is unified: even when all types are
  /// identity (no coercion needed), the same code path runs — identity
  /// classifications simply produce pass-through behavior identical to
  /// standard LLVM type conversion.
  LogicalResult
  matchAndRewrite(ExternalCallOp op, ExternalCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const llvm::Triple &triple = getTypeConverter()->getTarget().getTriple();
    CABICallHelper cabi(getTypeConverter(), getContext(), op.getOperation());

    // Step 1: Create ABI handler (DefaultCABIInfo for unsupported platforms)
    std::unique_ptr<CABIInfo> abiHandler = cabi.createABIHandler();

    // Determine number of fixed arguments (for variadic functions)
    size_t numFixedArgs = adaptor.getOperands().size();
    if (auto fnType = op.getFnType()) {
      numFixedArgs = fnType->getNumInputs();
      // Validate that the declared fixed arg count doesn't exceed actual
      // operands. A malformed op would cause out-of-bounds access in argument
      // classification.
      assert(
          numFixedArgs <= adaptor.getOperands().size() &&
          "variadic function declares more fixed args than provided operands");
    }

    // Step 2: Classify arguments and return value.
    // Identity classifications produce pass-through (no coercion).
    // Note: createABIInfo always returns a valid handler (DefaultCABIInfo
    // for unsupported platforms), so abiHandler is never null.
    SmallVector<CoercionInfo> argClassifications = cabi.classifyArgs(
        abiHandler.get(), op.getOperandTypes(), loc, numFixedArgs);
    CoercionInfo returnClassification;
    if (!op.getResults().empty()) {
      returnClassification =
          cabi.classifyReturn(abiHandler.get(), op.getResult().getType(), loc);
    }

    // Step 3: Build function signature from ABI classifications.
    // Identity classifications produce the same types as standard conversion.
    Type llvmReturnType;
    if (!op.getResults().empty()) {
      llvmReturnType =
          getTypeConverter()->convertType(op.getResult().getType());
    }
    bool isVariadic = op.getFnType().has_value();
    auto [signature, usesSRet] = cabi.buildFunctionType(
        argClassifications, returnClassification, adaptor.getOperands(),
        llvmReturnType, numFixedArgs, isVariadic);

    // Step 4: Get passthrough attributes
    mlir::ArrayAttr passthrough = attachTargetPassthroughAttrs(
        rewriter, getTypeConverter()->getTarget(), op.getFuncAttrsAttr());
    mlir::ArrayAttr argAttrs = op.getArgAttrsAttr();
    mlir::DictionaryAttr resAttrs = op.getResAttrsAttr();
    mlir::ArrayAttr resArrayAttrs;
    if (resAttrs)
      resArrayAttrs = rewriter.getArrayAttr(resAttrs);
    auto memory = dyn_cast_or_null<LLVM::MemoryEffectsAttr>(op.getMemoryAttr());

    // Step 5: Lookup existing function (unified path - no early return!)
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(op.getCallee().getValue());

    if (func && func.getFunctionType() != signature) {
      return mlir::emitError(loc,
                             "existing function with conflicting signature")
                 .attachNote(func.getLoc())
             << "see function declaration here";
    }
    if (func &&
        std::make_tuple(func.getPassthroughAttr(), func.getArgAttrsAttr(),
                        func.getResAttrsAttr(), func.getMemoryEffectsAttr()) !=
            std::make_tuple(passthrough, argAttrs, resArrayAttrs, memory)) {
      return mlir::emitError(loc,
                             "existing function with conflicting attributes")
                 .attachNote(func.getLoc())
             << "see function declaration here";
    }

    // Step 6: Create function if needed (only branch on creation)
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();
      func = LLVM::LLVMFuncOp::create(rewriter,
                                      mlir::UnknownLoc::get(getContext()),
                                      op.getCallee(), signature);
      func.setPassthroughAttr(passthrough);

      // Set arg attrs first (remapped to LLVM parameter indices), so that
      // sret and byval attributes can overlay on top without being overwritten.
      if (mlir::ArrayAttr remapped =
              remapArgAttrs(argAttrs, argClassifications, usesSRet, rewriter)) {
        func.setArgAttrsAttr(remapped);
      }

      // Add sret attribute on the hidden first parameter
      if (usesSRet) {
        Type llvmRetType =
            getTypeConverter()->convertType(op.getResult().getType());
        func.setArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName(),
                        mlir::TypeAttr::get(llvmRetType));
      }

      // Add byval attributes for indirect (MEMORY class) arguments.
      // Only needed on x86-64 (ARM64 doesn't use byval).
      if (triple.isX86()) {
        addByvalAttrsToFunc(func, argClassifications, op, numFixedArgs,
                            usesSRet, op.getFnType().has_value());
      }

      // resAttrs: skip when sret is active (no LLVM-level return value)
      if (resAttrs && !usesSRet)
        func.setResAttrsAttr(resArrayAttrs);
      if (memory)
        func.setMemoryEffectsAttr(memory);
      symtab.insert(func);
    }

    // Step 7: Build call arguments
    auto [callArgs, sretPointer] = cabi.buildCallArgs(
        argClassifications, returnClassification, adaptor.getOperands(),
        llvmReturnType, loc, rewriter);

    // Darwin ARM64: bitcast variadic floats < 64 bits to integer to prevent
    // LLVM's float→double promotion. Darwin's flat va_list reads all variadic
    // args from the GP save area, so float values must be in GPRs (as
    // integers). Linux AAPCS64 has a separate VR save area; floats stay as
    // floats and land in SIMD registers where va_arg for HFA structs reads
    // them.
    if (op.getFnType().has_value() && triple.isAArch64() &&
        triple.isOSDarwin()) {
      applyARM64VariadicFloatBitcast(callArgs, argClassifications, numFixedArgs,
                                     usesSRet, loc, rewriter);
    }

    // Step 8: Create call
    LLVM::CallOp call = createLLVMCall(rewriter, loc, func, callArgs);

    // Add byval on call for variadic indirect args (x86-64 only)
    if (op.getFnType().has_value() && triple.isX86()) {
      addByvalAttrsToCall(call, argClassifications, op, numFixedArgs, usesSRet,
                          rewriter);
    }

    // Step 9: Handle return value
    if (op.getResults().empty()) {
      // Void return
      rewriter.eraseOp(op);
    } else if (!returnClassification.isIdentity()) {
      // ABI coercion: reverse coercion on the return value.
      Value callResult = call.getNumResults() > 0 ? call.getResult() : Value();
      Value result =
          cabi.extractReturn(returnClassification, callResult, sretPointer,
                             llvmReturnType, loc, rewriter);
      rewriter.replaceOp(op, result);
    } else {
      // Identity return: use standard replacement
      replaceCallWithLLVMCall(rewriter, op, call);
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void M::KGEN::populateLowerPOPExternalCallPatterns(
    mlir::RewritePatternSet &patterns, POPToLLVMTypeConverter &typeConverter,
    mlir::SymbolTable &symtab) {
  patterns.insert<ConvertPOPExternalCall>(typeConverter, symtab);
}
