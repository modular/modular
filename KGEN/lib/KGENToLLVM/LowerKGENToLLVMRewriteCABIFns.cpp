//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LowerKGENToLLVMRewriteCABIFns.h"
#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

/// Emit a stack alloca for `allocType` at b's current insertion point.
/// The alignment is the max of `allocType` and `storeType` ABI alignments,
/// ensuring the subsequent store of `storeType` into the alloca is not
/// under-aligned.
static Value createEntryAlloca(ImplicitLocOpBuilder &b, LLVMPointerType ptrType,
                               Type allocType, Type storeType,
                               const M::KGEN::LLVMDataLayout &dl) {
  uint64_t align =
      std::max(dl.getTypeABIAlign(allocType), dl.getTypeABIAlign(storeType));
  Value count = ConstantOp::create(b, b.getI32IntegerAttr(1));
  return AllocaOp::create(b, ptrType, allocType, count, align);
}

/// For each original entry-block argument, add C ABI block arg(s) per
/// `argInfos` and emit reconstruction code that converts the C ABI value(s)
/// back to the original Mojo type. Returns one reconstructed Value per
/// original arg; the caller is responsible for replacing uses and erasing the
/// originals.
static SmallVector<Value>
reconstructCABIArguments(Block *entry, ArrayRef<M::KGEN::CoercionInfo> argInfos,
                         ArrayRef<BlockArgument> origArgs,
                         ImplicitLocOpBuilder &b, LLVMPointerType ptrType,
                         MLIRContext *ctx, const M::KGEN::LLVMDataLayout &dl) {
  SmallVector<Value> reconstructed;
  for (auto [coercion, origArg] : llvm::zip(argInfos, origArgs)) {
    Type origType = origArg.getType();
    b.setLoc(origArg.getLoc());
    if (coercion.isIdentity()) {
      // No coercion needed: add a new block arg of the same type so the
      // original can be erased uniformly below.
      reconstructed.push_back(entry->addArgument(origType, origArg.getLoc()));
    } else if (coercion.useIndirect) {
      // Indirect: add a pointer block arg and load the original type from it.
      Value ptrArg = entry->addArgument(ptrType, origArg.getLoc());
      reconstructed.push_back(LoadOp::create(b, origType, ptrArg));
    } else if (coercion.isTwoRegister()) {
      // Two-register: add two args, pack them into a struct, store to an
      // alloca sized for origType, then reload as origType (bitcast via mem).
      Value arg1 = entry->addArgument(coercion.coercedType, origArg.getLoc());
      Value arg2 =
          entry->addArgument(coercion.coercedSecondType, origArg.getLoc());
      Type pairTy = LLVMStructType::getLiteral(
          ctx, {coercion.coercedType, coercion.coercedSecondType});
      Value alloca = createEntryAlloca(b, ptrType, origType, pairTy, dl);
      Value pair = UndefOp::create(b, pairTy);
      pair = InsertValueOp::create(b, pair, arg1, size_t{0});
      pair = InsertValueOp::create(b, pair, arg2, size_t{1});
      StoreOp::create(b, pair, alloca);
      reconstructed.push_back(LoadOp::create(b, origType, alloca));
    } else {
      // Single-register coercion (SSE or Integer): bitcast via store + load.
      assert(coercion.coercedType);
      Value coercedArg =
          entry->addArgument(coercion.coercedType, origArg.getLoc());
      Value alloca =
          createEntryAlloca(b, ptrType, origType, coercion.coercedType, dl);
      StoreOp::create(b, coercedArg, alloca);
      reconstructed.push_back(LoadOp::create(b, origType, alloca));
    }
  }
  return reconstructed;
}

/// Rewrite every ReturnOp in `func` to store its value through `retAlloca`
/// and reload as `coercedType`, performing a bitcast-via-memory.
static void rewriteReturnsToCoercedType(LLVMFuncOp func, Value retAlloca,
                                        Type coercedType) {
  func.walk([&](ReturnOp ret) {
    if (ret.getOperands().empty())
      return;
    ImplicitLocOpBuilder rb(ret.getLoc(), ret);
    StoreOp::create(rb, ret.getArg(), retAlloca);
    ret->setOperands(ValueRange({LoadOp::create(rb, coercedType, retAlloca)}));
  });
}

/// Apply C ABI return coercion to `func`: rewrite all ReturnOps and, for the
/// sret case, insert the hidden result-pointer as block arg 0. Returns the
/// new LLVM return type that replaces the original.
static Type applyCABIReturnCoercion(LLVMFuncOp func, Block *entry,
                                    const M::KGEN::CoercionInfo &retInfo,
                                    Type origRetType, ImplicitLocOpBuilder &b,
                                    LLVMPointerType ptrType, MLIRContext *ctx,
                                    Location loc,
                                    const M::KGEN::LLVMDataLayout &dl) {
  if (retInfo.useSRet) {
    // sret: the caller passes a hidden result pointer in arg 0 (x8 on ARM64).
    // Store the return value through it and return void instead.
    Value sretArg = entry->insertArgument(0u, ptrType, loc);
    func.walk([&](ReturnOp ret) {
      ImplicitLocOpBuilder rb(ret.getLoc(), ret);
      StoreOp::create(rb, ret.getArg(), sretArg);
      ret->setOperands(ValueRange());
    });
    return LLVMVoidType::get(ctx);
  }

  // Single- or two-register: bitcast the original return value via memory.
  Type coercedType =
      retInfo.isTwoRegister()
          ? LLVMStructType::getLiteral(
                ctx, {retInfo.coercedType, retInfo.coercedSecondType})
          : retInfo.coercedType;
  assert(coercedType);
  b.setInsertionPointToStart(entry);
  Value retAlloca = createEntryAlloca(b, ptrType, origRetType, coercedType, dl);
  rewriteReturnsToCoercedType(func, retAlloca, coercedType);
  return coercedType;
}

} // namespace

namespace M::KGEN {

void processCABIFunctionDefinition(LLVMFuncOp func, CABIInfo &abiInfo) {
  Location loc = func.getLoc();
  MLIRContext *ctx = func.getContext();
  SmallVector<Type> origArgTypes = llvm::to_vector(func.getArgumentTypes());
  Type origRetType = func.getFunctionType().getReturnType();

  // Classify each argument and return type per platform C ABI rules.
  SmallVector<CoercionInfo> argInfos;
  for (Type argTy : origArgTypes)
    argInfos.push_back(
        abiInfo.classifyArgumentType(argTy, loc, /*isVariadic=*/false));
  CoercionInfo retInfo = abiInfo.classifyReturnType(origRetType, loc);

  bool anyArgCoercion = llvm::any_of(
      argInfos, [](const CoercionInfo &ci) { return !ci.isIdentity(); });
  if (!anyArgCoercion && retInfo.isIdentity())
    return;

  // For external declarations, just update the type — no body to rewrite.
  if (func.isExternal()) {
    auto ptrType = LLVMPointerType::get(ctx);
    SmallVector<Type> newParamTypes;
    if (retInfo.useSRet)
      newParamTypes.push_back(ptrType);
    for (auto [ci, origTy] : llvm::zip(argInfos, origArgTypes)) {
      if (ci.isIdentity()) {
        newParamTypes.push_back(origTy);
      } else if (ci.useIndirect) {
        newParamTypes.push_back(ptrType);
      } else if (ci.isTwoRegister()) {
        newParamTypes.push_back(ci.coercedType);
        newParamTypes.push_back(ci.coercedSecondType);
      } else {
        assert(ci.coercedType);
        newParamTypes.push_back(ci.coercedType);
      }
    }
    Type newRetType;
    if (retInfo.useSRet) {
      newRetType = LLVMVoidType::get(ctx);
    } else if (retInfo.isTwoRegister()) {
      newRetType = LLVMStructType::getLiteral(
          ctx, {retInfo.coercedType, retInfo.coercedSecondType});
    } else if (retInfo.coercedType) {
      newRetType = retInfo.coercedType;
    } else {
      newRetType = origRetType;
    }
    func.setType(LLVMFunctionType::get(newRetType, newParamTypes,
                                       func.getFunctionType().isVarArg()));
    if (retInfo.useSRet)
      func.setArgAttr(0, LLVMDialect::getStructRetAttrName(),
                      mlir::TypeAttr::get(origRetType));
    return;
  }

  Block *entry = &func.getBody().front();
  auto ptrType = LLVMPointerType::get(ctx);
  ImplicitLocOpBuilder b(loc, entry, entry->begin());
  const LLVMDataLayout &dl = abiInfo.getDataLayout();

  // Step 1: Coerce arguments — replace original block args with C ABI args
  // and emit reconstruction code at the entry block top.
  SmallVector<BlockArgument> origArgs = llvm::to_vector(func.getArguments());
  auto reconstructed =
      reconstructCABIArguments(entry, argInfos, origArgs, b, ptrType, ctx, dl);
  for (auto [origArg, reconVal] : llvm::zip(origArgs, reconstructed))
    origArg.replaceAllUsesWith(reconVal);
  entry->eraseArguments(0, origArgTypes.size());

  // Step 2: Coerce return value — rewrite returns and compute the new type.
  Type newRetType =
      retInfo.isIdentity()
          ? origRetType
          : applyCABIReturnCoercion(func, entry, retInfo, origRetType, b,
                                    ptrType, ctx, loc, dl);

  // Step 3: Update the function signature to reflect the coerced ABI.
  func.setType(LLVMFunctionType::get(newRetType,
                                     llvm::to_vector(entry->getArgumentTypes()),
                                     func.getFunctionType().isVarArg()));

  // On ARM64, the sret pointer must go in x8 (XR / indirect result location
  // register), not x0. LLVM only uses x8 when llvm.sret is set on param 0.
  // Without this, C callers put the return address in x8 but the function
  // reads x0 → ABI mismatch and crash.
  if (retInfo.useSRet)
    func.setArgAttr(0, LLVMDialect::getStructRetAttrName(),
                    mlir::TypeAttr::get(origRetType));
}

} // namespace M::KGEN
