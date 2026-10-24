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
/// `sigClass` and emit reconstruction code that converts the C ABI value(s)
/// back to the original Mojo type. Returns one reconstructed Value per
/// original arg; the caller is responsible for replacing uses and erasing the
/// originals.
static SmallVector<Value>
reconstructCABIArguments(LLVMFunctionType newFnTy, Block *entry,
                         const M::KGEN::SignatureClassification &sigClass,
                         ArrayRef<BlockArgument> origArgs,
                         ImplicitLocOpBuilder &b, MLIRContext *ctx,
                         const M::KGEN::LLVMDataLayout &dl) {
  SmallVector<Value> reconstructed;
  auto ptrType = LLVMPointerType::get(ctx);
  for (auto [argIdx, coercion, origArg] :
       llvm::enumerate(sigClass.args, origArgs)) {
    Location loc = origArg.getLoc();
    b.setLoc(loc);
    Type origType = origArg.getType();
    unsigned paramIdx = sigClass.getLLVMParamIndex(argIdx);
    auto paramTy = newFnTy.getParamType(paramIdx);
    if (coercion.isIdentity()) {
      // No coercion needed: add a new block arg of the same type so the
      // original can be erased uniformly below.
      reconstructed.push_back(entry->addArgument(paramTy, loc));
    } else if (coercion.useIndirect) {
      // Indirect: add a pointer block arg and load the original type from it.
      Value ptrArg = entry->addArgument(paramTy, loc);
      reconstructed.push_back(LoadOp::create(b, origType, ptrArg));
    } else if (coercion.isTwoRegister()) {
      // Two-register: add two args, pack them into a struct, store to an
      // alloca sized for origType, then reload as origType (bitcast via mem).
      Value arg1 = entry->addArgument(paramTy, loc);
      Value arg2 = entry->addArgument(newFnTy.getParamType(paramIdx + 1), loc);
      assert(arg1.getType() == coercion.coercedType && "C ABI mismatch");
      assert(arg2.getType() == coercion.coercedSecondType && "C ABI mismatch");
      Type pairTy =
          LLVMStructType::getLiteral(ctx, {arg1.getType(), arg2.getType()});
      Value alloca = createEntryAlloca(b, ptrType, origType, pairTy, dl);
      Value pair = UndefOp::create(b, pairTy);
      pair = InsertValueOp::create(b, pair, arg1, size_t{0});
      pair = InsertValueOp::create(b, pair, arg2, size_t{1});
      StoreOp::create(b, pair, alloca);
      reconstructed.push_back(LoadOp::create(b, origType, alloca));
    } else {
      // Single-register coercion (SSE or Integer): bitcast via store + load.
      assert(coercion.coercedType);
      Value coercedArg = entry->addArgument(paramTy, loc);
      assert(coercedArg.getType() == coercion.coercedType && "C ABI mismatch");
      Value alloca =
          createEntryAlloca(b, ptrType, origType, coercedArg.getType(), dl);
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

void processCABIFunctionDefinition(LLVMFuncOp func, CABIInfo &abiInfo,
                                   bool isResultSigned) {
  Location loc = func.getLoc();
  MLIRContext *ctx = func.getContext();
  SmallVector<Type> origArgTypes = llvm::to_vector(func.getArgumentTypes());
  Type origRetType = func.getFunctionType().getReturnType();

  // Classify the whole signature per platform C ABI rules. The arg/return
  // types are already LLVM types here.
  SignatureClassification sigClass = abiInfo.computeSignatureInfo(
      origArgTypes, origRetType, loc, CABIRole::callee(isResultSigned));
  SmallVector<CoercionInfo> &argInfos = sigClass.args;
  CoercionInfo &retInfo = sigClass.ret;

  // A signature that needs no coercion keeps its types, but still needs its
  // attributes, such as the extension of a narrow integer result.
  auto applyCABIAttrs = [&] {
    for (auto [paramIdx, attrs] :
         llvm::enumerate(sigClass.getParamAttrs(origArgTypes, origRetType))) {
      for (NamedAttribute attr : attrs)
        func.setArgAttr(paramIdx, attr.getName(), attr.getValue());
    }
    for (NamedAttribute attr : sigClass.getResultAttrs(ctx))
      func.setResultAttr(0, attr.getName(), attr.getValue());
  };

  bool anyArgCoercion = llvm::any_of(
      argInfos, [](const CoercionInfo &ci) { return !ci.isIdentity(); });
  if (!anyArgCoercion && retInfo.isIdentity()) {
    applyCABIAttrs();
    return;
  }

  auto ptrType = LLVMPointerType::get(ctx);
  LLVMFunctionType newFnTy = sigClass.getFunctionType(
      ctx, origArgTypes, origRetType, func.getFunctionType().isVarArg());

  if (!func.isExternal()) {
    Block *entry = &func.getBody().front();
    ImplicitLocOpBuilder b(loc, entry, entry->begin());
    const LLVMDataLayout &dl = abiInfo.getDataLayout();

    // Step 1: Coerce arguments — replace original block args with C ABI args
    // and emit reconstruction code at the entry block top.
    SmallVector<BlockArgument> origArgs = llvm::to_vector(func.getArguments());
    auto reconstructed = reconstructCABIArguments(newFnTy, entry, sigClass,
                                                  origArgs, b, ctx, dl);
    for (auto [origArg, reconVal] : llvm::zip(origArgs, reconstructed))
      origArg.replaceAllUsesWith(reconVal);
    entry->eraseArguments(0, origArgTypes.size());

    if (!retInfo.isIdentity()) {
      applyCABIReturnCoercion(func, entry, retInfo, origRetType, b, ptrType,
                              ctx, loc, dl);
    }
  }

  func.setType(newFnTy);
  applyCABIAttrs();
}

} // namespace M::KGEN
