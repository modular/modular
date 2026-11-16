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
#include "CABICallHelpers.h"
#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

/// For each original entry-block argument, add C ABI block arg(s) per
/// `sigClass` and emit reconstruction code that converts the C ABI value(s)
/// back to the original Mojo type. Returns one reconstructed Value per
/// original arg; the caller is responsible for replacing uses and erasing the
/// originals.
static SmallVector<Value>
reconstructCABIArguments(LLVMFunctionType newFnTy, LLVMFuncOp func,
                         const M::KGEN::SignatureClassification &sigClass,
                         ArrayRef<BlockArgument> origArgs,
                         ImplicitLocOpBuilder &b, MLIRContext *ctx,
                         const M::KGEN::LLVMDataLayout &dl) {
  SmallVector<Value> reconstructed;
  Block *entry = &func.getBody().front();
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
    } else {
      // One or two registers: make the C ABI value, then coerce it to the
      // original type through memory.
      Value coerced = entry->addArgument(paramTy, loc);
      assert(coerced.getType() == coercion.coercedType && "C ABI mismatch");
      if (coercion.isTwoRegister()) {
        Value second =
            entry->addArgument(newFnTy.getParamType(paramIdx + 1), loc);
        assert(second.getType() == coercion.coercedSecondType &&
               "C ABI mismatch");
        Value pair =
            UndefOp::create(b, M::KGEN::getCoercedValueType(coercion, ctx));
        pair = InsertValueOp::create(b, pair, coerced, size_t{0});
        coerced = InsertValueOp::create(b, pair, second, size_t{1});
      }
      reconstructed.push_back(
          M::KGEN::coerceFromC(b, loc, func, coerced, origType, dl));
    }
  }
  return reconstructed;
}

/// Make the C ABI result slot of a definition: for sret, the hidden result
/// pointer as block arg 0 (x8 on ARM64); otherwise, memory at the top of the
/// entry block through which the result is coerced.
static M::KGEN::CABIDefinition
createCABIResultSlot(LLVMFuncOp func, const M::KGEN::CoercionInfo &retInfo,
                     Type origRetType, ImplicitLocOpBuilder &b,
                     LLVMPointerType ptrType, MLIRContext *ctx, Location loc,
                     const M::KGEN::LLVMDataLayout &dl) {
  if (retInfo.useSRet) {
    Block *entry = &func.getBody().front();
    return {retInfo, entry->insertArgument(0u, ptrType, loc), Type()};
  }

  // One or two registers: coerce the result through memory.
  Type coercedType = M::KGEN::getCoercedValueType(retInfo, ctx);
  assert(coercedType);
  Value slot =
      M::KGEN::createCoercionSlot(b, loc, func, origRetType, coercedType, dl);
  return {retInfo, slot, coercedType};
}

} // namespace

namespace M::KGEN {

void rewriteCABIReturn(ReturnOp ret, const CABIDefinition &def) {
  if (!def.retSlot || ret.getOperands().empty())
    return;
  ImplicitLocOpBuilder rb(ret.getLoc(), ret);
  // sret: store the result through the hidden pointer and return void.
  if (def.ret.useSRet) {
    StoreOp::create(rb, ret.getArg(), def.retSlot);
    ret->setOperands(ValueRange());
    return;
  }
  ret->setOperands(ValueRange({coerceThroughSlot(
      rb, ret.getLoc(), ret.getArg(), def.coercedRetType, def.retSlot)}));
}

CABIDefinition rewriteCABIFunctionEntry(LLVMFuncOp func, CABIInfo &abiInfo,
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
  SmallVector<NamedAttrList> paramAttrs =
      sigClass.getParamAttrs(origArgTypes, origRetType, func.getArgAttrsAttr());
  auto applyCABIAttrs = [&] {
    func.setAllArgAttrs(llvm::map_to_vector(
        paramAttrs, [&](const NamedAttrList &attrs) -> Attribute {
          return attrs.getDictionary(ctx);
        }));
    for (NamedAttribute attr : sigClass.getResultAttrs(ctx))
      func.setResultAttr(0, attr.getName(), attr.getValue());
  };

  bool anyArgCoercion = llvm::any_of(
      argInfos, [](const CoercionInfo &ci) { return !ci.isIdentity(); });
  if (!anyArgCoercion && retInfo.isIdentity()) {
    applyCABIAttrs();
    return {retInfo, Value(), Type()};
  }

  auto ptrType = LLVMPointerType::get(ctx);
  LLVMFunctionType newFnTy = sigClass.getFunctionType(
      ctx, origArgTypes, origRetType, func.getFunctionType().isVarArg());

  CABIDefinition def{retInfo, Value(), Type()};
  if (!func.isExternal()) {
    Block *entry = &func.getBody().front();
    ImplicitLocOpBuilder b(loc, entry, entry->begin());
    const LLVMDataLayout &dl = abiInfo.getDataLayout();

    // Step 1: Coerce arguments — replace original block args with C ABI args
    // and emit reconstruction code at the entry block top.
    SmallVector<BlockArgument> origArgs = llvm::to_vector(func.getArguments());
    auto reconstructed =
        reconstructCABIArguments(newFnTy, func, sigClass, origArgs, b, ctx, dl);
    for (auto [origArg, reconVal] : llvm::zip(origArgs, reconstructed))
      origArg.replaceAllUsesWith(reconVal);
    entry->eraseArguments(0, origArgTypes.size());

    if (!retInfo.isIdentity()) {
      def = createCABIResultSlot(func, retInfo, origRetType, b, ptrType, ctx,
                                 loc, dl);
    }
  }

  func.setType(newFnTy);
  applyCABIAttrs();
  return def;
}

} // namespace M::KGEN
