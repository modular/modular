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

#include "CABICallHelpers.h"
#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace M::KGEN;

bool M::KGEN::isSignedIntegerScalar(Type type) {
  auto simd = dyn_cast<SIMDType>(type);
  if (!simd || simd.getResolvedSize() != 1)
    return false;
  std::optional<KGENDType> dtype = simd.getResolvedDType();
  return dtype && dtype->isSInt();
}

//===----------------------------------------------------------------------===//
// Coercion through memory
//===----------------------------------------------------------------------===//

Type M::KGEN::getCoercedValueType(const CoercionInfo &coercion,
                                  MLIRContext *ctx) {
  if (coercion.isTwoRegister()) {
    return LLVMStructType::getLiteral(
        ctx, {coercion.coercedType, coercion.coercedSecondType});
  }
  return coercion.coercedType;
}

Value M::KGEN::createCoercionSlot(OpBuilder &b, Location loc, LLVMFuncOp func,
                                  Type mojoType, Type coercedType,
                                  const LLVMDataLayout &dataLayout) {
  // The data layout gives a vector the alignment of its element, which can be
  // less than the LLVM alignment. Thus the slot has the coerced type, which
  // LLVM aligns, unless the Mojo type is larger.
  Type slotType = dataLayout.getTypeAllocSize(mojoType) >
                          dataLayout.getTypeAllocSize(coercedType)
                      ? mojoType
                      : coercedType;
  uint64_t align = std::max(dataLayout.getTypeABIAlign(mojoType),
                            dataLayout.getTypeABIAlign(coercedType));
  // Keep the alignment of the slot type when it is sufficient.
  if (align <= static_cast<uint64_t>(dataLayout.getTypeABIAlign(slotType)))
    align = 0;

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&func.getBody().front());
  Value count = ConstantOp::create(b, loc, b.getI32IntegerAttr(1));
  return AllocaOp::create(b, loc, LLVMPointerType::get(b.getContext()),
                          slotType, count, align);
}

Value M::KGEN::coerceThroughSlot(OpBuilder &b, Location loc, Value value,
                                 Type destType, Value slot) {
  StoreOp::create(b, loc, value, slot);
  return LoadOp::create(b, loc, destType, slot);
}

Value M::KGEN::coerceToC(OpBuilder &b, Location loc, LLVMFuncOp func,
                         Value value, Type coercedType,
                         const LLVMDataLayout &dataLayout) {
  Value slot = createCoercionSlot(b, loc, func, value.getType(), coercedType,
                                  dataLayout);
  return coerceThroughSlot(b, loc, value, coercedType, slot);
}

Value M::KGEN::coerceFromC(OpBuilder &b, Location loc, LLVMFuncOp func,
                           Value value, Type mojoType,
                           const LLVMDataLayout &dataLayout) {
  Value slot =
      createCoercionSlot(b, loc, func, mojoType, value.getType(), dataLayout);
  return coerceThroughSlot(b, loc, value, mojoType, slot);
}

//===----------------------------------------------------------------------===//
// CABICallHelper implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<CABIInfo> CABICallHelper::createABIHandler() const {
  const llvm::Triple &triple = tc->getTarget().getTriple();
  // POPToLLVMTypeConverter inherits from LLVMDataLayout
  const LLVMDataLayout &dataLayout = *tc;
  return M::KGEN::createCABIInfo(triple, ctx, dataLayout);
}

Value CABICallHelper::createEntryBlockAlloca(DirectRewriter &rewriter,
                                             Location loc,
                                             Type elemType) const {
  auto ptrType = LLVMPointerType::get(ctx);

  // Save current insertion point and emit alloca at the start of the
  // entry block.
  OpBuilder::InsertionGuard guard(rewriter);
  Block &entryBlock = getParentFunc().getBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);

  Value count =
      ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  return AllocaOp::create(rewriter, loc, ptrType, elemType, count,
                          /*alignment=*/0);
}

LLVMFuncOp CABICallHelper::getParentFunc() const {
  auto func = parentOp->getParentOfType<LLVMFuncOp>();
  assert(func && "external_call must be inside an LLVM function");
  return func;
}

SmallVector<Value> CABICallHelper::prepareArg(const CoercionInfo &coercion,
                                              Value originalArg, Location loc,
                                              DirectRewriter &rewriter) const {
  SmallVector<Value> result;

  if (coercion.isIdentity()) {
    // No coercion - pass directly
    result.push_back(squashPointlessCasts(originalArg));
    return result;
  }

  if (coercion.useIndirect) {
    // Allocate stack space, store struct, pass pointer
    Value alloca = createEntryBlockAlloca(rewriter, loc, originalArg.getType());
    StoreOp::create(rewriter, loc, originalArg, alloca);
    result.push_back(alloca);
    return result;
  }

  assert(coercion.coercedType && "non-identity, non-indirect coercion must "
                                 "have a coercedType — classifier bug");
  Value coerced = coerceToC(rewriter, loc, getParentFunc(), originalArg,
                            getCoercedValueType(coercion, ctx), *tc);
  if (!coercion.isTwoRegister()) {
    result.push_back(coerced);
    return result;
  }
  result.push_back(ExtractValueOp::create(rewriter, loc, coerced, 0));
  result.push_back(ExtractValueOp::create(rewriter, loc, coerced, 1));
  return result;
}

std::pair<SmallVector<Value>, Value>
CABICallHelper::buildCallArgs(ArrayRef<CoercionInfo> argClass,
                              const CoercionInfo &retClass,
                              ValueRange originalArgs, Type origRetTy,
                              Location loc, DirectRewriter &rewriter) const {
  SmallVector<Value> callArgs;
  Value sretPointer = nullptr;

  // Prepare sret pointer if needed
  if (retClass.useSRet) {
    sretPointer = createEntryBlockAlloca(rewriter, loc, origRetTy);
    callArgs.push_back(sretPointer);
  }

  // Prepare each argument
  for (auto [idx, coercion] : llvm::enumerate(argClass)) {
    SmallVector<Value> coercedArgs =
        prepareArg(coercion, originalArgs[idx], loc, rewriter);
    callArgs.append(coercedArgs.begin(), coercedArgs.end());
  }

  return {callArgs, sretPointer};
}

Value CABICallHelper::extractReturn(const CoercionInfo &retClass,
                                    Value callResult, Value sretPointer,
                                    Type origRetTy, Location loc,
                                    DirectRewriter &rewriter) const {
  if (retClass.useSRet) {
    // Load from sret pointer
    return LoadOp::create(rewriter, loc, origRetTy, sretPointer);
  }

  if (retClass.isIdentity()) {
    // No coercion - use directly
    return callResult;
  }

  assert(callResult.getType() == getCoercedValueType(retClass, ctx) &&
         "a register result must have the coerced type");
  return coerceFromC(rewriter, loc, getParentFunc(), callResult, origRetTy,
                     *tc);
}

CABICallPrep CABICallHelper::prepareCall(TypeRange argTypes,
                                         ValueRange argValues, Type origRetTy,
                                         Location loc, DirectRewriter &rewriter,
                                         size_t numFixedArgs,
                                         bool isVariadic) const {
  auto handler = createABIHandler();

  // Classification runs on LLVM types so register/layout rules see the same
  // shape the C compiler does.
  SmallVector<Type> llvmArgTypes;
  llvmArgTypes.reserve(argTypes.size());
  for (Type t : argTypes)
    llvmArgTypes.push_back(tc->convertType(t));
  Type llvmRetTy = origRetTy ? tc->convertType(origRetTy) : Type();

  SmallVector<bool> isArgSigned =
      llvm::map_to_vector(argTypes, isSignedIntegerScalar);
  SignatureClassification sigClass = handler->computeSignatureInfo(
      llvmArgTypes, llvmRetTy, loc, CABIRole::caller(isArgSigned),
      numFixedArgs);

  LLVMFunctionType sig = sigClass.getFunctionType(
      ctx, argValues.getTypes(), origRetTy, isVariadic, numFixedArgs);
  auto [callArgs, sretPtr] = buildCallArgs(sigClass.args, sigClass.ret,
                                           argValues, origRetTy, loc, rewriter);
  return {sig,       std::move(sigClass), std::move(llvmArgTypes),
          llvmRetTy, std::move(callArgs), sretPtr};
}

/// Return the number of operands `call` passes to its callee, including the
/// variadic arguments of a call to a variadic callee.
///
/// Note this is *not* `getArgOperands().size()`: llvm/llvm-project#214724,
/// changes `CallOpInterface::getArgOperands` to report only the operands
/// corresponding to the callee's declared parameters (excluding the variadic
/// tail).
static size_t getNumCallArgs(CallOp call) {
  // The callee is operand 0 for indirect calls (no callee symbol attribute).
  size_t numConsumed = call.getCallee().has_value() ? 0 : 1;
  return call.getCalleeOperands().size() - numConsumed;
}

Value CABICallHelper::finishCall(CallOp call, const CABICallPrep &prep,
                                 TailKind tailKind, Location loc,
                                 DirectRewriter &rewriter) const {
  const SignatureClassification &sigClass = prep.classification;

  // A direct call lowers from the call-site attributes, so they must be set
  // even when the declaration has them. The variadic tail is included: a
  // by-memory variadic argument needs its byval copy too.
  SmallVector<NamedAttrList> paramAttrs =
      sigClass.getParamAttrs(prep.argTypes, prep.retType);
  assert(paramAttrs.size() == getNumCallArgs(call) &&
         "C ABI classification and call operands disagree");
  if (llvm::any_of(paramAttrs,
                   [](const NamedAttrList &attrs) { return !attrs.empty(); }))
    call.setArgAttrsAttr(rewriter.getArrayAttr(llvm::map_to_vector(
        paramAttrs, [&](const NamedAttrList &attrs) -> Attribute {
          return attrs.getDictionary(ctx);
        })));
  assert(sigClass.getResultAttrs(ctx).empty() &&
         "Not expecting call-side result attrs");

  // A tail call frees the caller's frame before the callee runs. The sret
  // alloca and the copy behind a plain indirect argument live there; a
  // `byval` pointee does not, since the attribute makes the call sequence
  // copy it. See the requirements of `tail` in LLVM's LangRef.
  bool usesCallerFrame = sigClass.ret.useSRet ||
                         llvm::any_of(sigClass.args, [](const CoercionInfo &c) {
                           return c.useIndirect && !c.useByval;
                         });
  if (!usesCallerFrame)
    applyTailKind(call, tailKind);

  if (!prep.retType)
    return {};
  Value callResult = sigClass.ret.useSRet ? Value() : call.getResult();
  return extractReturn(sigClass.ret, callResult, prep.sretPtr, prep.retType,
                       loc, rewriter);
}
