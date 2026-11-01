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

  // Find the enclosing LLVM function.
  auto func = parentOp->getParentOfType<LLVMFuncOp>();
  assert(func && "external_call must be inside an LLVM function");

  // Save current insertion point and emit alloca at the start of the
  // entry block.
  OpBuilder::InsertionGuard guard(rewriter);
  Block &entryBlock = func.getBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);

  Value count =
      ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  return AllocaOp::create(rewriter, loc, ptrType, elemType, count,
                          /*alignment=*/0);
}

Value CABICallHelper::bitcastViaMemory(Value sourceValue, Type destType,
                                       Type allocType, Location loc,
                                       DirectRewriter &rewriter) const {
  Value alloca = createEntryBlockAlloca(rewriter, loc, allocType);
  StoreOp::create(rewriter, loc, sourceValue, alloca);
  return LoadOp::create(rewriter, loc, destType, alloca);
}

Value CABICallHelper::createOffsetGEP(Value basePtr, int64_t byteOffset,
                                      Location loc,
                                      DirectRewriter &rewriter) const {
  auto ptrType = LLVMPointerType::get(ctx);
  return GEPOp::create(rewriter, loc, ptrType, rewriter.getI8Type(), basePtr,
                       GEPArg(byteOffset), GEPNoWrapFlags::inbounds);
}

SmallVector<Value>
CABICallHelper::prepareTwoRegisterArgument(Value originalArg, Type firstType,
                                           Type secondType, Location loc,
                                           DirectRewriter &rewriter) const {
  // Allocate struct containing both register types
  Type allocType = LLVMStructType::getLiteral(ctx, {firstType, secondType});
  Value alloca = createEntryBlockAlloca(rewriter, loc, allocType);

  // Store original struct
  StoreOp::create(rewriter, loc, originalArg, alloca);

  // Load first register at offset 0
  Value first = LoadOp::create(rewriter, loc, firstType, alloca);

  // Load second register at the byte offset of the second struct field.
  // This equals getTypeStoreSize(firstType) because all implemented ABIs
  // guarantee firstType is exactly 8 bytes (i64 or f64 for System V AMD64;
  // i64 for ARM64 AAPCS non-HFA), so the LLVM struct {firstType, secondType}
  // never has alignment padding between the two slots.
  int64_t offset = tc->getTypeStoreSize(firstType);
  Value gep = createOffsetGEP(alloca, offset, loc, rewriter);
  Value second = LoadOp::create(rewriter, loc, secondType, gep);

  return {first, second};
}

Value CABICallHelper::handleTwoRegisterReturn(Value callResult, Type firstType,
                                              Type secondType,
                                              Type originalReturnType,
                                              Location loc,
                                              DirectRewriter &rewriter) const {
  // Allocate struct containing both register types
  Type allocType = LLVMStructType::getLiteral(ctx, {firstType, secondType});
  Value alloca = createEntryBlockAlloca(rewriter, loc, allocType);

  // Extract and store first register
  Value first = ExtractValueOp::create(rewriter, loc, callResult, 0);
  StoreOp::create(rewriter, loc, first, alloca);

  // Extract and store second register at the byte offset of the second field.
  // See prepareTwoRegisterArgument for why this offset is always correct.
  int64_t offset = tc->getTypeStoreSize(firstType);
  Value gep = createOffsetGEP(alloca, offset, loc, rewriter);
  Value second = ExtractValueOp::create(rewriter, loc, callResult, 1);
  StoreOp::create(rewriter, loc, second, gep);

  // Load as original struct type
  return LoadOp::create(rewriter, loc, originalReturnType, alloca);
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

  // Bitcast struct to integer/float type via store-to-stack + load pattern.
  //
  // Following Clang's approach: allocate using the coerced type(s) to
  // ensure the allocation is large enough for the loads. This prevents UB
  // when coercion rounds up (e.g., 3-byte struct -> i32 = 4 bytes).
  assert(coercion.coercedType && "non-identity, non-indirect coercion must "
                                 "have a coercedType — classifier bug");

  if (coercion.isTwoRegister()) {
    assert(coercion.coercedSecondType &&
           "two-register classification must have coercedSecondType");
    // Two registers: prepare both register values
    auto values =
        prepareTwoRegisterArgument(originalArg, coercion.coercedType,
                                   coercion.coercedSecondType, loc, rewriter);
    result.append(values.begin(), values.end());
  } else {
    // Single register: use bitcast helper
    Value coercedVal = bitcastViaMemory(originalArg, coercion.coercedType,
                                        coercion.coercedType, loc, rewriter);
    result.push_back(coercedVal);
  }

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

  if (retClass.isTwoRegister()) {
    assert(retClass.coercedType && retClass.coercedSecondType &&
           "two-register classification must have both coerced types");
    // Two-register return: extract both values and reconstruct struct
    return handleTwoRegisterReturn(callResult, retClass.coercedType,
                                   retClass.coercedSecondType, origRetTy, loc,
                                   rewriter);
  }

  // Single-register: bitcast from coerced type back to struct
  assert(retClass.coercedType &&
         "non-identity, non-sret, non-two-register return must have a "
         "coercedType — classifier bug");
  return bitcastViaMemory(callResult, origRetTy, retClass.coercedType, loc,
                          rewriter);
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
