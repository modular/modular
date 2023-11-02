//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCALLINGCONVENTION
#define GEN_PASS_DEF_LOWERINPUTCONVENTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// LowerCallingConvention
//===----------------------------------------------------------------------===//

namespace {
struct LowerCallingConventionPass
    : KGEN::impl::LowerCallingConventionBase<LowerCallingConventionPass> {
  void runOnOperation() override;
};
} // namespace

static std::pair<bool, SmallVector<Type>> removeNoneTypes(TypeRange types) {
  SmallVector<Type> newTypes;
  for (Type type : types)
    if (!isa<KGEN::NoneType>(type))
      newTypes.push_back(type);
  return {newTypes.size() != types.size(), std::move(newTypes)};
}

/// Replace all the `!kgen.none` results in a signature.
static SignatureType removeNoneResults(SignatureType signature) {
  auto [anyNone, newResults] = removeNoneTypes(signature.getValueResults());
  // Micro-optimization: don't hash a new type if it won't change.
  if (!anyNone)
    return signature;
  return SignatureType::get(
      FunctionType::get(signature.getContext(), signature.getValueInputs(),
                        newResults),
      signature.getInputConventions(), signature.getFnEffects());
}

/// Remove none types from the results of debuginfo subroutine types as well.
static DebugInfo::DISubroutineType
removeDINoneResults(DebugInfo::DISubroutineType type) {
  // None types in the subroutine type will be wrapped in an unresolved type.
  SmallVector<DebugInfo::DIType> newTypes;
  for (DebugInfo::DIType type : type.getResultTypes()) {
    auto unresolved = dyn_cast<DebugInfo::DIUnresolvedMLIRType>(type);
    if (!unresolved || !isa<KGEN::NoneType>(unresolved.getType()))
      newTypes.push_back(type);
  }
  if (newTypes.size() == type.getResultTypes().size())
    return type;
  return DebugInfo::DISubroutineType::get(type.getContext(),
                                          type.getArgumentTypes(), newTypes);
}

static void rewriteCallingConventions(Operation &op) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(removeNoneResults);
  replacer.addReplacement(removeDINoneResults);

  auto walkFn = [&](Operation *op) {
    // Recursively replace all signatures in the operation. This will handle the
    // signatures of `kgen.func`, `kgen.extern.func`, `kgen.stage_closure`, and
    // `lit.async.execute`.
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                               /*replaceAttrs=*/true);

    // Handle exiting terminators.
    if (isa<ReturnOp, HLCF::YieldOp, HLCF::BreakOp>(op)) {
      // Remove none results.
      SmallVector<Value> newOperands;
      for (Value operand : op->getOperands())
        if (!isa<KGEN::NoneType>(operand.getType()))
          newOperands.push_back(operand);
      op->setOperands(newOperands);
      return;
    }

    // Handle `hlcf.if`, `hlcf.loop`, `kgen.call`, and `kgen.call_signature`.
    if (isa<HLCF::IfOp, HLCF::LoopOp, CallOp, CallSignatureOp>(op)) {
      auto [anyNone, newResults] = removeNoneTypes(op->getResultTypes());
      // Exit early if there are no none results.
      if (!anyNone)
        return;
      OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                           newResults);
      // Micro-optimization: set the DictionaryAttr directly to avoid a re-hash.
      state.attributes = op->getAttrDictionary();
      for (Region &region : op->getRegions())
        state.addRegion()->takeBody(region);
      Operation *newOp = OpBuilder(op).create(state);

      // Lazily construct a none constant only when needed.
      Value noneImpl;
      auto getNone = [&] {
        if (!noneImpl) {
          noneImpl = OpBuilder(op).create<ParamConstantOp>(
              op->getLoc(), NoneAttr::get(op->getContext()));
        }
        return noneImpl;
      };

      unsigned newResultIdx = 0;
      for (Value result : op->getResults()) {
        if (!isa<KGEN::NoneType>(result.getType()))
          result.replaceAllUsesWith(newOp->getResult(newResultIdx++));
        else if (!result.use_empty())
          result.replaceAllUsesWith(getNone());
      }
      assert(newResultIdx == newOp->getNumResults());
      op->erase();
      return;
    }

    // Handle `pop.coroutine.promise`. None types take up no space, so it is
    // safe to bitcast the promise pointer.
    if (auto promise = dyn_cast<POP::CoroutinePromiseOp>(op)) {
      auto [anyNone, newResults] =
          removeNoneTypes(cast<StructType>(promise.getType().getElementAsType())
                              .getParameterizedElementTypes());
      if (!anyNone)
        return;

      OpBuilder b(promise);
      Value newPromise = b.create<POP::CoroutinePromiseOp>(
          promise.getLoc(),
          PointerType::get(StructType::get(op->getContext(), newResults)),
          promise.getCoroutine());
      Value casted = b.create<POP::PointerBitcastOp>(
          promise.getLoc(), promise.getType(), newPromise);
      promise.replaceAllUsesWith(casted);
      promise.erase();
      return;
    }
  };
  op.walk(walkFn);
}

void LowerCallingConventionPass::runOnOperation() {
  mlir::parallelForEach(&getContext(), getOperation().getOps(),
                        rewriteCallingConventions);
}

//===----------------------------------------------------------------------===//
// LowerInputConventions
//===----------------------------------------------------------------------===//

namespace {
struct LowerInputConventionsPass
    : KGEN::impl::LowerInputConventionsBase<LowerInputConventionsPass> {
  void runOnOperation() override;
};
} // namespace

/// Return whether the given type needs could input convention lowering.
static bool shouldLower(Type type) {
  // Only pointer types should be lowered.
  auto argPtr = dyn_cast<PointerType>(type);
  if (!argPtr)
    return false;

  // StructType carries passability information directly.
  if (auto structType = dyn_cast<StructType>(argPtr.getElementAsType()))
    return !structType.getIsMemoryOnly();

  // We must be dealing with something register passable (e.g. index).
  return true;
}

/// Return whether the given type with the given convention needs lowering.
static bool shouldLower(Type type, ValueInputConvention convention) {
  if (convention == ValueInputConvention::BorrowedInMem ||
      convention == ValueInputConvention::OwnedInMem ||
      convention == ValueInputConvention::ByRefResult ||
      convention == ValueInputConvention::InitSelf)
    return shouldLower(type);
  return false;
}

/// Return the given signature with the argument conventions lowered. If no
/// lowering is necessary, the function returns null.
static SignatureType lowerSignature(SignatureType sig) {
  ArrayRef<ValueInputConvention> oldConvs = sig.getInputConventions();
  SmallVector<ValueInputConvention> newConvs(oldConvs);
  bool needsLowering = false;
  for (auto [idx, argTy, convention] :
       llvm::enumerate(sig.getValueInputs(), oldConvs)) {
    if (shouldLower(argTy, convention)) {
      needsLowering = true;
      newConvs[idx] = ValueInputConvention::None;
    }
  }
  if (!needsLowering)
    return {};
  return SignatureType::get(sig.getContext(), sig.getInputParamTypes(),
                            sig.getResultParamTypes(), sig.getValues(),
                            newConvs, sig.getFnEffects(), sig.getMetadata());
}

/// Lower the input conventions for a KGEN::CallOp if needed.
static void lowerCallOp(Operation *op) {
  auto callOp = dyn_cast<CallOp>(op);
  if (!callOp)
    return;
  if (SignatureType newSig = lowerSignature(callOp.getCalleeSignature())) {
    callOp.setCalleeAttr(
        SymbolConstantAttr::get(callOp.getCallee().getSymbol(), newSig));
  }
}

static void lowerInputConventions(Operation &op) {
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    if (SignatureType newSig = lowerSignature(funcOp.getSignature()))
      funcOp.setSignature(newSig);
    op.walk(lowerCallOp);
  }
}

void LowerInputConventionsPass::runOnOperation() {
  // TODO(#20700): implement the actual lowering instead of just nuking the
  // conventions.
  mlir::parallelForEach(&getContext(), getOperation().getOps(),
                        lowerInputConventions);
}
