//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERINPUTCONVENTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerInputConventionsPass
    : KGEN::impl::LowerInputConventionsBase<LowerInputConventionsPass> {
  void runOnOperation() override;
};
} // namespace

/// Return the lowered type for an in-memory passed argument. If lowering is not
/// needed, return null.
static Type lowerPointerType(Type type) {
  // Only pointer types should be lowered.
  auto argPtr = dyn_cast<PointerType>(type);
  if (!argPtr)
    return {};

  // We don't lower memory-only structs.
  Type elType = argPtr.getElementAsType();
  if (auto structType = dyn_cast<StructType>(elType))
    if (structType.getIsMemoryOnly())
      return {};

  // We must be dealing with something register passable (e.g. index).
  return elType;
}

/// Lowers the given signature if needed, and returns the non-result argument
/// indices (on the input signature) that needed to be changed. A flag is also
/// returned to indicate if the result of a signature with `byref_result` was
/// changed, in which case the new signature will no longer have that argument.
static std::tuple<SignatureType, SmallVector<size_t>, bool>
lowerSignature(SignatureType sig) {
  ArrayRef<ValueInputConvention> oldConvs = sig.getInputConventions();
  SmallVector<ValueInputConvention> newConvs(oldConvs);

  ArrayRef<Type> oldInputTypes = sig.getValueInputs();
  SmallVector<Type> newInputTypes(oldInputTypes);

  ArrayRef<Type> oldResTypes = sig.getValueResults();
  SmallVector<Type> newResTypes(oldResTypes);

  bool changedRes = false;
  SmallVector<size_t> changedIndices;
  for (auto [idx, argTy, convention] :
       llvm::enumerate(sig.getValueInputs(), oldConvs)) {
    if (convention == ValueInputConvention::BorrowedInMem ||
        convention == ValueInputConvention::OwnedInMem) {
      if (Type newArgTy = lowerPointerType(argTy)) {
        // Update the info needed for the new signature.
        newConvs[idx] = ValueInputConvention::None;
        newInputTypes[idx] = newArgTy;
        changedIndices.push_back(idx);
      }
    } else if (convention == ValueInputConvention::ByRefResult) {
      assert(oldResTypes.size() == 1);

      if (!isa<KGEN::NoneType>(oldResTypes[0]))
        continue;

      if (Type newArgTy = lowerPointerType(argTy)) {
        newResTypes[0] = newArgTy;
        changedRes = true;
      }
    }
  }

  SignatureType newSig;
  if (changedRes || !changedIndices.empty()) {
    auto newFnType = FunctionType::get(
        sig.getContext(), ArrayRef<Type>(newInputTypes).drop_front(changedRes),
        newResTypes);
    newSig = SignatureType::get(
        newFnType,
        ArrayRef<ValueInputConvention>(newConvs).drop_front(changedRes),
        sig.getFnEffects(), sig.getMetadata());
  }

  return std::make_tuple(newSig, std::move(changedIndices), changedRes);
}

/// Helper to perform the bulk of the lowering for `kgen.call` and
/// `kgen.call_signature` ops.
static std::pair<SignatureType, SmallVector<Value>>
lowerCallOpImpl(Operation *op, Operation::operand_range oldOperands,
                SignatureType oldSig) {
  auto [newSig, changedIndices, changedRes] = lowerSignature(oldSig);
  if (!newSig)
    return {newSig, {}};

  // Calculate the new operands, accounting for a potentially promoted result.
  ImplicitLocOpBuilder b(op->getLoc(), op);
  SmallVector<Value> newOperands(oldOperands.drop_front(changedRes));
  for (size_t idx : changedIndices)
    newOperands[idx] = b.create<POP::LoadOp>(oldOperands[idx]);

  // Update the result, if needed.
  if (changedRes) {
    b.setInsertionPointAfter(op);

    // First, we need to deal with anyone who might rely on the old result.
    OpResult res = op->getResult(0);
    // TODO(#24996): remove this when we have a verifier that checks this.
    assert(isa<KGEN::NoneType>(res.getType()));
    if (!res.use_empty()) {
      auto none = b.create<ParamConstantOp>(b.getAttr<NoneAttr>());
      res.replaceAllUsesWith(none);
    }

    // Then we store the new function result into the old byref result.
    res.setType(newSig.getValueResults()[0]);
    b.create<POP::StoreOp>(res, oldOperands[0]);
  }

  return {newSig, std::move(newOperands)};
}

/// Lower the input conventions for a KGEN::CallOp if needed.
static void lowerCallOp(CallOp callOp) {
  auto [newSig, newOperands] = lowerCallOpImpl(callOp, callOp.getOperands(),
                                               callOp.getCalleeSignature());
  if (!newSig)
    return;

  callOp->setOperands(newOperands);
  callOp.setCalleeAttr(
      SymbolConstantAttr::get(callOp.getCallee().getSymbol(), newSig));
}

/// Lower the input conventions for a KGEN::CallSignatureOp if needed.
static void lowerCallSignatureOp(CallSignatureOp callOp) {
  TypedValue<SignatureType> callee = callOp.getCallee();
  SignatureType oldSig = callee.getType();
  auto [newSig, newOperands] =
      lowerCallOpImpl(callOp, callOp.getArguments(), oldSig);
  if (!newSig)
    return;

  callOp->setOperands(1, oldSig.getNumInputs(), newOperands);
  callee.setType(newSig);
}

/// Lower the input conventions for a KGEN::FuncOp if needed.
static void lowerFuncOp(FuncOp funcOp) {
  SignatureType oldSig = funcOp.getSignature();
  auto [newSig, changedIndices, changedRes] = lowerSignature(oldSig);
  if (!newSig)
    return;

  Region &body = funcOp.getBodyRegion();
  auto b = OpBuilder::atBlockBegin(&body.front());
  for (size_t idx : changedIndices) {
    BlockArgument arg = body.getArgument(idx);
    auto ptr = b.create<POP::StackAllocationOp>(arg.getLoc(), arg.getType(), 1);
    auto storeOp = b.create<POP::StoreOp>(arg.getLoc(), arg, ptr);
    arg.setType(newSig.getValueInputs()[idx]);
    arg.replaceAllUsesExcept(ptr, storeOp);
  }

  if (changedRes) {
    BlockArgument byrefResArg = body.getArgument(0);
    auto newResPtr = b.create<POP::StackAllocationOp>(byrefResArg.getLoc(),
                                                      byrefResArg.getType(), 1);
    byrefResArg.replaceAllUsesWith(newResPtr);
    body.eraseArgument(0);

    Operation *returnOp = body.front().getTerminator();
    b.setInsertionPoint(returnOp);
    auto newRes = b.create<POP::LoadOp>(returnOp->getLoc(), newResPtr);
    returnOp->setOperand(0, newRes);
  }
  funcOp.setSignature(newSig);
}

static void lowerInputConventions(Operation &op) {
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    lowerFuncOp(funcOp);

    // Lower the ops in the function body.
    op.walk([](Operation *op) {
      if (auto callOp = dyn_cast<CallOp>(op))
        return lowerCallOp(callOp);
      if (auto callOp = dyn_cast<CallSignatureOp>(op))
        return lowerCallSignatureOp(callOp);
    });

    // We must do this in a second pass, otherwise ops like kgen.call_signature
    // would be difficult to identify for lowering (since their argument types
    // would be lowered already).
    op.walk([](Operation *op) {
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement([](SignatureType sig) {
        auto [newSig, _, __] = lowerSignature(sig);
        return newSig ? newSig : sig;
      });
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true,
                                 /*replaceAttrs=*/true);
    });
  }
}

void LowerInputConventionsPass::runOnOperation() {
  mlir::parallelForEach(&getContext(), getOperation().getOps(),
                        lowerInputConventions);
}
