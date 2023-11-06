//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
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
static Type lowerOwnedOrBorrowedInMem(Type type) {
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

/// Lowers the given signature if needed, and returns the argument indices that
/// needed to be changed.
static std::pair<SignatureType, SmallVector<size_t>>
lowerSignature(SignatureType sig) {
  ArrayRef<ValueInputConvention> oldConvs = sig.getInputConventions();
  SmallVector<ValueInputConvention> newConvs(oldConvs);

  ArrayRef<Type> oldInputTypes = sig.getValueInputs();
  SmallVector<Type> newInputTypes(oldInputTypes);

  SmallVector<size_t> changedIndices;
  for (auto [idx, argTy, convention] :
       llvm::enumerate(sig.getValueInputs(), oldConvs)) {
    if (convention == ValueInputConvention::BorrowedInMem ||
        convention == ValueInputConvention::OwnedInMem) {
      if (Type newArgTy = lowerOwnedOrBorrowedInMem(argTy)) {
        // Update the info needed for the new signature.
        newConvs[idx] = ValueInputConvention::None;
        newInputTypes[idx] = newArgTy;
        changedIndices.push_back(idx);
      }
    }
  }

  if (changedIndices.empty())
    return std::make_pair(SignatureType(), std::move(changedIndices));
  auto newFnType =
      FunctionType::get(sig.getContext(), newInputTypes, sig.getValueResults());
  auto newSig = SignatureType::get(newFnType, newConvs, sig.getFnEffects(),
                                   sig.getMetadata());
  return std::make_pair(newSig, std::move(changedIndices));
}

/// Lower the input conventions for a KGEN::CallOp if needed.
static void lowerCallOp(CallOp callOp) {
  auto [newSig, changedIndices] = lowerSignature(callOp.getCalleeSignature());
  if (!newSig)
    return;

  OpBuilder b(callOp);
  SmallVector<Value> newOperands(callOp.getOperands());
  for (size_t idx : changedIndices) {
    newOperands[idx] =
        b.create<POP::LoadOp>(callOp.getLoc(), callOp.getOperand(idx));
  }
  callOp->setOperands(newOperands);
  callOp.setCalleeAttr(
      SymbolConstantAttr::get(callOp.getCallee().getSymbol(), newSig));
}

/// Lower the input conventions for a KGEN::CallSignatureOp if needed.
static void lowerCallSignatureOp(CallSignatureOp callOp) {
  TypedValue<SignatureType> callee = callOp.getCallee();
  auto [newSig, changedIndices] = lowerSignature(callee.getType());
  if (!newSig)
    return;

  OpBuilder b(callOp);
  SmallVector<Value> newOperands(callOp.getArguments());
  for (size_t idx : changedIndices) {
    newOperands[idx] =
        b.create<POP::LoadOp>(callOp.getLoc(), callOp.getArguments()[idx]);
  }
  callOp->setOperands(1, newOperands.size(), newOperands);
  callee.setType(newSig);
}

/// Lower the input conventions for a KGEN::FuncOp if needed.
static void lowerFuncOp(FuncOp funcOp) {
  SignatureType oldSig = funcOp.getSignature();
  auto [newSig, changedIndices] = lowerSignature(oldSig);
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
        auto [newSig, _] = lowerSignature(sig);
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
