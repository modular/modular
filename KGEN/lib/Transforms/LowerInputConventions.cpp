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
