//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
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
  Type elType = argPtr.getElementType();
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
    } else if (convention == ValueInputConvention::ByRefResult ||
               convention == ValueInputConvention::InitSelf) {
      Type loweredByrefResTy = lowerPointerType(argTy);
      if (!loweredByrefResTy)
        continue;
      changedRes = true;

      Type oldResType = oldResTypes[0];
      if (sig.isThrows()) {
        // If the function is throwing, replace the success type in the variant.
        auto resVariant = cast<VariantType>(oldResType);
        assert(resVariant.getNumTypes() == 2);
        SmallVector<Type> variantTypes(resVariant.getTypes());
        variantTypes[1] = loweredByrefResTy;
        newResTypes[0] = VariantType::get(variantTypes);
      } else {
        // If the function doesn't throw, we will return the lowered type.
        newResTypes[0] = loweredByrefResTy;
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
    newOperands[idx - changedRes] = b.create<POP::LoadOp>(oldOperands[idx]);

  // Update the result, if needed.
  if (changedRes) {
    b.setInsertionPointAfter(op);

    OpResult res = op->getResult(0);
    if (newSig.isThrows()) {
      Type oldVariantTy = cast<KGEN::VariantType>(res.getType());

      // We create an HCLF::IfOp, with a condition checking if there is no error
      // (i.e. the then branch will handle normal return). Users of the old
      // result (e.g. error handling) will now take the result of this IfOp.
      auto cond = b.create<VariantIsOp>(res, 1);
      auto ifOp = b.create<HLCF::IfOp>(oldVariantTy, cond);
      res.replaceAllUsesExcept(ifOp.getResult(0), cond);
      res.setType(newSig.getValueResults()[0]);

      // Populate the then branch (normal return).
      Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
      b.setInsertionPointToStart(thenBlock);
      auto resVal = b.create<VariantTakeOp>(res, 1);
      b.create<POP::StoreOp>(resVal, oldOperands[0]);

      auto none = b.create<ParamConstantOp>(b.getAttr<NoneAttr>());
      Value thenRes = b.create<VariantCreateOp>(oldVariantTy, none, 1);
      b.create<HLCF::YieldOp>(thenRes);

      // Populate the else branch (error return).
      Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
      b.setInsertionPointToStart(elseBlock);
      auto err = b.create<VariantTakeOp>(res, 0);
      Value elseRes = b.create<VariantCreateOp>(oldVariantTy, err, 0);
      b.create<HLCF::YieldOp>(elseRes);
    } else {
      // If the callee doesn't throw, we simply make every use take a new none.
      if (!res.use_empty()) {
        auto none = b.create<ParamConstantOp>(b.getAttr<NoneAttr>());
        res.replaceAllUsesWith(none);
      }

      // Then just store the new callee result into the old byref result.
      res.setType(newSig.getValueResults()[0]);
      b.create<POP::StoreOp>(res, oldOperands[0]);
    }
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

/// Emit IR for repacking the returned variant in the body of a throwing
/// function that we are currently lowering. This returns the new variant result
/// of the give type `newVariantTy`.
static Value repackFuncVariantResult(ReturnOp returnOp,
                                     VariantType newVariantTy,
                                     POP::StackAllocationOp newResPtr) {
  Value oldRetVal = returnOp.getOperand(0);
  ImplicitLocOpBuilder b(returnOp.getLoc(), returnOp);

  // We check the result is coming from. If we can guarantee that it's either an
  // error or not, we can just repack the error or the valid result.
  if (auto variantCreateOp =
          dyn_cast_or_null<VariantCreateOp>(oldRetVal.getDefiningOp())) {
    size_t idx = variantCreateOp.getIndex();
    if (idx == 1) {
      // This is guaranteed to be a normal return.
      auto loadedRes = b.create<POP::LoadOp>(newResPtr);
      return b.create<VariantCreateOp>(newVariantTy, loadedRes, 1);
    }

    // This is guaranteed to be an error return.
    assert(idx == 0 && "unexpected variant type creation");
    Value err = variantCreateOp.getOperand();
    return b.create<VariantCreateOp>(newVariantTy, err, 0);
  }

  // We can't guarantee what the result is, so we emit conditional variant
  // repacking. We create an HCLF::IfOp, with a condition checking if there is
  // no error (i.e. the then branch will handle normal return). The result of
  // this IfOp is what we will return.
  auto cond = b.create<VariantIsOp>(oldRetVal, 1);
  auto ifOp = b.create<HLCF::IfOp>(newVariantTy, cond);
  Value newRetVal = ifOp.getResult(0);

  // Populate the then branch (normal return).
  Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
  b.setInsertionPointToStart(thenBlock);
  auto loadedRes = b.create<POP::LoadOp>(newResPtr);
  Value thenRes = b.create<VariantCreateOp>(newVariantTy, loadedRes, 1);
  b.create<HLCF::YieldOp>(thenRes);

  // Populate the else branch (error return).
  Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
  b.setInsertionPointToStart(elseBlock);
  auto err = b.create<VariantTakeOp>(oldRetVal, 0);
  Value elseRes = b.create<VariantCreateOp>(newVariantTy, err, 0);
  b.create<HLCF::YieldOp>(elseRes);

  return newRetVal;
}

/// Lower the input conventions for a KGEN::FuncOp if needed.
static void lowerFuncOp(FuncOp funcOp) {
  SignatureType oldSig = funcOp.getSignature();
  auto [newSig, changedIndices, changedRes] = lowerSignature(oldSig);
  if (!newSig)
    return;

  // Argument locations do not have subprogram scopes. If we have debuginfo,
  // make sure to add it.
  DebugInfo::DISubprogramAttr spAttr = funcOp.getSubprogramScope();
  auto addDI = [&](Location loc) -> Location {
    if (!spAttr)
      return loc;
    return FusedLoc::get(loc.getContext(), loc, spAttr);
  };

  Region &body = funcOp.getBodyRegion();
  auto b = OpBuilder::atBlockBegin(&body.front());
  for (size_t idx : changedIndices) {
    BlockArgument arg = body.getArgument(idx);
    Location loc = addDI(arg.getLoc());
    auto ptr = b.create<POP::StackAllocationOp>(loc, arg.getType(), 1);
    auto storeOp = b.create<POP::StoreOp>(loc, arg, ptr);
    arg.setType(newSig.getValueInputs()[idx - changedRes]);
    arg.replaceAllUsesExcept(ptr, storeOp);
  }

  if (changedRes) {
    BlockArgument byrefResArg = body.getArgument(0);
    auto newResPtr = b.create<POP::StackAllocationOp>(
        addDI(byrefResArg.getLoc()), byrefResArg.getType(), 1);
    byrefResArg.replaceAllUsesWith(newResPtr);
    body.eraseArgument(0);

    // Find all return sites in the function and rewrite them.
    body.walk([&, newSig = newSig](ReturnOp returnOp) {
      b.setInsertionPoint(returnOp);

      if (newSig.isThrows()) {
        // If the function throws, we need to potentially unpack and repack the
        // result/error variant.
        auto newVariantTy = cast<VariantType>(newSig.getValueResults()[0]);
        Value newRetVal =
            repackFuncVariantResult(returnOp, newVariantTy, newResPtr);
        returnOp.setOperand(0, newRetVal);
      } else {
        // If the function doesn't throw, we just load and return the new
        // result.
        auto newRes = b.create<POP::LoadOp>(returnOp.getLoc(), newResPtr);
        returnOp.setOperand(0, newRes);
      }
    });
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
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([](SignatureType sig) {
      auto [newSig, _, __] = lowerSignature(sig);
      return newSig ? newSig : sig;
    });
    auto anyRegTypeType = TypeType::get(op.getContext());
    replacer.addReplacement([&](TypeConstantAttr type) {
      // Canonicalize metatypes.
      return TypeConstantAttr::get(type.getValue(), anyRegTypeType);
    });
    op.walk([&](Operation *op) {
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
