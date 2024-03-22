//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass performs the lowering of argument input conventions of concrete
// functions. This pass must run before inlining, but after elaboration. This
// pass will:
//
// 1. Move register passable types passed as `{owned,borrowed}_in_mem` to be
//    passed in register.
// 2. TODO(#20700): Promote register passable `byref_result` and `init_self`
//    arguments to function results.
//    - TODO(#20700): This also handles functions that throw.
// 3. TODO(#20700):Sets all argument conventions to `none`, i.e. only `none`
//    conventions are legal after this in the pipeline.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERARGCONVENTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerArgConventionsPass
    : KGEN::impl::LowerArgConventionsBase<LowerArgConventionsPass> {
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
static std::tuple<SignatureType, SmallVector<size_t>, SmallVector<Type>, bool,
                  bool, bool>
lowerSignature(SignatureType sig) {
  ArrayRef<ArgConvention> oldConvs = sig.getArgConventions();
  SmallVector<ArgConvention> newConvs(oldConvs);

  ArrayRef<Type> oldInputTypes = sig.getArguments();
  SmallVector<Type> newInputTypes(oldInputTypes);

  ArrayRef<Type> oldResTypes = sig.getResults();
  SmallVector<Type> newResTypes(oldResTypes);

  bool changedInitSelf = false, changedByRefResult = false,
       changedByRefError = false;
  SmallVector<size_t> changedIndices;
  for (auto [idx, argTy, convention] :
       llvm::enumerate(sig.getArguments(), oldConvs)) {
    if (convention == ArgConvention::BorrowedInMem ||
        convention == ArgConvention::OwnedInMem) {
      if (Type newArgTy = lowerPointerType(argTy)) {
        // Update the info needed for the new signature.
        newConvs[idx] = ArgConvention::None;
        newInputTypes[idx] = newArgTy;
        changedIndices.push_back(idx);
      }
    } else if (convention == ArgConvention::ByRefResult ||
               convention == ArgConvention::ByRefError ||
               convention == ArgConvention::InitSelf) {
      Type loweredByrefResTy = lowerPointerType(argTy);
      if (!loweredByrefResTy)
        continue;
      if (convention == ArgConvention::ByRefResult)
        changedByRefResult = true;
      else if (convention == ArgConvention::ByRefError)
        changedByRefError = true;
      else
        changedInitSelf = true;

      if (sig.isThrows()) {
        // If the function is throwing, append the result type.
        newResTypes.push_back(loweredByrefResTy);
        // Make sure the error type comes first.
        if (convention == ArgConvention::ByRefError && changedInitSelf)
          std::swap(newResTypes[1], newResTypes[2]);
        // If both the error and the result type are register-passable, then
        // return a `!kgen.variant<ErrT, ValT>`.
        if (changedByRefError && (changedInitSelf || changedByRefResult)) {
          newResTypes.assign(
              1, VariantType::get({newResTypes[1], newResTypes[2]}));
        }
      } else {
        // If the function doesn't throw, we will return the lowered type.
        newResTypes[0] = loweredByrefResTy;
      }
    }
  }

  SignatureType newSig;
  if (changedByRefResult || changedByRefError || changedInitSelf ||
      !changedIndices.empty()) {
    auto newInputTypesAR = ArrayRef<Type>(newInputTypes);
    auto newConvAR = ArrayRef<ArgConvention>(newConvs);
    if (changedInitSelf) {
      // Drop the first argument.
      newInputTypesAR = newInputTypesAR.drop_front();
      newConvAR = newConvAR.drop_front();
    }
    if (changedByRefResult) {
      // Drop the last argument.
      newInputTypesAR = newInputTypesAR.drop_back();
      newConvAR = newConvAR.drop_back();
    }
    if (changedByRefError) {
      // Drop the last argument.
      newInputTypesAR = newInputTypesAR.drop_back();
      newConvAR = newConvAR.drop_back();
      // If we had but didn't change the byref_result, then we need to drop the
      // second-last argument. Just swap the elements in the underlying vector.
      if (!changedByRefResult && sig.hasMemoryOnlyResult()) {
        std::swap(newInputTypes.end()[-2], newInputTypes.end()[-1]);
        std::swap(newConvs.end()[-2], newConvs.end()[-1]);
      }
    }

    auto newFnType =
        FunctionType::get(sig.getContext(), newInputTypesAR, newResTypes);
    newSig = SignatureType::get(newFnType, newConvAR, sig.getFnEffects(),
                                sig.getMetadata());
  }

  return std::make_tuple(newSig, std::move(changedIndices),
                         std::move(newResTypes), changedInitSelf,
                         changedByRefResult, changedByRefError);
}

/// Helper to perform the bulk of the lowering for `kgen.call` and
/// `kgen.call_indirect` ops.
static void lowerCallOpImpl(
    Operation *op, Operation::operand_range oldOperands, SignatureType oldSig,
    function_ref<void(Operation *, SignatureType, ValueRange)> updateArgs) {
  auto [newSig, changedIndices, newResTypes, changedInitSelf,
        changedByRefResult, changedByRefError] = lowerSignature(oldSig);
  if (!newSig)
    return;

  // Calculate the new operands, accounting for a potentially promoted result.
  ImplicitLocOpBuilder b(op->getLoc(), op);
  SmallVector<Value> newOperands(
      oldOperands.drop_front(changedInitSelf).drop_back(changedByRefResult));
  if (changedByRefError)
    newOperands.erase(
        std::prev(newOperands.end(),
                  1 + (!changedByRefResult && oldSig.hasMemoryOnlyResult())));
  for (size_t idx : changedIndices) {
    newOperands[idx - changedInitSelf] =
        b.create<POP::LoadOp>(oldOperands[idx]);
  }

  // Now update the result, if needed.
  if (changedInitSelf || changedByRefResult || changedByRefError) {
    b.setInsertionPointAfter(op);

    OpResult res = op->getResult(0);
    if (newSig.isThrows()) {
      // If the callee throws and both error and result were rewritten into a
      // variant, then we have to extract the relevant values from the variant.
      if (changedByRefError && (changedByRefResult || changedInitSelf)) {
        // Replace the i1 with a variant check.
        res.setType(newSig.getResults()[0]);
        auto isError = b.create<VariantIsOp>(res, 0);
        res.replaceAllUsesExcept(isError, isError);

        auto ifOp = b.create<HLCF::IfOp>(isError);
        b.createBlock(&ifOp.getThenRegion());
        b.create<POP::StoreOp>(
            b.create<VariantTakeOp>(res, 0),
            oldOperands.end()[-1 - oldSig.hasMemoryOnlyResult()]);
        b.create<HLCF::YieldOp>();

        b.createBlock(&ifOp.getElseRegion());
        b.create<POP::StoreOp>(
            b.create<VariantTakeOp>(res, 1),
            oldOperands[changedInitSelf ? 0 : oldOperands.size() - 1]);
        b.create<HLCF::YieldOp>();
      } else {
        // In this case, we need to reallocate the operation with a different
        // number of results.
        OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                             newResTypes);
        state.attributes = op->getAttrDictionary();
        Operation *newOp = b.create(state);
        res.replaceAllUsesWith(newOp->getResult(0));

        // Store the relevant result in the branch in which it is known to have
        // a valid value.
        auto ifOp = b.create<HLCF::IfOp>(newOp->getResult(0));
        Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
        b.create<HLCF::YieldOp>();
        Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
        b.create<HLCF::YieldOp>();
        b.setInsertionPointToStart(changedByRefError ? thenBlock : elseBlock);
        b.create<POP::StoreOp>(
            newOp->getResult(1),
            changedByRefError
                ? oldOperands.end()[-1 - oldSig.hasMemoryOnlyResult()]
                : oldOperands[changedInitSelf ? 0 : oldOperands.size() - 1]);
        op->erase();
        op = newOp;
      }
    } else {
      // If the callee doesn't throw, we simply make every use take a new none.
      if (!res.use_empty()) {
        auto none = b.create<ParamConstantOp>(b.getAttr<NoneAttr>());
        res.replaceAllUsesWith(none);
      }

      // Then just store the new callee result into the old byref result.
      res.setType(newSig.getResults()[0]);
      b.create<POP::StoreOp>(
          res, oldOperands[changedInitSelf ? 0 : oldOperands.size() - 1]);
    }
  }

  // Update the callee type and the operands.
  updateArgs(op, newSig, newOperands);
}

/// Lower the input conventions for a KGEN::CallOp if needed.
static void lowerCallOp(CallOp callOp) {
  lowerCallOpImpl(
      callOp, callOp.getOperands(), callOp.getCalleeSignature(),
      [](Operation *op, SignatureType newSig, ValueRange newOperands) {
        auto callOp = cast<CallOp>(op);
        callOp->setOperands(newOperands);
        callOp.setCalleeAttr(
            SymbolConstantAttr::get(callOp.getCallee().getSymbol(), newSig));
      });
}

/// Lower the input conventions for a KGEN::CallIndirectOp if needed.
static void lowerCallIndirectOp(CallIndirectOp callOp) {
  SignatureType oldSig = callOp.getCallee().getType();
  lowerCallOpImpl(
      callOp, callOp.getArguments(), oldSig,
      [&oldSig](Operation *op, SignatureType newSig, ValueRange newOperands) {
        auto callOp = cast<CallIndirectOp>(op);
        callOp->setOperands(1, oldSig.getNumArguments(), newOperands);
        callOp.getCallee().setType(newSig);
      });
}

/// Emit IR for repacking the returned variant in the body of a throwing
/// function that we are currently lowering. This returns the new variant result
/// of the give type `newVariantTy`.
static Value repackFuncVariantResult(ReturnOp returnOp,
                                     VariantType newVariantTy, Value newResPtr,
                                     Value newErrPtr) {
  Value oldRetVal = returnOp.getOperand(0);
  ImplicitLocOpBuilder b(returnOp.getLoc(), returnOp);

  // We check the result is coming from. If we can guarantee that it's either an
  // error or not, we can just repack the error or the valid result.
  BoolAttr isError;
  if (mlir::matchPattern(oldRetVal, mlir::m_Constant(&isError))) {
    if (!isError.getValue()) {
      // This is guaranteed to be a normal return.
      return b.create<VariantCreateOp>(newVariantTy,
                                       b.create<POP::LoadOp>(newResPtr), 1);
    }
    // This is guaranteed to be an error return.
    return b.create<VariantCreateOp>(newVariantTy,
                                     b.create<POP::LoadOp>(newErrPtr), 0);
  }

  // We can't guarantee what the result is, so we emit conditional variant
  // repacking. We create an HCLF::IfOp, with a condition checking if there is
  // no error (i.e. the then branch will handle normal return). The result of
  // this IfOp is what we will return.
  auto ifOp = b.create<HLCF::IfOp>(newVariantTy, oldRetVal);

  // Populate the then branch (normal return).
  Block *thenBlock = b.createBlock(&ifOp.getThenRegion());
  b.setInsertionPointToStart(thenBlock);
  Value thenRes = b.create<VariantCreateOp>(
      newVariantTy, b.create<POP::LoadOp>(newErrPtr), 0);
  b.create<HLCF::YieldOp>(thenRes);

  // Populate the else branch (error return).
  Block *elseBlock = b.createBlock(&ifOp.getElseRegion());
  b.setInsertionPointToStart(elseBlock);
  Value elseRes = b.create<VariantCreateOp>(
      newVariantTy, b.create<POP::LoadOp>(newResPtr), 1);
  b.create<HLCF::YieldOp>(elseRes);

  return ifOp.getResult(0);
}

/// Lower the input conventions for a KGEN::FuncOp if needed.
static void lowerFuncOp(FuncOp funcOp) {
  SignatureType oldSig = funcOp.getSignature();
  auto [newSig, changedIndices, newResTypes, changedInitSelf,
        changedByRefResult, changedByRefError] = lowerSignature(oldSig);
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
    arg.setType(newSig.getArguments()[idx - changedInitSelf]);
    arg.replaceAllUsesExcept(ptr, storeOp);
  }

  Value newResPtr, newErrPtr;
  if (changedByRefError) {
    BlockArgument byrefResArg =
        body.getArguments().end()[-1 - oldSig.hasMemoryOnlyResult()];
    newErrPtr = b.create<POP::StackAllocationOp>(addDI(byrefResArg.getLoc()),
                                                 byrefResArg.getType(), 1);
    byrefResArg.replaceAllUsesWith(newErrPtr);
    body.eraseArgument(byrefResArg.getArgNumber());
  }
  if (changedInitSelf || changedByRefResult) {
    size_t argNumber = changedInitSelf ? 0 : body.getArguments().size() - 1;
    BlockArgument byrefResArg = body.getArgument(argNumber);
    newResPtr = b.create<POP::StackAllocationOp>(addDI(byrefResArg.getLoc()),
                                                 byrefResArg.getType(), 1);
    byrefResArg.replaceAllUsesWith(newResPtr);
    body.eraseArgument(argNumber);
  }

  if (changedInitSelf || changedByRefResult || changedByRefError) {
    // Find all return sites in the function and rewrite them.
    body.walk([&, newSig = newSig, changedByRefError = changedByRefError,
               changedInitSelf = changedInitSelf,
               changedByRefResult = changedByRefResult](ReturnOp returnOp) {
      b.setInsertionPoint(returnOp);

      // If the function doesn't throw, we just load and return the new
      // result.
      if (!newSig.isThrows()) {
        auto newRes = b.create<POP::LoadOp>(returnOp.getLoc(), newResPtr);
        returnOp.setOperand(0, newRes);
        return;
      }

      // If the function throws and we rewrote both the error and the
      // byref_result, we need to potentially unpack and repack the
      // result/error variant.
      if (changedByRefError && (changedInitSelf || changedByRefResult)) {
        auto newVariantTy = cast<VariantType>(newSig.getResults()[0]);
        Value newRetVal = repackFuncVariantResult(returnOp, newVariantTy,
                                                  newResPtr, newErrPtr);
        returnOp.setOperand(0, newRetVal);
        return;
      }

      // Otherwise, we load either the error or the result, depending on which
      // got rewritten.
      Value toLoad = changedByRefError ? newErrPtr : newResPtr;
      assert(toLoad && "should have been rewritten");
      Value newRes = b.create<POP::LoadOp>(returnOp.getLoc(), toLoad);
      returnOp->insertOperands(1, newRes);
    });
  }
  funcOp.setSignature(newSig);
}

static void lowerArgConventions(Operation &op) {
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    lowerFuncOp(funcOp);

    // Lower the ops in the function body.
    op.walk([](Operation *op) {
      if (auto callOp = dyn_cast<CallOp>(op))
        return lowerCallOp(callOp);
      if (auto callOp = dyn_cast<CallIndirectOp>(op))
        return lowerCallIndirectOp(callOp);
    });

    // We must do this in a second pass, otherwise ops like kgen.call_indirect
    // would be difficult to identify for lowering (since their argument types
    // would be lowered already).
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([](SignatureType sig) {
      SignatureType newSig = std::get<0>(lowerSignature(sig));
      return newSig ? newSig : sig;
    });
    auto metatype = TypeType::get(op.getContext());
    replacer.addReplacement([&](TypeConstantAttr type) {
      // Canonicalize metatypes.
      return TypeConstantAttr::get(type.getValue(), metatype);
    });
    op.walk([&](Operation *op) {
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true, /*replaceAttrs=*/true);
    });
  }
}

void LowerArgConventionsPass::runOnOperation() {
  mlir::parallelForEach(&getContext(), getOperation().getOps(),
                        lowerArgConventions);
}
