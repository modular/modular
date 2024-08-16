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
// 2. Promote register passable `byref_result` and `init_self` arguments to
//    function results.
//    - This also handles functions that throw.
// 3. Sets all argument conventions to `none`, i.e. only `none`
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

namespace {
struct LoweredSignature {
  SignatureType newSig;
  SmallVector<size_t> changedIndices;
  SmallVector<Type> newResTypes;

  int valIdx = -1, errIdx = -1;
  /// This enum indicates whether a byref_result/init_self argument and/or a
  /// byref_error argument were promoted.
  enum ABI { Neither, ErrorOnly, ValueOnly, Both };
  int abiLowering = Neither;

  /// Drop elements in a vector corresponding to the original input signature's
  /// arguments.
  template <typename T>
  void dropOperandsFrom(SmallVectorImpl<T> &operands) {
    // Drop error or result arguments that were dropped, starting with the
    // greatermost so the indices remain valid.
    int eIdx = errIdx, vIdx = valIdx;
    if (eIdx > vIdx)
      std::swap(eIdx, vIdx);
    if (vIdx != -1)
      operands.erase(operands.begin() + vIdx);
    if (eIdx != -1)
      operands.erase(operands.begin() + eIdx);
  }

  unsigned mapOperandIndex(unsigned index) {
    if (valIdx != -1 && static_cast<unsigned>(valIdx) < index)
      return index - 1;
    return index;
  }

  bool isBoth() { return abiLowering == Both; }
};
} // namespace

/// Lowers the given signature if needed, and returns the non-result argument
/// indices (on the input signature) that needed to be changed. A flag is also
/// returned to indicate if the result of a signature with `byref_result` was
/// changed, in which case the new signature will no longer have that argument.
static LoweredSignature lowerSignature(SignatureType sig) {
  ArrayRef<ArgConvention> oldConvs = sig.getArgConventions();
  SmallVector<ArgConvention> newConvs(oldConvs);

  ArrayRef<Type> oldInputTypes = sig.getArguments();
  SmallVector<Type> newInputTypes(oldInputTypes);

  LoweredSignature s;
  Type errType, valType;
  s.newResTypes.assign(sig.getResults().begin(), sig.getResults().end());
  for (auto [idx, argTy, convention] :
       llvm::enumerate(sig.getArguments(), oldConvs)) {
    if (convention == ArgConvention::BorrowedInMem ||
        convention == ArgConvention::OwnedInMem) {
      if (Type newArgTy = lowerPointerType(argTy)) {
        // Update the info needed for the new signature.
        newConvs[idx] = convention == ArgConvention::OwnedInMem
                            ? ArgConvention::OwnedInReg
                            : ArgConvention::BorrowedInReg;
        newInputTypes[idx] = newArgTy;
        s.changedIndices.push_back(idx);
      }
      // Don't alter the result convention for async functions. The coroutine
      // lowering expects this ABI.
    } else if ((SignatureType::isResultSlot(convention) && !sig.isAsync()) ||
               convention == ArgConvention::InitSelf) {
      Type loweredByrefResTy = lowerPointerType(argTy);
      if (!loweredByrefResTy)
        continue;
      bool isError = convention == ArgConvention::ByRefError;
      s.abiLowering |=
          isError ? LoweredSignature::ErrorOnly : LoweredSignature::ValueOnly;
      (isError ? errType : valType) = loweredByrefResTy;
      (isError ? s.errIdx : s.valIdx) = idx;
    }
  }

  if (s.abiLowering != LoweredSignature::Neither) {
    if (!sig.isThrows()) {
      s.newResTypes[0] = valType;
    } else {
      // Make sure the error type always comes first.
      if (errType)
        s.newResTypes.push_back(errType);
      if (valType)
        s.newResTypes.push_back(valType);
      // If both are being rewritten, pack them into a variant.
      if (s.isBoth())
        s.newResTypes.assign(1, VariantType::get({errType, valType}));
    }
  }

  if (s.abiLowering != LoweredSignature::Neither || !s.changedIndices.empty()) {
    // Erase inout results promoted to register results from the argument list.
    s.dropOperandsFrom(newInputTypes);
    s.dropOperandsFrom(newConvs);

    auto newFnType =
        FunctionType::get(sig.getContext(), newInputTypes, s.newResTypes);
    s.newSig = SignatureType::get(newFnType, newConvs, sig.getFnEffects(),
                                  sig.getMetadata());
  }

  return s;
}

/// Helper to perform the bulk of the lowering for `kgen.call` and
/// `kgen.call_indirect` ops.
static void lowerCallOpImpl(
    Operation *op, Operation::operand_range oldOperands, SignatureType oldSig,
    function_ref<void(Operation *, SignatureType, ValueRange)> updateArgs) {
  LoweredSignature s = lowerSignature(oldSig);
  SignatureType newSig = s.newSig;
  if (!newSig)
    return;

  // Calculate the new operands, accounting for a potentially promoted result.
  ImplicitLocOpBuilder b(op->getLoc(), op);
  SmallVector<Value> newOperands(oldOperands);
  s.dropOperandsFrom(newOperands);
  for (size_t idx : s.changedIndices) {
    newOperands[s.mapOperandIndex(idx)] =
        b.create<POP::LoadOp>(oldOperands[idx]);
  }

  // Now update the result, if needed.
  if (s.abiLowering != LoweredSignature::Neither) {
    b.setInsertionPointAfter(op);

    OpResult res = op->getResult(0);
    if (newSig.isThrows()) {
      // If the callee throws and both error and result were rewritten into a
      // variant, then we have to extract the relevant values from the variant.
      if (s.isBoth()) {
        // Replace the i1 with a variant check.
        res.setType(newSig.getResults()[0]);
        auto isError = b.create<VariantIsOp>(res, 0);
        res.replaceAllUsesExcept(isError, isError);

        auto ifOp = b.create<HLCF::IfOp>(isError);
        b.createBlock(&ifOp.getThenRegion());
        b.create<POP::StoreOp>(b.create<VariantGetOp>(res, 0),
                               oldOperands[s.errIdx]);
        b.create<HLCF::YieldOp>();

        b.createBlock(&ifOp.getElseRegion());
        b.create<POP::StoreOp>(b.create<VariantGetOp>(res, 1),
                               oldOperands[s.valIdx]);
        b.create<HLCF::YieldOp>();
      } else {
        // In this case, we need to reallocate the operation with a different
        // number of results.
        OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                             s.newResTypes);
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
        bool errorOnly = s.abiLowering == LoweredSignature::ErrorOnly;
        b.setInsertionPointToStart(errorOnly ? thenBlock : elseBlock);
        b.create<POP::StoreOp>(newOp->getResult(1),
                               oldOperands[errorOnly ? s.errIdx : s.valIdx]);
        op->erase();
        op = newOp;
      }
    } else {
      // If the callee doesn't throw, we simply make every use take a new none.
      if (!res.use_empty()) {
        auto none = b.create<ParamConstantOp>(b.getAttr<NoneAttr>());
        res.replaceAllUsesWith(none);
      }

      // Then just store the new callee result into the old inout result.
      res.setType(newSig.getResults()[0]);
      b.create<POP::StoreOp>(res, oldOperands[s.valIdx]);
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
  LoweredSignature s = lowerSignature(oldSig);
  SignatureType newSig = s.newSig;
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
  for (size_t idx : s.changedIndices) {
    BlockArgument arg = body.getArgument(idx);
    Location loc = addDI(arg.getLoc());
    auto ptr = b.create<POP::StackAllocationOp>(loc, arg.getType(), 1);
    auto storeOp = b.create<POP::StoreOp>(loc, arg, ptr);
    arg.setType(newSig.getArguments()[s.mapOperandIndex(idx)]);
    arg.replaceAllUsesExcept(ptr, storeOp);
  }

  Value newResPtr, newErrPtr;
  auto dropAlloca = [&b, &body, &addDI](std::pair<Value *, int> pair) {
    if (pair.second == -1)
      return;
    BlockArgument arg = body.getArgument(pair.second);
    *pair.first =
        b.create<POP::StackAllocationOp>(addDI(arg.getLoc()), arg.getType());
    arg.replaceAllUsesWith(*pair.first);
    body.eraseArgument(pair.second);
  };
  std::pair<Value *, int> valPair{&newResPtr, s.valIdx},
      errPair{&newErrPtr, s.errIdx};
  if (errPair.second > valPair.second)
    std::swap(errPair, valPair);
  dropAlloca(valPair);
  dropAlloca(errPair);

  if (s.abiLowering != LoweredSignature::Neither) {
    // Find all return sites in the function and rewrite them.
    body.walk([&](ReturnOp returnOp) {
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
      if (s.isBoth()) {
        auto newVariantTy = cast<VariantType>(newSig.getResults()[0]);
        Value newRetVal = repackFuncVariantResult(returnOp, newVariantTy,
                                                  newResPtr, newErrPtr);
        returnOp.setOperand(0, newRetVal);
        return;
      }

      // Otherwise, we load either the error or the result, depending on which
      // got rewritten.
      Value toLoad = s.errIdx != -1 ? newErrPtr : newResPtr;
      assert(toLoad && "should have been rewritten");
      Value newRes = b.create<POP::LoadOp>(returnOp.getLoc(), toLoad);
      returnOp->insertOperands(1, newRes);
    });
  }
  funcOp.setSignature(newSig);
}

void LowerArgConventionsPass::runOnOperation() {
  FuncOp func = getOperation();
  lowerFuncOp(func);

  // Lower the ops in the function body.
  func.walk([](Operation *op) {
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
    SignatureType newSig = lowerSignature(sig).newSig;
    return newSig ? newSig : sig;
  });
  auto metatype = TypeType::get(&getContext());
  replacer.addReplacement([&](TypeConstantAttr type) {
    // Canonicalize metatypes.
    return TypeConstantAttr::get(type.getMlirType(), metatype);
  });
  func.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
  });
}
