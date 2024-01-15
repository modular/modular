//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines logic that reasons about value and memory object lifetimes:
// what an operation defines and consumes.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LifetimeTrackable
//===----------------------------------------------------------------------===//

LifetimeTrackable::LifetimeTrackable(Value v) {
  if (!v) // Null value isn't tracked.
    return;

  // LetReg starts out initialized with its own value.
  if (auto letReg = v.getDefiningOp<LetRegDeclOp>()) {
    name = letReg.getNameAttr();
    isIndirect = false;
    startsUninit = true; // Initialized at its definition point.
    endsUninit = true;
    return;
  }
  // VarLetDeclOp is uninit and ends that way.
  if (auto varLet = v.getDefiningOp<VarLetDeclOp>()) {
    name = varLet.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endsUninit = true;
    return;
  }

  // Global variable references start and end initialzied.
  if (auto globalRef = v.getDefiningOp<GlobalVarRefOp>()) {
    // FIXME: The global variable's name is attached to the symbol op.
    name = StringAttr::get(v.getContext(), "(global variable)");
    isIndirect = true;
    startsUninit = false;
    endsUninit = false;
    return;
  }

  if (v.getDefiningOp<LoadConsumeOp>() ||
      v.getDefiningOp<LIT::StructCreateOp>() ||
      v.getDefiningOp<ParamMaterializeOp>() ||
      v.getDefiningOp<VariantCreateOp>()) {
    name = StringAttr::get(v.getContext(), "(anonymous value)");
    isIndirect = false;
    startsUninit = true;
    endsUninit = true;
    return;
  }

  // The lit.ownership.end_lifetime op ends a register/mem lifetime and creates
  // a new one.  This defines the properties of its new lifetime.
  if (auto endLifetime = v.getDefiningOp<OwnershipEndLifetimeOp>()) {
    name = StringAttr::get(v.getContext(), "(transferred value)");
    isIndirect = !endLifetime.getIsReg();
    startsUninit = true;
    endsUninit = true;
    return;
  }

  // The lit.ref.from_pointer op takes an lifetime-tracked reference.  We
  // unconditionally model this as same liveness on entry to the function as on
  // exit, because some control flow paths may never execute the operation.
  //
  // When the op is executed to take ownership of the raw pointer,
  // CheckLifetimes will notice its actual effect: if it is init on entry and
  // uninit on exit, CheckLifetimes will ensure the value gets consumed or
  // destroyed.
  if (auto refFromPtr = v.getDefiningOp<RefFromPointerOp>()) {
    name = StringAttr::get(v.getContext(), "(reference value)");
    isIndirect = true;
    startsUninit = endsUninit = refFromPtr.getEndUninit();
    return;
  }

  // HandleVariantOp is currently always idiomatically for throwing function
  // results and owns the result.  When it is generalized for pattern matching
  // etc, it should take a discriminator to indicate what the result is.
  if (v.getDefiningOp<HandleVariantOp>()) {
    name = StringAttr::get(v.getContext(), "(call result)");
    isIndirect = false;
    startsUninit = true;
    endsUninit = true;
    return;
  }

  // VariantTakeOp starts out initialized with its own value.
  if (auto letReg = v.getDefiningOp<VariantTakeOp>()) {
    name = StringAttr::get(v.getContext(), "(call result)");
    isIndirect = false;
    startsUninit = true; // Initialized at its definition point.
    endsUninit = true;
    return;
  }

  /// Owned results of function calls are tracked as being initialized when
  /// defined but needing to be destroyed by the end of function.
  if (OpResult res = dyn_cast<OpResult>(v)) {
    if (auto call = dyn_cast<KGENCallOpInterface>(res.getOwner())) {
      if (call.getCalleeType().hasOwnedRegisterResult()) {
        name = StringAttr::get(v.getContext(), "(call result)");
        isIndirect = false;
        startsUninit = true;
        endsUninit = true;
      }
    }
    if (auto call = dyn_cast<LIT::CallSignatureOp>(res.getOwner())) {
      if (call.getCallee().getType().hasOwnedRegisterResult()) {
        name = StringAttr::get(v.getContext(), "(call result)");
        isIndirect = false;
        startsUninit = true;
        endsUninit = true;
      }
    }
  }

  // If this is a function argument, check to see what ownership it has.
  auto bbArg = dyn_cast<BlockArgument>(v);
  if (!bbArg || !bbArg.getOwner())
    return;
  Operation *parentOp = bbArg.getOwner()->getParentOp();
  /// FIXME: https://github.com/modularml/modular/issues/21818
  if (auto tryBlock = dyn_cast<LIT::TryOp>(parentOp)) {
    // Except blocks own the error instance(s) they accept.
    // We assume that except blocks only have error typed arguments.
    if (bbArg.getOwner() == &tryBlock.getExceptRegion().front()) {
      isIndirect = false;
      startsUninit = true;
      endsUninit = true;
      name = StringAttr::get(v.getContext(), "(error argument # " +
                                                 Twine(bbArg.getArgNumber()) +
                                                 ")");
      return;
    }
  }

  auto func = dyn_cast<LIT::FuncOp>(parentOp);
  if (!func)
    return;
  LITSignatureType signature = func.getSignature();

  unsigned argIdx = bbArg.getArgNumber();
  switch (signature.getInputConvention(argIdx)) {
  case ValueInputConvention::BorrowedInReg:
    // This is immutable so don't need to be tracked.
    return;

  case ValueInputConvention::BorrowedInMem:
    // This is actually a register value passed to VariadicListMem - it
    // doesn't need to be tracked.
    if (isa<VariadicType>(bbArg.getType()))
      return;

    // Borrowed memory objects don't need lifetime tracking given that they are
    // immutable, but we do want to reason about their aliasing properties for
    // return slot optimization etc.
    isIndirect = true;
    startsUninit = false;
    endsUninit = false;
    break;

  case ValueInputConvention::OwnedInReg:
    // TODO(#21861): support variadic arguments
    assert(!isa<VariadicType>(bbArg.getType()) &&
           "variadic OwnedInMem not supported yet");
    isIndirect = false;
    startsUninit = false;
    endsUninit = true;
    break;
  case ValueInputConvention::OwnedInMem:
    // TODO(#21861): support variadic arguments
    assert(!isa<VariadicType>(bbArg.getType()) &&
           "variadic OwnedInMem not supported yet");
    isIndirect = true;
    startsUninit = false;
    endsUninit = true;
    break;
  case ValueInputConvention::ByRefResult:
    // __result__ slots in raising functions do not properly model the behavior
    // when an error is thrown, so we don't track them here, they get special
    // support in CheckLifetimes.
    if (signature.isThrows())
      return;
    isIndirect = true;
    startsUninit = true;
    endsUninit = false;
    break;
  case ValueInputConvention::InitSelf:
    // Unlike byref-result, we allow memberwise initialization of 'self' in an
    // init method to construct a full value.
    isIndirect = true;
    startsUninit = true;
    endsUninit = false;
    isFullObjectLiveOnEntry = true;
    break;
  case ValueInputConvention::ByRef:
    // TODO(#21861): support variadic inout arguments
    assert(!isa<VariadicType>(bbArg.getType()) &&
           "variadic inout not supported yet");
    isIndirect = true;
    startsUninit = false;
    endsUninit = false;
    break;
  case ValueInputConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  ArrayRef<StringAttr> argNames = signature.getArgNames();
  name = argNames[argIdx];
  if (name.empty()) {
    name = StringAttr::get(v.getContext(), "(positional-only argument # " +
                                               Twine(argIdx) + ")");
  }
}

/// This constructor checks to see if the value is trackable or a field of a
/// trackable.  If so it identifies the underlying object being referenced. If
/// not, this returns a null value.
Value LifetimeTrackable::findUnderlyingValueFromField(Value value) {
  // If there are any GEP operations into the struct, dig through them.
  bool hadGEP = false;

  while (true) {
    if (auto structGER = value.getDefiningOp<RefStructGEROp>()) {
      hadGEP = true;
      value = structGER.getContainer();
    } else if (auto rebindOp = value.getDefiningOp<RebindOp>()) {
      value = rebindOp.getOperand();
    } else if (auto immut = value.getDefiningOp<RefImmutOp>()) {
      value = immut.getOperand();
    } else {
      break;
    }
  }

  // Check if there is a base value.
  LifetimeTrackable result(value);
  // If we had a GEP of this value but it doesn't have indirect storage, then
  // we aren't actually tracking the pointers off this value field sensitively,
  // so we can't be confident about what is going on with it.
  if (!result || (hadGEP && !result.isIndirect))
    return Value();
  // Otherwise, use whatever we found.
  return value;
}

//===----------------------------------------------------------------------===//
// OperationValueEffects
//===----------------------------------------------------------------------===//

/// This computes the effects that an operation has on any operands and result
/// values. This information is used by both phases of CheckLifetimes.
OverallOpValueEffect LIT::getOperationValueEffects(
    Operation &op, SmallVectorImpl<OperandEffect> &operands,
    SmallVectorImpl<ResultEffect> &results,
    SmallVectorImpl<std::pair<LifetimeAccess, TypedAttr>> &lifetimes) {
  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.
  if (isa<DebugInfo::ValueOp>(op)) {
    operands.push_back(OperandEffect::ignore);
    return {};
  }

  // These ops are handled specially.
  if (isa<RefStructGEROp, RebindOp, RefImmutOp>(op)) {
    operands.push_back(OperandEffect::ignore);
    results.push_back(ResultEffect::ignore);
    return {};
  }

  // RefStore consumes its operand and transfers it into the result.
  if (isa<LIT::RefStoreOp>(op)) {
    operands.append({OperandEffect::regConsume, OperandEffect::memStoreOwned});
    return {};
  }

  // A load is a use of whatever fields are being referenced.  If this is the
  // /last/ use of a value, emit a destructor of that value.  LoadOps are used
  // to model a /borrow/ of the underlying value, so they don't define a new
  // value.
  if (isa<RefLoadOp>(op)) {
    operands.push_back(OperandEffect::memLoad);
    results.push_back(ResultEffect::ignore);
    return {};
  }

  if (isa<OwnershipUseOp>(op)) {
    if (isa<RefType>(op.getOperand(0).getType()))
      operands.push_back(OperandEffect::memLoad);
    else
      operands.push_back(OperandEffect::regUse);
    return {};
  }

  // These operations consume their operands and define a result.
  if (isa<LoadConsumeOp, LIT::StructCreateOp, ParamMaterializeOp>(op)) {
    results.push_back(ResultEffect::regDefine);
    if (isa<LoadConsumeOp>(op))
      operands.push_back(OperandEffect::memConsume);
    else
      operands.resize(op.getNumOperands(), OperandEffect::regConsume);
    return {};
  }

  // lit.ownership.deflvalue is like an inout use of the pointer.
  if (isa<OwnershipDefLValueOp>(op)) {
    operands.push_back(OperandEffect::memInOut);
    return {};
  }

  // lit.ownership.end_lifetime consumes its operand then defines its result.
  if (auto ownershipEnd = dyn_cast<OwnershipEndLifetimeOp>(op)) {
    bool isIndirect = !ownershipEnd.getIsReg();
    operands.push_back(isIndirect ? OperandEffect::memConsume
                                  : OperandEffect::regConsume);
    results.push_back(isIndirect ? ResultEffect::memDefineInitToUninit
                                 : ResultEffect::regDefine);
    return {};
  }

  // lit.letreg.decl defines its own value after using its operand.
  // kgen.variant.create/kgen.variant.take consumes and produces an owned value.
  if (isa<LetRegDeclOp, VariantCreateOp, VariantTakeOp>(op)) {
    operands.push_back(OperandEffect::regConsume);
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  // RefFromPointerOp creates a new lifetime tracked value.  The 'startsUninit'
  // field impacts the execution of the operation (now), not its modeling at
  // start of the function.  We have to assume its liveness at start of function
  // is the same as its liveness at end of function because not all control
  // flow paths will execute the operation.
  if (auto refFromPtr = dyn_cast<RefFromPointerOp>(op)) {
    operands.push_back(OperandEffect::ignore); // Ignore the pointer input.
    ResultEffect effect;
    if (refFromPtr.getStartUninit()) {
      effect = refFromPtr.getEndUninit() ? ResultEffect::memDefineUninitToUninit
                                         : ResultEffect::memDefineUninitToInit;
    } else {
      effect = refFromPtr.getEndUninit() ? ResultEffect::memDefineInitToUninit
                                         : ResultEffect::memDefineInitToInit;
    }
    results.push_back(effect);
    return {};
  }

  // A yield from a HandleVariantOp consumes the operand.
  if (isa<YieldOp>(op)) {
    operands.push_back(OperandEffect::regConsume);
    return {};
  }

  if (isa<GlobalVarRefOp>(op)) {
    results.push_back(ResultEffect::memDefineInitToInit);
    return {};
  }

  // FIXME: CaptureListCreate is a CallSignatureOp's but not really calls?
  if (isa<CaptureListCreate>(op)) {
    // FIXME: Unclear how to handle the result of this.
    results.push_back(ResultEffect::ignore);
    return {};
  }

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (isa<LIT::CallSignatureOp, KGENCallOpInterface>(op)) {
    SignatureType signature;
    OperandRange callArguments = op.getOperands();
    ArrayRef<ValueInputConvention> conventions;
    if (auto directCall = dyn_cast<KGENCallOpInterface>(op)) {
      // These all have the callee as a parameter, not operand.
      signature = directCall.getCalleeType();
      conventions = signature.getInputConventions();

      // CreateClosureOp has a subset of the operands of a call.
      if (isa<CreateClosureOp>(op))
        conventions = conventions.take_front(op.getNumOperands());

      assert(conventions.size() == op.getNumOperands());
    } else {
      signature = cast<LIT::CallSignatureOp>(op).getCallee().getType();
      conventions = signature.getInputConventions();

      // We use the callee value, and process the rest as operands.
      operands.push_back(OperandEffect::regUse);
      assert(signature.getInputConventions().size() == op.getNumOperands() - 1);
      callArguments = callArguments.drop_front();
    }

    for (auto [convention, arg] : llvm::zip(conventions, callArguments)) {
      bool isIndirect = SignatureType::hasAddress(convention);
      switch (convention) {
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::OwnedInMem:
        operands.push_back(isIndirect ? OperandEffect::memConsume
                                      : OperandEffect::regConsume);
        break;
      case ValueInputConvention::BorrowedInReg:
      case ValueInputConvention::BorrowedInMem:
        operands.push_back(isIndirect ? OperandEffect::memLoad
                                      : OperandEffect::regUse);
        break;
      case ValueInputConvention::ByRef:
        operands.push_back(OperandEffect::memInOut);
        break;
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::InitSelf:
        operands.push_back(OperandEffect::memStoreOwned);
        break;
      case ValueInputConvention::None:
        llvm_unreachable("none convention not permitted in lit");
      }

      // If the accessed value is a kgen.variadic<> of references, notice the
      // extra lifetimes.
      // TODO: Generalize this beyond kgen.variadic.  We need to move the
      // access kind from the !lit.ref onto the lifetime though!
      Type argType = arg.getType();
      if (isIndirect) // Strip off the memory convention wrapper.
        argType = cast<RefType>(argType).getElementType();
      if (auto variadicType = dyn_cast<VariadicType>(argType)) {
        if (auto refType = dyn_cast<RefType>(variadicType.getElementType())) {
          // The callee is allowed to mutate the pointed-to value unless known
          // to be non-mut.
          auto accessType = refType.isMutableKnown(false)
                                ? LifetimeAccess::read
                                : LifetimeAccess::write;
          lifetimes.push_back({accessType, refType.getLifetime()});
        }
      }
    }

    // If the result is defining an owned register value, then we treat this as
    // a definition.
    if (signature.hasOwnedRegisterResult()) {
      results.push_back(ResultEffect::regDefine);
    } else if (op.getNumResults()) {
      assert(op.getNumResults() == 1);
      results.push_back(ResultEffect::ignore);
    }
    return {};
  }

  // A return consumes all the live-out values from the function.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp, KGEN::UnreachableOp>(op)) {
    // We always consume the result register - even if it is often trivial.
    operands.resize(op.getNumOperands(), OperandEffect::regConsume);
    return OverallOpValueEffect::terminatorOp;
  }

  if (isa<OwnershipMarkDestroyedOp>(op)) {
    operands.push_back(OperandEffect::memMarkDestroyed);
    return {};
  }

  // Local control flow ops.
  if (isa<HLCF::BreakOp, HLCF::ContinueOp>(op))
    return OverallOpValueEffect::localControlFlowOp;
  if (isa<LIT::TryRaiseOp>(op)) {
    operands.resize(op.getNumOperands(), OperandEffect::regConsume);
    return OverallOpValueEffect::localControlFlowOp;
  }

  // If-like operations.
  //
  // Note that we don't express the result definition of HandleVariant here.
  // that is because we need special processing in the dtor insertion pass.
  if (isa<HLCF::IfOp, ParamIfOp, HandleVariantOp>(op)) {
    // i1 value is never owned, and the markers are not used either.
    if (size_t num = op.getNumOperands())
      operands.append(num, OperandEffect::ignore);

    // TODO: IfOp could return an owned result.
    if (size_t num = op.getNumResults())
      results.resize(num, ResultEffect::ignore);
    return OverallOpValueEffect::ifLikeOp;
  }

  /// This is HLCF::LoopOp.
  if (isa<HLCF::LoopOp>(op))
    return OverallOpValueEffect::loopOp;

  if (isa<LIT::TryOp>(op))
    return OverallOpValueEffect::tryOp;

  return OverallOpValueEffect::unknownOp;
}
