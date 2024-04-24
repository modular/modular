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
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
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

  // VarDeclOp is uninit and ends that way.
  if (auto varDecl = v.getDefiningOp<VarDeclOp>()) {
    // Implicit temporaries get names like "*anonymous", don't print that!
    if (varDecl.getKind() == VarDeclKind::Synthesized)
      name = StringAttr::get(v.getContext(), "(expression temporary)");
    else
      name = varDecl.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endInitState = EndsUninit;
    return;
  }

  // Global variable references start and end initialzied.
  if (auto globalRef = v.getDefiningOp<GlobalVarRefOp>()) {
    // FIXME: The global variable's name is attached to the symbol op.
    name = StringAttr::get(v.getContext(), "(global variable)");
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsInit;
    return;
  }

  if (v.getDefiningOp<LoadConsumeOp>() ||
      v.getDefiningOp<LIT::StructCreateOp>() ||
      v.getDefiningOp<ParamMaterializeOp>()) {
    name = StringAttr::get(v.getContext(), "(anonymous value)");
    isIndirect = false;
    startsUninit = true;
    endInitState = EndsUninit;
    return;
  }

  // The lit.transfer_reg_ownership op ends a register lifetime and creates
  // a new one.
  if (auto endLifetime = v.getDefiningOp<TransferRegOwnershipOp>()) {
    name = StringAttr::get(v.getContext(), "(transferred value)");
    isIndirect = false;
    startsUninit = true;
    endInitState = EndsUninit;
    return;
  }

  // The lit.transfer_mem_ownership op ends a memory lifetime and creates
  // a new one.
  if (auto endLifetime = v.getDefiningOp<TransferMemOwnershipOp>()) {
    name = endLifetime.getParamDecl().getName();
    isIndirect = true;
    startsUninit = true;
    endInitState = EndsUninit;
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
    startsUninit = refFromPtr.getEndUninit();
    endInitState = startsUninit ? EndsUninit : EndsInit;
    return;
  }

  // This is a horrible hack for the REPL. :-(
  if (auto refFromPtr = v.getDefiningOp<RefFromPointerREPLOp>()) {
    name = refFromPtr.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnNormal;
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
        endInitState = EndsUninit;
      }
    }
    if (auto call = dyn_cast<LIT::CallIndirectOp>(res.getOwner())) {
      if (call.getCallee().getType().hasOwnedRegisterResult()) {
        name = StringAttr::get(v.getContext(), "(call result)");
        isIndirect = false;
        startsUninit = true;
        endInitState = EndsUninit;
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
      endInitState = EndsUninit;
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
  switch (signature.getArgConvention(argIdx)) {
  case ArgConvention::BorrowedInReg:
    // This is immutable so don't need to be tracked.
    return;

  case ArgConvention::BorrowedInMem:
    // Borrowed memory objects don't need lifetime tracking given that they are
    // immutable, but we do want to reason about their aliasing properties for
    // return slot optimization etc.
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsInit;
    break;

  case ArgConvention::OwnedInReg:
    isIndirect = false;
    startsUninit = false;
    endInitState = EndsUninit;
    break;
  case ArgConvention::OwnedInMem:
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsUninit;
    break;
  case ArgConvention::ByRefResult:
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnNormal;
    break;
  case ArgConvention::ByRefError:
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnError;
    break;
  case ArgConvention::InitSelf:
    // Unlike byref-result, we allow memberwise initialization of 'self' in an
    // init method to construct a full value.
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnNormal;
    isFullObjectLiveOnEntry = true;
    break;
  case ArgConvention::ByRef:
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsInit;
    break;
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  name = signature.getArgName(argIdx);
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

/// This is a helper for LIT::getOperationEffects split out since calls are so
/// interesting.
static void
getCallOpEffects(Operation &op,
                 SmallVectorImpl<std::pair<Value, OperandEffect>> &operands,
                 SmallVectorImpl<ResultEffect> &results,
                 SmallVectorImpl<TypedAttr> &lifetimes) {
  LITSignatureType signature;
  OperandRange callArguments = op.getOperands();
  ArrayRef<ArgConvention> conventions;
  size_t argIdxOffset = 0;

  if (auto directCall = dyn_cast<KGENCallOpInterface>(op)) {
    // These all have the callee as a parameter, not operand.
    signature = directCall.getCalleeType();
    conventions = signature.getArgConventions();

    // CreateClosureOp has a subset of the operands of a call.
    if (isa<CreateClosureOp>(op)) {
      conventions = conventions.take_front(op.getNumOperands());
      argIdxOffset = 1;
    }

    assert(conventions.size() == op.getNumOperands());
  } else {
    auto call = cast<LIT::CallIndirectOp>(op);
    signature = call.getCallee().getType();
    conventions = signature.getArgConventions();

    // We use the callee value, and process the rest as operands.
    operands.push_back({op.getOperand(0), OperandEffect::regUse});
    assert(signature.getArgConventions().size() == op.getNumOperands() - 1);
    callArguments = callArguments.drop_front();
    argIdxOffset = 1;
  }

  /// Argument conventions cause a direct use of the register of pointee, and
  /// handling them specifically allows us to be field sensitive in cases where
  /// the access is directly attributable to a Value.
  auto getOperandEffectForConvention =
      [throws = signature.isThrows()](ArgConvention conv) -> OperandEffect {
    switch (conv) {
    case ArgConvention::OwnedInReg:
      return OperandEffect::regConsume;
    case ArgConvention::OwnedInMem:
      return OperandEffect::memConsume;
    case ArgConvention::BorrowedInReg:
      return OperandEffect::regUse;
    case ArgConvention::BorrowedInMem:
      return OperandEffect::memLoad;
    case ArgConvention::ByRef:
      return OperandEffect::memInOut;
    case ArgConvention::ByRefError:
      return OperandEffect::memStoreConditional;
    case ArgConvention::ByRefResult:
    case ArgConvention::InitSelf:
      return throws ? OperandEffect::memStoreConditional
                    : OperandEffect::memStoreOwned;
    case ArgConvention::None:
      llvm_unreachable("none convention not permited in Mojo");
    }
    llvm_unreachable("invalid input convention");
  };

  SmallVector<Type> typesAccessibleByCallee;
  auto addArgument = [&](Value arg, ArgConvention conv) {
    // Get normal argument effect.
    auto effect = getOperandEffectForConvention(conv);
    Type argType = arg.getType();

    // If this is a borrowed register of a !lit.ref, then we know that this is
    // an explicitly declared low-level reference.
    // TODO(references): This is a hack because we can't get lifetime of self.
    bool isIndirect = SignatureType::hasAddress(conv);
    if (conv == ArgConvention::BorrowedInReg)
      if (auto ref = dyn_cast<RefType>(argType)) {
        effect = ref.isMutableKnown(false) ? OperandEffect::memLoad
                                           : OperandEffect::memInOut;
        isIndirect = true;
      }

    // If this is a normal register use, and if the value is a reference
    // (whether the argument convention is fancy or if it is an explicitly
    // passed reference) treat this as a field sensitive access so we can
    operands.push_back({arg, effect});

    // If this is a memConsume or memStoreOwned, then the lifetime of the
    // reference is handled directly, strip it off.  Otherwise handle borrowed,
    // inout, etc operands as just any-old reference use.
    if (isIndirect)
      argType = cast<RefType>(argType).getElementType();

    // In addition to the direct (field-sensitive) effect of loading/storing
    // the bits, the callee may do whatever it wants with lifetimes embedded
    // in the type.  Collect all of these so we can process them.
    typesAccessibleByCallee.push_back(argType);
  };

  for (auto [idx, arg, convention] :
       llvm::enumerate(callArguments, conventions)) {
    if (auto splat = arg.getDefiningOp<POP::VariadicSplatOp>()) {
      addArgument(splat.getOperand(), splat.getType().getConvention());
      continue;
    }

    // As a special hack, we directly handle the effects and "see through" a
    // pop.variadic.create, so we can model the effects of the variadic.create
    // instead of seeing abstract uses of the lifetimes.  This provides two
    // benefits:
    //   1) it allows us to reason about the varargs uses field-sensitively,
    //      e.g. you can pass `a.x` through varargs and `a.y` through an inout
    //      without the compiler imagining a conflict on "a" just like other
    //      arguments.
    //   2) given "direct" access information, it allows us to model 'owned'
    //      argument conventions which consume the operand, something lifetime
    //      accesses cannot model (because it requires field sensitivity).
    if (auto vararg = arg.getDefiningOp<POP::VariadicCreateOp>()) {
      auto varargConvention = vararg.getType().getConvention();
      for (auto varOperand : vararg.getOperands())
        addArgument(varOperand, varargConvention);
      continue;
    }

    if (auto pack = arg.getDefiningOp<RefPackCreateOp>()) {
      if (signature.isPackVarArg(idx + argIdxOffset)) {
        auto argConvention =
            signature.getPackVarArgConvention(idx + argIdxOffset);
        for (auto packOperand : pack.getOperands())
          addArgument(packOperand, argConvention);
        continue;
      }
    }

    addArgument(arg, convention);
  }

  // Look at the types accessible by the callee to see if there are any
  // lifetime accesses.
  for (auto [num, argType] : llvm::enumerate(typesAccessibleByCallee)) {
    auto dre = dyn_cast<DeclRefType>(argType);
    if (!dre)
      continue;

    for (auto paramValue : dre.getParamValues()) {
      // If the type captured a lifetime, the callee may touch the location
      // with the mutability of the target access.
      if (isa<LifetimeType>(paramValue.getType()))
        lifetimes.push_back(paramValue);
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
}

/// This computes the effects that an operation has on any operands and result
/// values. This information is used by both phases of CheckLifetimes.
OverallOpValueEffect LIT::getOperationEffects(
    Operation &op, SmallVectorImpl<std::pair<Value, OperandEffect>> &operands,
    SmallVectorImpl<ResultEffect> &results,
    SmallVectorImpl<TypedAttr> &lifetimes) {
  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.  These indexing operations are handled specially.
  if (isa<DebugInfo::ValueOp, DebugInfo::KillOp, RefStructGEROp, RebindOp,
          RefImmutOp>(op)) {
    if (op.getNumResults() == 1)
      results.push_back(ResultEffect::ignore);
    return {};
  }

  /// When all of the operands of an instruction have an effect and they are
  /// in a fixed order, this helper can help specify them.
  auto setOperandEffects = [&](ArrayRef<OperandEffect> effects) {
    assert(effects.size() == op.getNumOperands() && "operand count mismatch");
    for (auto [operand, effect] : llvm::zip(op.getOperands(), effects))
      operands.push_back({operand, effect});
  };

  // RefStore consumes its operand and transfers it into the result.
  if (isa<LIT::RefStoreOp>(op)) {
    setOperandEffects(
        {OperandEffect::regConsume, OperandEffect::memStoreOwned});
    return {};
  }

  // A load is a use of whatever fields are being referenced.  If this is
  // the /last/ use of a value, emit a destructor of that value.  LoadOps
  // are used to model a /borrow/ of the underlying value, so they don't
  // define a new value.
  if (auto load = dyn_cast<RefLoadOp>(op)) {
    operands.push_back({load.getOperand(), OperandEffect::memLoad});
    results.push_back(ResultEffect::ignore);
    return {};
  }
  if (auto load = dyn_cast<LoadConsumeOp>(op)) {
    operands.push_back({load.getOperand(), OperandEffect::memConsume});
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  if (auto use = dyn_cast<OwnershipUseOp>(op)) {
    auto effect = isa<RefType>(op.getOperand(0).getType())
                      ? OperandEffect::memLoad
                      : OperandEffect::regUse;
    operands.push_back({use.getOperand(), effect});
    return {};
  }

  // These ops consume their operands, struct.create and param.materialize
  // define a result.
  if (isa<LIT::StructCreateOp, ParamMaterializeOp>(op)) {
    for (Value o : op.getOperands())
      operands.push_back({o, OperandEffect::regConsume});
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  // lit.ownership.deflvalue is like an inout use of the ref.
  if (auto deflvalue = dyn_cast<OwnershipDefLValueOp>(op)) {
    operands.push_back({deflvalue.getOperand(), OperandEffect::memInOut});
    return {};
  }

  // lit.transfer_reg_ownership consumes its operand then defines its result.
  if (auto transfer = dyn_cast<TransferRegOwnershipOp>(op)) {
    operands.push_back({transfer.getOperand(), OperandEffect::regConsume});
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  // lit.transfer_mem_ownership consumes its operand then defines its result.
  if (auto transfer = dyn_cast<TransferMemOwnershipOp>(op)) {
    operands.push_back({transfer.getOperand(), OperandEffect::memConsume});
    results.push_back(ResultEffect::memDefineInitToUninit);
    return {};
  }

  // RefFromPointerOp creates a new lifetime tracked value.  The
  // 'startsUninit' field impacts the execution of the operation (now), not
  // its modeling at start of the function.  We have to assume its liveness
  // at start of function is the same as its liveness at end of function
  // because not all control flow paths will execute the operation.
  if (auto refFromPtr = dyn_cast<RefFromPointerOp>(op)) {
    // Ignore the pointer input.
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

  if (isa<GlobalVarRefOp>(op)) {
    results.push_back(ResultEffect::memDefineInitToInit);
    return {};
  }

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (isa<LIT::CallIndirectOp, KGENCallOpInterface>(op)) {
    getCallOpEffects(op, operands, results, lifetimes);
    return {};
  }

  // A return consumes all the live-out values from the function.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp, KGEN::UnreachableOp>(op)) {
    // We always consume the result register - even if it is often trivial.
    for (auto o : op.getOperands())
      operands.push_back({o, OperandEffect::regConsume});
    return OverallOpValueEffect::terminatorOp;
  }

  if (auto mark = dyn_cast<OwnershipMarkInitializedOp>(op)) {
    operands.push_back({mark.getOperand(), OperandEffect::memStoreOwned});
    return {};
  }

  if (auto mark = dyn_cast<OwnershipMarkDestroyedOp>(op)) {
    operands.push_back({mark.getOperand(), OperandEffect::memMarkDestroyed});
    return {};
  }

  // Local control flow ops.
  if (isa<HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp>(op))
    return OverallOpValueEffect::localControlFlowOp;

  // If-like operations.
  if (isa<ParamIfOp>(op)) {
    // i1 value is never owned, and the markers are not used either.
    // TODO: IfOp could return an owned result.
    if (size_t num = op.getNumResults())
      results.resize(num, ResultEffect::ignore);
    return OverallOpValueEffect::ifLikeOp;
  }

  if (isa<HLCF::ElifOp, HLCF::SwitchOp, HLCF::IfOp>(op)) {
    if (size_t num = op.getNumResults())
      results.resize(num, ResultEffect::ignore);
    return OverallOpValueEffect::acyclicControlFlowNodeOp;
  }

  /// This is HLCF::LoopOp.
  if (isa<HLCF::LoopOp>(op))
    return OverallOpValueEffect::loopOp;

  if (isa<LIT::TryOp>(op))
    return OverallOpValueEffect::tryOp;

  return OverallOpValueEffect::unknownOp;
}
