//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines logic that reasons about value and memory object origins:
// what an operation defines and consumes.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/OriginTrackable.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Return true if the specified MLIR type is obviously a trivial register type.
static bool isTypeObviouslyTrivial(Type type) {
  return isa<KGEN::NoneType, IntegerType, RefType>(type);
}

//===----------------------------------------------------------------------===//
// OriginTrackable
//===----------------------------------------------------------------------===//

OriginTrackable::OriginTrackable(Value v) {
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

    // If this is a vardecl shadow of a register passable 'out' argument, then
    // the value is treated as if its whole-object bit is live on entry.  This
    // allows it to be fieldwise assigned.
    if (varDecl.getKind() == VarDeclKind::InitOutArg)
      isFullObjectLiveOnEntry = true;
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
      v.getDefiningOp<ParamMaterializeOp>()) {
    name = StringAttr::get(v.getContext(), "(anonymous value)");
    isIndirect = false;
    startsUninit = true;
    endInitState = EndsUninit;
    return;
  }

  // The lit.ref.from_pointer op takes an origin-tracked reference.  We
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
    // We have several different common cases, including:
    // 1) a trivial MLIR type that doesn't matter, or a register passable
    //    trivial type that we can track it for convenience.
    // 2) an non-trivial register passable (e.g. Arc) which we need to track
    //    as being defined by the call and needing to be consumed before the
    //    function exit.
    // 3) a ref-result type, which is tracked by origin tracking but isn't
    //    owned.  We don't (and can't) track this, but CheckLifetimes will
    //    notice the origin it contains.

    // If this is a ref result (#3) or a raw !lit.ref returned with
    // __mlir_type, we *can't* track it as an owned result, because this
    // doesn't own the value!
    if (isTypeObviouslyTrivial(v.getType()))
      return;

    if (isa<KGENCallOpInterface, LIT::CallIndirectOp>(res.getOwner())) {
      // Otherwise we tell CheckLifetimes to track it because it is either an
      // owned register result or it doesn't matter because it is trivial.
      name = StringAttr::get(v.getContext(), "(call result)");
      isIndirect = false;
      startsUninit = true;
      endInitState = EndsUninit;
      return;
    }

    // The results of an if operation can be owned.
    if (isa<HLCF::IfOp, HLCF::ElifOp>(res.getOwner())) {
      name = StringAttr::get(v.getContext(), "(if result)");
      isIndirect = false;
      startsUninit = true;
      endInitState = EndsUninit;
      return;
    }

    // Otherwise, some other unknown op result.
    return;
  }

  // If this is a function argument, check to see what ownership it has.
  auto bbArg = dyn_cast<BlockArgument>(v);
  if (!bbArg || !bbArg.getOwner())
    return;
  auto func = dyn_cast<FnOp>(bbArg.getOwner()->getParentOp());
  if (!func)
    return;
  FnTypeGeneratorType signature = func.getFuncTypeGenerator();

  unsigned argIdx = bbArg.getArgNumber();
  switch (signature.getArgConvention(argIdx)) {
  case ArgConvention::ReadReg:
    // This is immutable so don't need to be tracked.
    return;

  case ArgConvention::ReadMem:
  case ArgConvention::Mut:
  case ArgConvention::MutRef:
  case ArgConvention::Ref:
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsInit;
    break;

  case ArgConvention::OwnedReg:
    isIndirect = false;
    startsUninit = false;
    endInitState = EndsUninit;
    break;
  case ArgConvention::OwnedMem:
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsUninit;
    break;
  case ArgConvention::ByRefResult:
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnNormal;

    // Initializers allow member-wise initialization of 'self' to construct a
    // full value.
    if (func.getSpecialFunctionInfo().isInitializer())
      isFullObjectLiveOnEntry = true;
    break;
  case ArgConvention::ByRefError:
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnError;
    break;
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
Value OriginTrackable::findUnderlyingValueFromField(Value value) {
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
  OriginTrackable result(value);
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
static void getCallOpEffects(
    Operation &op, SmallVectorImpl<std::pair<Value, OperandEffect>> &operands,
    SmallVectorImpl<ResultEffect> &results, SmallVectorImpl<TypedAttr> &origins,
    CachedOriginFinder &originFinder) {
  FnType signature;
  OperandRange callArguments = op.getOperands();
  ArrayRef<ArgConvention> conventions;
  size_t argIdxOffset = 0;

  if (auto directCall = dyn_cast<KGENCallOpInterface>(op)) {
    // These all have the callee as a parameter, not operand.
    signature = directCall.getCalleeType().getBody();
    conventions = signature.getArgConventions().drop_back(
        signature.getNumAsyncReturnSlots());

    // CreateClosureOp has a subset of the operands of a call.
    if (isa<CreateClosureOp>(op)) {
      conventions = conventions.take_front(op.getNumOperands());
      argIdxOffset = 1;
    }

    assert(conventions.size() == op.getNumOperands());
  } else {
    auto call = cast<LIT::CallIndirectOp>(op);
    signature = call.getCallee().getType().getBody();
    conventions = signature.getArgConventions();

    // We use the callee value, and process the rest as operands.
    operands.push_back({op.getOperand(0), OperandEffect::regUse});
    assert(conventions.size() == op.getNumOperands() - 1);
    callArguments = callArguments.drop_front();
    argIdxOffset = 1;
  }

  /// Argument conventions cause a direct use of the register of pointee, and
  /// handling them specifically allows us to be field sensitive in cases where
  /// the access is directly attributable to a Value.
  auto getOperandEffectForConvention = [](ArgConvention conv,
                                          Type argType) -> OperandEffect {
    switch (conv) {
    case ArgConvention::OwnedReg:
      return OperandEffect::regConsume;
    case ArgConvention::OwnedMem:
      return OperandEffect::memConsume;
    case ArgConvention::ReadReg:
      return OperandEffect::regUse;
    case ArgConvention::ReadMem:
    case ArgConvention::Mut:
    case ArgConvention::MutRef:
    case ArgConvention::Ref: {
      bool isMut = cast<RefType>(argType).isMutableKnown(true);
      return isMut ? OperandEffect::memMut : OperandEffect::memLoad;
    }
    case ArgConvention::ByRefError:
    case ArgConvention::ByRefResult:
      return OperandEffect::memStoreOwned;
    }
    llvm_unreachable("invalid input convention");
  };

  SmallVector<Type> typesAccessibleByCallee;
  auto addArgument = [&](Value arg, ArgConvention conv,
                         bool noIndirect = false) {
    // Get normal argument effect.
    Type argType = arg.getType();
    auto effect = getOperandEffectForConvention(conv, argType);

    // If this is a normal register use, and if the value is a reference
    // (whether the argument convention is fancy or if it is an explicitly
    // passed reference) treat this as a field sensitive access so we can
    operands.push_back({arg, effect});

    // If the caller doesn't want us to add type-based origin effects, don't.
    if (noIndirect)
      return;

    // Do not add type-based origin effects for result arguments.  They are
    // returned, not accessed and therefore don't conflict with the inputs.
    if (conv == ArgConvention::ByRefResult || conv == ArgConvention::ByRefError)
      return;

    // If this is a memConsume or memStoreOwned, then the origin of the
    // reference is handled directly, strip it off.  Otherwise handle read,
    // mut, etc operands as just any-old reference use.
    if (hasAddress(conv))
      argType = cast<RefType>(argType).getElementType();

    // In addition to the direct (field-sensitive) effect of loading/storing
    // the bits, the callee may do whatever it wants with origins embedded
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
    // instead of seeing abstract uses of the origins.  This provides two
    // benefits:
    //   1) given "direct" access information, it allows us to model 'owned'
    //      argument conventions which consume the operand, something origin
    //      accesses cannot model (because it requires field sensitivity).
    //   2) it allows us to reason about the varargs uses field-sensitively,
    //      e.g. you can pass `a.x` through varargs and `a.y` through a 'mut'
    //      without the compiler imagining a conflict on "a" just like other
    //      arguments.
    // TODO(field-sensitive origins): remove this hack.
    if (auto vararg = arg.getDefiningOp<POP::VariadicCreateOp>()) {
      auto varargConvention = vararg.getType().getConvention();
      for (auto varOperand : vararg.getOperands())
        addArgument(varOperand, varargConvention);
      continue;
    }

    // If this is a pack, dig out the pack create so we can model owned
    // arguments correctly.
    // TODO: It would be nice to handle more fine grain effects in a general way
    // on calls.  This is a hack.
    // TODO(field-sensitive origins): remove this hack.
    // TODO: This should be removed. This is disabled for packs passed by-ref
    // when they are owned.
    if (signature.isPack(idx)) {
      auto packVal = RefPackCreateOp::findRefPackCreate(arg);
      assert(packVal && "couldn't decode variadic pack information!");

      if (auto pack = packVal.getDefiningOp<RefPackCreateOp>()) {
        if (signature.isPack(idx + argIdxOffset)) {
          auto argConvention =
              signature.getPackVarArgConvention(idx + argIdxOffset);
          for (auto packOperand : pack.getOperands())
            addArgument(packOperand, argConvention);

          // Also add the pack itself so the VariadicPack doesn't get destroyed
          // too early.  We already handled all the individual elements, so
          // don't redundantly process them.  Doing so is a problem for owned
          // operands.
          addArgument(arg, convention, /*noIndirect=*/true);
          if (argConvention != ArgConvention::OwnedMem)
            typesAccessibleByCallee.push_back(arg.getType());
          continue;
        }
      }
      /// Zero argument packs are kgen.param.constant but they have no
      /// references anyway.
      assert(packVal.getDefiningOp<ParamConstantOp>());
    }

    addArgument(arg, convention);
  }

  // Look at the types accessible by the callee to see if there are any
  // origin accesses.
  {
    SmallVector<TypedAttr> originsUsedByTypes = originFinder.findOriginsIn(
        typesAccessibleByCallee, signature.getCaptureOrigins());
    origins.append(originsUsedByTypes.begin(), originsUsedByTypes.end());
  }

  // If the result is defining an owned register value, then we treat this as
  // a definition
  if (op.getNumResults()) {
    assert(op.getNumResults() == 1);
    results.push_back(isTypeObviouslyTrivial(op.getResult(0).getType())
                          ? ResultEffect::ignore
                          : ResultEffect::regDefine);
  }
}

/// This computes the effects that an operation has on any operands and result
/// values. This information is used by both phases of CheckLifetimes.
OverallOpValueEffect LIT::getOperationEffects(
    Operation &op, SmallVectorImpl<std::pair<Value, OperandEffect>> &operands,
    SmallVectorImpl<ResultEffect> &results, SmallVectorImpl<TypedAttr> &origins,
    CachedOriginFinder &originFinder) {
  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.  These indexing operations are handled specially.
  if (isa<RefStructGEROp, RebindOp, RefImmutOp>(op) ||
      llvm::isa_and_nonnull<DebugInfo::DebugInfoDialect>(op.getDialect())) {
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
  if (isa<ParamMaterializeOp>(op)) {
    for (Value o : op.getOperands())
      operands.push_back({o, OperandEffect::regConsume});
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  // RefFromPointerOp creates a new origin tracked value.  The
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
    getCallOpEffects(op, operands, results, origins, originFinder);
    return {};
  }

  // A return consumes all the live-out values from the function.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp, KGEN::UnreachableOp,
          HLCF::YieldOp>(op)) {
    // We always consume the result register - even if it is often trivial.
    for (auto o : op.getOperands())
      operands.push_back({o, OperandEffect::regConsume});

    // Yield doesn't need any special processing, just handling of its operands.
    if (isa<HLCF::YieldOp>(op))
      return {};

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

  if (auto mark = dyn_cast<OwnershipMarkConsumedOp>(op)) {
    operands.push_back({mark.getOperand(), OperandEffect::memConsume});
    return {};
  }

  // Local control flow ops.
  if (isa<HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp, ParamForBreakOp,
          ParamForContinueOp>(op))
    return OverallOpValueEffect::localControlFlowOp;

  // If-like operations.
  if (isa<ParamIfOp, HLCF::IfOp>(op)) {
    // If-like ops can return owned results.
    for (auto result : op.getResults()) {
      auto effect = isTypeObviouslyTrivial(result.getType())
                        ? ResultEffect::ignore
                        : ResultEffect::regDefine;
      results.push_back(effect);
    }
    return OverallOpValueEffect::ifLikeOp;
  }

  if (isa<HLCF::ElifOp>(op)) {
    // If-like ops can return owned results.
    for (auto result : op.getResults()) {
      auto effect = isTypeObviouslyTrivial(result.getType())
                        ? ResultEffect::ignore
                        : ResultEffect::regDefine;
      results.push_back(effect);
    }
    return OverallOpValueEffect::elifOp;
  }

  /// This is HLCF::LoopOp.
  if (isa<HLCF::LoopOp, ParamForOp>(op))
    return OverallOpValueEffect::loopOp;

  if (isa<LIT::TryOp>(op))
    return OverallOpValueEffect::tryOp;

  assert(!isa<HLCF::SwitchOp>(op) && "Only created by LowerSuspension Points");
  return OverallOpValueEffect::unknownOp;
}

//===----------------------------------------------------------------------===//
// CachedOriginFinder
//===----------------------------------------------------------------------===//

/// Unpack the specified value of OriginType into a set of referenced
/// origins. Returns true if any origins were found.
static bool handleOriginAttr(TypedAttr attr,
                             SmallVectorImpl<TypedAttr> &results) {
  bool foundAny = false;

  // Look through unions to find the values referenced.
  processRawOrigin(attr, [&](TypedAttr raw) {
    // FIXME(origins): This shouldn't happen; UncheckedCallEmission isn't
    // forming captures correctly for async functions with implicit origin
    // refs.
    if (isa<ImplicitOriginRefAttr>(OriginMutCastAttr::strip(attr)))
      return;

    results.push_back(raw);
    foundAny = true;
  });
  return foundAny;
}

template <typename TypeOrAttr>
static bool scanForOrigins(TypeOrAttr pvalue,
                           DenseSet<const void *> &typesAndAttrsWithoutOrigins,
                           DenseMap<const void *, bool> &visited,
                           SmallVectorImpl<TypedAttr> &results) {
  const void *pvaluePtr = pvalue.getAsOpaquePointer();

  // Ignore types we have already scanned.
  if (typesAndAttrsWithoutOrigins.contains(pvaluePtr))
    return false;
  if (auto it = visited.find(pvaluePtr); it != visited.end())
    return it->second;

  // If this has origin type, process it.
  bool handled = false;
  bool hasOrigin = false;
  if constexpr (std::is_base_of_v<Attribute, TypeOrAttr>)
    if (auto typedAttr = dyn_cast<TypedAttr>(pvalue))
      if (isa<OriginType>(typedAttr.getType())) {
        hasOrigin |= handleOriginAttr(typedAttr, results);
        handled = true;
      }

  if (!handled) {
    // Recursively check for any nested types, e.g. the input/outputs of a
    // function type, types like !pop.scalar<ty> etc.
    pvalue.walkImmediateSubElements(
        [&](Attribute attr) {
          hasOrigin |= scanForOrigins(attr, typesAndAttrsWithoutOrigins,
                                      visited, results);
        },
        [&](Type type) {
          hasOrigin |= scanForOrigins(type, typesAndAttrsWithoutOrigins,
                                      visited, results);
        });
  }

  // If we can prove that this subtree doesn't contain origins, then remember
  // this so we don't revisit this type/attribute in the future.
  if (!hasOrigin)
    typesAndAttrsWithoutOrigins.insert(pvaluePtr);
  else
    // We don't need to visit the same attribute more than once to find origin
    // references. This is required to prevent splatting the parameter
    // expression tree.
    visited.try_emplace(pvaluePtr, hasOrigin);
  return hasOrigin;
}

/// This method finds all the origins buried in the specified type,
/// returning them as a list.  This typically will return ParamRefAttr's or
/// ImmutCast(ParamRefAttr)'s if a mutable origin is accessed immutably.
SmallVector<TypedAttr>
CachedOriginFinder::findOriginsIn(ArrayRef<Type> types,
                                  ArrayRef<TypedAttr> captures) {
  SmallVector<TypedAttr> results;

  // Scan each type, accumulating the results; the set avoid revisiting nodes
  // that we know cannot have origins.
  DenseMap<const void *, bool> visited;
  for (Type type : types)
    scanForOrigins(type, typesAndAttrsWithoutOrigins, visited, results);
  for (TypedAttr capture : captures)
    scanForOrigins(capture, typesAndAttrsWithoutOrigins, visited, results);
  return results;
}
