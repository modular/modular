//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core of call emission, without compatibility/type
// checking and error emission.
//
//===----------------------------------------------------------------------===//

#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "ParserEvaluationContext.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// This helper function emits a call to VariadicPack(refPackValue) and returns
/// the result value.  'variadicPackType' is the fully bound VariadicPack type
/// per the function signature.
static CValue
emitVariadicPackConstructor(ASTType variadicPackType, TypedAttr originToUse,
                            const ExprNode *expr, ExprEmitter &emitter,
                            std::function<CValue(RefPackType)> refPackBuilder) {
  RefPackType packType = variadicPackType.getVariadicPackInfo(emitter.shared);

  // If there was no origin specified, use an immortal one with the same
  // mutability.
  if (!originToUse)
    originToUse = OriginUnionAttr::get(packType.getOrigin().getType());

  // Rebind the !lit.ref.pack with the common origin.
  packType = RefPackType::get(packType.getVariadic(), originToUse,
                              packType.getAddressSpace());

  // Build the !lit.ref.pack or #lit.ref.pack value with the adjusted origin.
  CValue refPackValue = refPackBuilder(packType);

  // Emit a VariadicPack constructor call taking the #lit.ref.pack and a
  // bool indicating whether the argument is owned.
  CallOperands operands;
  operands.add({refPackValue, expr});

  ValueDest packDest(ExprContext::EC_PackArgument);

  auto variadicPackStructDecl =
      cast<StructDeclOp>(variadicPackType.getDecl(emitter.shared));
  SmallVector<TypedAttr> bindings(
      cast<LIT::StructType>(variadicPackType).getParamValues());
  // NOTE: `bindings[0]` and `bindings[1]` are expected to be the Mojo `Bool`
  // type, and `bindings[2]` is an Origin.
  assert(bindings.size() == 5 && isa<LIT::StructType>(bindings[0].getType()) &&
         isa<LIT::StructType>(bindings[1].getType()) &&
         isa<LIT::StructType>(bindings[2].getType()) &&
         isa<AnyTraitType>(bindings[3].getType()) &&
         isa<VariadicType>(bindings[4].getType()) &&
         "Not a VariadicPack struct?");

  // Construct the pack type without parameters so we re-infer the origin which
  // is different on the caller side (the union of the argument origins) than
  // the declared callee side (a parameter).
  ParserParameterEvaluator evaluator(emitter.shared);
  for (auto [idx, currBinding] : llvm::enumerate(bindings)) {
    // Do not clear the `is_owned` parameter since it's not inferrable from the
    // operands to VariadicPack. It has to be set explicitly based on what
    // convention was used to construct the pack.
    if (idx != 1) // Index `1` is the `is_owned` parameter.
      currBinding =
          UnboundAttr::get(evaluator.getReboundType(currBinding.getType()));
    evaluator.addInputValue(currBinding);
  }
  ASTType unboundVariadicPackType =
      variadicPackStructDecl.bindReference(bindings);

  return emitter.emitConstructorCall(unboundVariadicPackType,
                                     std::move(operands), expr,
                                     CallSyntax::kTypeCall, packDest);
}

//===----------------------------------------------------------------------===//
// CallEmitter (implementation detail)
//===----------------------------------------------------------------------===//

class CallEmitter {
public:
  CallEmitter(RValue callee, const ExprNode *callExpr, ExprEmitter &emitter,
              ValueDest &dest)
      : emitter(emitter), callee(callee), callExpr(callExpr),
        loc(emitter.translateLocation(callExpr->getLoc())), dest(dest),
        calleeSig(cast<FuncTypeGeneratorType>(callee.getRValueType())),
        afterCallActions(*this) {}

  /// Emit IR for a single argument, according to its convention.
  AnyValue emitOneArgVal(ASTExprAnd<AnyValue> operand, unsigned argIdx,
                         ArgConvention convention, Type expectedType,
                         size_t sequenceIndex = 0);

  /// Emit all arguments and return their values in a vector. This function
  /// iterates by expected arguments since we're building the argument list of
  /// the call. Default arguments are applied (if available and an operand isn't
  /// provided for the arg), and variadics (including packs) are collected from
  /// the operand list and emitted as the appropriate variadic/pack type to the
  /// callee.
  FailureOr<SmallVector<ASTExprAnd<AnyValue>>>
  emitArgValues(const CallOperands &operands);

  /// This function emits the specified pre-emitted argument into a single MLIR
  /// Value suitable for passing to the callee with the specified convention.
  Value emitPreemittedArgumentAsDynamicValue(ASTExprAnd<AnyValue> argValAndExpr,
                                             ArgConvention convention,
                                             Type declaredArgType,
                                             ArrayRef<Value> callArgsSoFar);

  /// Emit a function call in a parameter context.
  TypedAttr
  emitCallInParamContext(ArrayRef<ASTExprAnd<AnyValue>> argumentValues);

  /// Emit any after-call actions collected during call emission.
  void emitAfterCallActions() { afterCallActions.emit(); }

  /// Emit warnings about incorrect code in a direct call.
  void emitDirectCallWarnings(LIT::CallOp call,
                              const CallOperands &callOperands);

  /// The underlying expression emitter instance.
  ExprEmitter &emitter;

  /// Given a call to a function with a memory only result and the desired value
  /// destination, decide if it is safe to directly emit into the slot.  Doing
  /// so requires a form of alias analysis to determine whether any input
  /// arguments could alias the result slot.  We cannot emit into the result
  /// slot when passing the value as an argument like 'x = foo(x)' or 'x = x +
  /// 1'.
  ///
  /// At this point, we've already applied implicit conversions and converted
  /// things to RValues or BValues as required by the argument convention, but
  /// things may still be in parameter space.
  bool isSafeToUseValueDestForDirectResult(ASTType destRValueType,
                                           ArrayRef<Value> argumentValues);

private:
  /// The (type-checked and resolved) callee we are emitting the call to.
  RValue callee;
  /// The call's expression node.
  const ExprNode *callExpr;
  /// The mlir location of the call expression above, stored for convenience.
  Location loc;
  /// The destination context we're emitting into.
  ValueDest &dest;
  /// The signature type of the callee, stored for convenience.
  FnTypeGeneratorType calleeSig;

  /// This struct accumulates information about IR to emit after the call, e.g.
  /// writebacks for computed mut lvalues, and origin markers.
  struct AfterCallActions {
    CallEmitter &callEmitter;

    // The first entry of this is a ValueDest for a DLValue that we can invoke
    // for the setter.
    SmallVector<std::pair<ValueDest, MLValue>> lvalueWritebacks;

    AfterCallActions(CallEmitter &callEmitter) : callEmitter(callEmitter) {}

    /// Emit all after-call actions.
    void emit();

    ~AfterCallActions() {
      // If an error happens before we emit the write backs, make sure to nuke
      // them so they don't crash the compiler.
      while (!lvalueWritebacks.empty())
        lvalueWritebacks.pop_back_val().first.resetForError();
    }
  } afterCallActions;

  /// Emit the given (remaining) operands as a variadic or pack sequence,
  /// appending to the given argument value vector.
  LogicalResult emitRemainingPosOperands(
      size_t argIdx, MutableArrayRef<ASTExprAnd<AnyValue>> remainingOperands,
      ArgConvention convention, Type expectedType,
      SmallVectorImpl<ASTExprAnd<AnyValue>> &argumentValues);
};

void CallEmitter::AfterCallActions::emit() {
  ExprEmitter &emitter = callEmitter.emitter;

  // Emit the elements and clear the writebacks so the ValueDest's get
  // destroyed when they are emitted into.
  while (!lvalueWritebacks.empty()) {
    // Get 'dest' (the computed LValue as a ValueDest) and 'lValue' (the memory
    // temporary we're working with) so we can do a writeback.
    auto [dest, lValue] = lvalueWritebacks.pop_back_val();

    // The lValue is the MLValue of the temporary holding the 'get'd value.  We
    // pass it as an MRValue to the "set" method, allowing the value to be
    // consumed directly by an 'owned' argument without a copy.
    if (!emitter.emitResult(MRValue(lValue), callEmitter.callExpr, dest))
      dest.resetForError();
  }
}

AnyValue CallEmitter::emitOneArgVal(ASTExprAnd<AnyValue> operand,
                                    unsigned argIdx, ArgConvention convention,
                                    Type expectedType, size_t sequenceIndex) {
  if (calleeSig.isPosVarArg(argIdx)) {
    // In the case of a variadic argument, we need to remove the
    // !pop.variadic<> wrapper to get the type to convert to.
    expectedType = cast<VariadicType>(expectedType).getElementType();
    convention = calleeSig.getPosVarArgConvention(argIdx);
  } else if (ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx)) {
    RefPackType packType = variadicPackType.getVariadicPackInfo(emitter.shared);

    // Operands being applied to a concrete pack type argument must be
    // converted to the pack element type at that index.  The calleeSig has the
    // pack type resolved to a concrete list of types it is expecting.
    expectedType =
        ASTType(packType.getVariadicIfResolved().getValues()[sequenceIndex]);
    // Get the !lit.ref with the origin and other paraphernalia.
    expectedType = packType.getElementRefTypeFor(expectedType);
    convention = calleeSig.getPackVarArgConvention(argIdx);
  }

  switch (convention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    // By-ref arguments, must be lvalues.
    assert(operand.ir.getIfLValue() && "Call should already be type checked");
    return operand.ir;
  case ArgConvention::OwnedMem:
    // Owned conventions pass rvalues.
    if (convention == ArgConvention::OwnedMem)
      expectedType = cast<RefType>(expectedType).getElementType();
    return emitter.emitRValue(operand, EC_CallArgValue, expectedType);

  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    // If we're in a parameter context, just leave it alone - param call
    // emission will handle it.
    if (!emitter.builder) {
      if (auto pv = operand.ir.getIfPValue())
        return pv;
    }

    // Emit the operand as a 'ref'.
    Value refValue = emitter.emitRefValue(operand, EC_CallRefArgValue);
    if (!refValue)
      return {};

    auto refValueType = cast<RefType>(refValue.getType());
    auto expectedRefType = cast<RefType>(expectedType);

    // Origins must be convertible, this is checked by OverloadFitness.
    // The destination may be less mutable because of canZeroCostConvert.
    // This also lazy materializes cast to immutable that MBValue avoided.
    if (!refValueType.isMutableKnown(false) &&
        expectedRefType.isMutableKnown(false)) {
      refValue = emitter.builder->create<RefImmutOp>(
          operand.expr->getLocation(emitter), refValue);
      refValueType = cast<RefType>(refValue.getType());
    }

    // The origins may disagree if we're converting a value to a
    // superset origin, e.g. "immortal -> X" or "X -> X|y".
    if (refValueType.getOrigin() != expectedRefType.getOrigin()) {
      refValue = emitter.builder->create<RebindOp>(
          operand.expr->getLocation(emitter),
          refValueType.getWithOrigin(expectedRefType.getOrigin()), refValue);
      refValueType = cast<RefType>(refValue.getType());
    }

    assert(refValueType == expectedType && "Should have exact match now");
    return CValue::getMValueForRef(refValue);
  }

  case ArgConvention::ReadMem:
    // by-ref arguments are converted to the expected r-value type.
    expectedType = cast<RefType>(expectedType).getElementType();
    [[fallthrough]];
  case ArgConvention::ReadReg:
    return emitter.emitBValue(operand, EC_CallArgValue, expectedType);
  }
  llvm_unreachable("unknown argument convention");
}

LogicalResult CallEmitter::emitRemainingPosOperands(
    size_t argIdx, MutableArrayRef<ASTExprAnd<AnyValue>> remainingOperands,
    ArgConvention convention, Type expectedType,
    SmallVectorImpl<ASTExprAnd<AnyValue>> &argumentValues) {
  // Emit all of the remaining values to make sure they're converted to the
  // right type.
  for (auto [idx, operand] : llvm::enumerate(remainingOperands)) {
    auto emittedArg =
        emitOneArgVal(operand, argIdx, convention, expectedType, idx);
    if (!emittedArg)
      return failure();
    operand.ir = emittedArg;
  }

  // If this is a variadic list, use the convention of the elements, not the
  // convention of the list itself.
  bool isPosVarArg = calleeSig.isPosVarArg(argIdx);
  if (isPosVarArg)
    convention = calleeSig.getPosVarArgConvention(argIdx);
  else if (calleeSig.getIfVariadicPack(argIdx))
    convention = calleeSig.getPackVarArgConvention(argIdx);

  // Handle emission in a compile-time context.  Parameter calls need to
  // generate parameter attributes.
  if (!emitter.builder) {
    SmallVector<TypedAttr> args;
    for (ASTExprAnd<AnyValue> operand : remainingOperands) {
      args.push_back(operand.ir.getIfPValue().get());
      if (!args.back()) {
        emitter.emitErrorForDynamicValueInParameter(callExpr,
                                                    "cannot use dynamic value");
        return failure();
      }
    }

    CValue argValue;
    if (isPosVarArg) {
      auto varType = cast<VariadicType>(expectedType);
      Type varElType = varType.getElementType();

      // If the element has a memory-only type, drop it into memory.
      if (hasAddress(convention)) {
        for (TypedAttr &arg : args)
          arg = StoreToMemAttr::get(arg, varElType);
      }
      auto newVarType = VariadicType::get(varElType, varType.getConvention());
      argValue = PValue(VariadicAttr::get(args, newVarType));
    } else {
      ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx);
      assert(variadicPackType && "Unknown variadic argument kind");
      // Bundle them up into a VariadicPack instance.
      argValue = emitVariadicPackConstructor(
          variadicPackType, /*origin*/ {}, callExpr, emitter,
          [&](RefPackType adjustedPackType) -> CValue {
            // RefPack elements are passed through memory.  Use adjustedPackType
            // to get the proper (immortal) origin installed.
            for (TypedAttr &arg : args)
              arg = StoreToMemAttr::get(
                  arg, adjustedPackType.getElementRefTypeFor(arg.getType()));
            return RefPackAttr::get(args, adjustedPackType);
          });
      if (!argValue)
        return failure();
    }
    argumentValues.push_back({argValue, remainingOperands[0].expr});
    return success();
  }

  // If not all remaining operands are compile-time values, use an operation to
  // create a variadic or pack sequence.
  SmallVector<Value> args;
  for (auto &operand : remainingOperands) {
    assert(convention != ArgConvention::ByRefResult &&
           "cannot have variadics of this convention, so can pass in empty "
           "callArgsSoFar");
    Value argVal = emitPreemittedArgumentAsDynamicValue(
        operand, convention, expectedType, ArrayRef<Value>());
    if (!argVal)
      return failure();
    args.push_back(argVal);

    // Variadic and pack arguments are always passed through memory. An
    // exception was carved out for trivial register-passable values, which
    // don't require origin tracking.
    // TODO(MOCO-726): Make variadics always pass through memory.
    if (hasAddress(convention) ||
        ASTType(argVal.getType()).isTrivial(callExpr->getLoc(), emitter.shared))
      continue;
    emitter.shared.emitError(
        operand.expr->getLoc(),
        "cannot bind non-trivial value to trivial variadic argument");
  }

  // If there are origins on anything, create a uniform representation and
  // cast to a common reference type.
  if (!args.empty() && isa<RefType>(args.back().getType())) {
    // If one arg is a reference, then they all are.
    SmallVector<TypedAttr> refOrigins;
    for (auto arg : args)
      refOrigins.push_back(cast<RefType>(arg.getType()).getOrigin());

    // All the origins will have the same OriginType, indicating the
    // reference mutability that the callee expected.
    OriginType commonOriginType = cast<OriginType>(refOrigins.back().getType());

    // If there is more than one element, they probably have different
    // origins, and thus need to be rebound into a common union of them.
    auto commonOrigin = OriginUnionAttr::get(refOrigins, commonOriginType);
    for (auto &arg : args) {
      auto argType = cast<RefType>(arg.getType());
      if (argType.getOrigin() == commonOrigin)
        continue; // Already the right origin.
      // Cast to common origin with a rebind.
      arg = emitter.builder->create<RebindOp>(
          loc, argType.getWithOrigin(commonOrigin), arg);
    }
  }

  // Given a reference type for a variadic list of pack element, return the same
  // type updated to the common origin of the elements.
  auto getCommonOrigin = [&]() -> TypedAttr {
    if (!args.empty())
      return cast<RefType>(args.back().getType()).getOrigin();
    return {};
  };

  CValue argVal;
  if (isPosVarArg) { // Positional homogenous varargs
    // Rebind the origin of the argument to the expected origin if needed.
    auto expectedVararg = cast<VariadicType>(expectedType);
    if (auto refType = dyn_cast<RefType>(expectedVararg.getElementType())) {
      auto origin = getCommonOrigin();
      if (!origin) // No arguments, use immortal with same mutability.
        origin = OriginUnionAttr::get(refType.getOrigin().getType());

      refType = refType.getWithOrigin(getCommonOrigin());
      expectedType = VariadicType::get(refType, expectedVararg.getConvention());
    }

    // Check for a splat.
    if (!args.empty() &&
        llvm::all_of(args, [&](Value operand) { return operand == args[0]; })) {
      argVal = SRValue(emitter.builder->create<POP::VariadicSplatOp>(
          loc, expectedType, args[0], args.size()));
    } else {
      argVal = SRValue(emitter.builder->create<POP::VariadicCreateOp>(
          loc, expectedType, args));
    }
  } else {
    // Bundle them up into a VariadicPack instance.
    ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx);
    assert(variadicPackType && "Must be a VariadicPack");
    argVal = emitVariadicPackConstructor(
        variadicPackType, getCommonOrigin(), callExpr, emitter,
        [&](RefPackType adjustedPackType) -> CValue {
          return SRValue(emitter.builder->create<RefPackCreateOp>(
              loc, adjustedPackType, args));
        });
    if (!argVal)
      return failure();
  }
  argumentValues.push_back({argVal, remainingOperands[0].expr});
  return success();
}

FailureOr<SmallVector<ASTExprAnd<AnyValue>>>
CallEmitter::emitArgValues(const CallOperands &operands) {
  // This is the index into the operands list for the next operand value to look
  // at for positional arguments.
  size_t posOperandIdx = 0;

  // We will collect argument names that were passed by keyword, so that we can
  // emit **kwargs arguments in the end with anything that's left.
  SmallPtrSet<StringAttr, 4> passedByKw;
  // We also remember if we had a **kwargs.
  MRValue kwargsDict;

  SmallVector<ASTExprAnd<AnyValue>> argumentValues;
  argumentValues.reserve(calleeSig.getNumArguments());

  PogListAttr argListAttr = calleeSig.getArgListAttrs();
  DefaultValueHandler defaultHandler(argListAttr);
  for (auto [argIdx, expectedType, convention, pogAttr] :
       llvm::enumerate(calleeSig.getArguments(), calleeSig.getArgConventions(),
                       argListAttr.getPogs())) {

    // If this is the return slot for a call, we need a temporary to emit into,
    // but don't know the type until the arguments (and their origins) are all
    // emitted. Just skip over it for now.
    if (isResultSlot(convention)) {
      assert(calleeSig.hasMemoryOnlyResult() ||
             (calleeSig.isThrows() &&
              pogAttr.getPassingKind() == PassingKind::Implicit));
      argumentValues.push_back({AnyValue(), callExpr});
      continue;
    }

    // See what the next positional argument is, skipping over any keywords.
    while (posOperandIdx < operands.size() && operands[posOperandIdx].keyword)
      ++posOperandIdx;

    // Process positional arguments.
    if (posOperandIdx < operands.size()) {
      // For a normal (not a vararg or a pack) positional argument, we just emit
      // it and add it to our list.
      if (!calleeSig.isPosVarArg(argIdx) && !calleeSig.isPack(argIdx)) {
        ASTExprAnd<AnyValue> operand = operands[posOperandIdx++];
        AnyValue argVal =
            emitOneArgVal(operand, argIdx, convention, expectedType);
        if (!argVal)
          return failure();
        argumentValues.push_back({argVal, operand.expr});
        continue;
      }

      // At this point, we must be dealing with variadic or pack arguments. We
      // handle these all at once (or fail).
      SmallVector<ASTExprAnd<AnyValue>> remainingOperands;
      do {
        auto &operand = operands[posOperandIdx];
        if (!operand.keyword)
          remainingOperands.push_back(operand);
        ++posOperandIdx;
      } while (posOperandIdx < operands.size());

      // NOTE: this implicitly assumes that variadics/packs are at the end.
      if (succeeded(emitRemainingPosOperands(argIdx, remainingOperands,
                                             convention, expectedType,
                                             argumentValues)))
        continue;

      return failure();
    }

    // If we ran out of operands, fulfill this with a keyword argument, default
    // value, empty variadic list, or empty pack.
    if (calleeSig.isPosVarArg(argIdx)) {
      // VarArgs arguments are fulfilled with an empty !kgen.variadic list.
      auto argAttr = VariadicAttr::get(ArrayRef<TypedAttr>(),
                                       cast<VariadicType>(expectedType));
      argumentValues.push_back({PValue(argAttr), callExpr});
      continue;
    }

    // Pack arguments are fulfilled with an empty #lit.ref.pack.
    if (ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx)) {
      assert(cast<VariadicAttr>(variadicPackType.getVariadicPackTypeList())
                 .getValues()
                 .empty() &&
             "pack type already checked against operand count");
      // Emit a VariadicPack constructor call.
      auto variadicPack = emitVariadicPackConstructor(
          variadicPackType, /*origin*/ {}, callExpr, emitter,
          [&](RefPackType adjustedPackType) -> CValue {
            return RefPackAttr::get(ArrayRef<TypedAttr>(), adjustedPackType);
          });
      if (!variadicPack)
        return failure();
      argumentValues.push_back({variadicPack, callExpr});
      continue;
    }

    if (calleeSig.isKwVarArg(argIdx)) {
      assert(!kwargsDict && "multiple **kwargs not supported yet");
      // If this is a variadic keyword argument, we initialize a dictionary.
      ValueDest dictDest(ExprContext::EC_KWArgsArgument);
      auto dict = emitter.emitConstructorCall(
          cast<RefType>(expectedType).getElementType(), {}, callExpr,
          CallSyntax::kTypeCall, dictDest);
      kwargsDict = emitter.emitMRValue({dict, callExpr}, EC_CallArgValue);
      argumentValues.push_back({kwargsDict, callExpr});
      continue;
    }

    StringAttr argName = pogAttr.getName();
    if (const OperandValue *kwOperandOr = operands.findKwArg(argName);
        kwOperandOr) {
      // The argument is passed as a keyword operand.
      AnyValue argVal =
          emitOneArgVal(*kwOperandOr, argIdx, convention, expectedType);
      if (!argVal)
        return failure();
      passedByKw.insert(argName);
      argumentValues.push_back({argVal, kwOperandOr->expr});
      continue;
    }

    // Otherwise, apply the default argument. We've ensured before that we
    // have a default argument for each missing operand.
    TypedAttr defaultOr = defaultHandler.getDefault(argIdx);
    assert(defaultOr);
    assert(convention != ArgConvention::Mut &&
           "by_ref argument cannot have defaults");
    argumentValues.push_back({PValue(defaultOr), callExpr});
    continue;
  }

  assert(posOperandIdx == operands.size() &&
         "typechecking confirmed that we would use up all positional operands");

  // Fill the **kwargs dict with values that we didn't bind to an argument.
  ValueDest kwargsDest(EC_KWArgsArgument);
  for (auto &operand : operands.values) {
    if (!operand.keyword || passedByKw.contains(operand.keyword))
      continue;

    assert(kwargsDict && "typechecking confirmed we have no **kwargs");

    SMLoc loc = operand.expr->getLoc();

    SyntheticNode tmpNode(loc);
    CValue literalKey = StringLiteralNode::emitCtorCall(
        operand.keyword.strref(), &tmpNode, kwargsDest, emitter);
    if (!literalKey)
      return {};

    // Then we set the element with the given key and the operand as value.
    CallOperands insertOperands(
        {{MLValue(kwargsDict), callExpr}, {literalKey, operand.expr}, operand});
    emitter.emitNamedMethodCall("_insert", std::move(insertOperands),
                                kwargsDest, CallSyntax::kMethodCall, callExpr);
  }

  return argumentValues;
}

/// Given a call to a function with a memory only result and the desired value
/// destination, decide if it is safe to directly emit into the slot.  Doing so
/// requires a form of alias analysis to determine whether any input arguments
/// could alias the result slot.  We cannot emit into the result slot when
/// passing the value as an argument like 'x = foo(x)' or 'x = x + 1'.
///
/// At this point, we've already applied implicit conversions and converted
/// things to RValues or BValues as required by the argument convention, formed
/// variadic list/packs, and emitted to the final SSA values that will get
/// passed.
bool CallEmitter::isSafeToUseValueDestForDirectResult(
    ASTType destRValueType, ArrayRef<Value> argValues) {

  // If the callee is returning a RefResult, don't do this.
  if (calleeSig.isRefResult())
    return false;

  // Check to see if the destination provides a buffer.  If not, it is safe to
  // emit into it, but it doesn't actually matter.
  MLValue destBuffer = dest.getDefinedMLValueIfExists(destRValueType, emitter);
  if (!destBuffer)
    return true;

  // If this is a throwing function, then we cannot write to a field of a
  // origin tracked value.  Consider:
  //     x.f = foo()
  // The contract for the result slot is that 'x.f' has to be uninitialized
  // before the function call, so we would have to destroy any live value in the
  // slot before calling the function:
  //     x.f.__del__()
  //     foo(x.f, __error__)  # Passed as byref_result
  // However, when the function throws, if 'x' is unused, we need to be able to
  // delete the full 'x' value.  However, we cannot call the destructor on 'x'
  // because the whole value isn't initialized.  Address this by not assigning
  // into submembers in throwing functions.
  if (calleeSig.isThrows()) {
    // See if the destination buffer is something that ownership can track.
    Value underlyingDest =
        OriginTrackable::findUnderlyingValueFromField(destBuffer);
    // If we don't know what it is, it may be a 'ref' result, handle
    // conservatively.
    if (!underlyingDest)
      return false;

    // Dig deeper into the nature of the thing we're assigning into to try to
    // enable a few important cases.
    OriginTrackable trackable(underlyingDest);

    // We can't allow assigning into a field because we cannot partially destroy
    // the value, but we can overwrite the whole thing.
    if (underlyingDest != destBuffer) {
      // byref_result arguments of initializers can be piecewise destroyed on a
      // thrown error.
      if (!trackable.isFullObjectLiveOnEntry ||
          trackable.endInitState !=
              OriginTrackable::ExitInitState::InitOnNormal)
        return false;
    }

    // We also cannot assign to values that must be live-out from the function
    // on an error return.  This includes (for example) by-ref arguments.
    if (trackable.endInitState == OriginTrackable::ExitInitState::EndsInit)
      return false;
  }

  // Collect all of the types of all the arguments so we can collect the
  // origins they may reference.
  SmallVector<Type> argTypes;
  for (auto [value, convention] :
       llvm::zip(argValues, calleeSig.getArgConventions())) {
    if (isResultSlot(convention))
      continue;

    argTypes.push_back(value.getType());
  }

  SmallPtrSet<Attribute, 2> destOrigins;

  // We're not doing field sensitive comparisons below, so strip down to the
  // base origin for comparisons.
  // TODO: Make this more aggressive with field information.
  processRawOrigin(cast<RefType>(destBuffer.getType()).getOrigin(),
                   [&](TypedAttr origin) {
                     while (auto fieldAttr = dyn_cast<OriginFieldAttr>(origin))
                       origin = fieldAttr.getBase();
                     // AnyOrigin is assumed to be ok since it is used for
                     // UnsafePointer etc.
                     if (!isa<AnyOriginAttr>(origin))
                       destOrigins.insert(origin);
                   });

  if (destOrigins.empty())
    return true;

  // Check to see if any of the the origins they may be accessing are the
  // origin in question.  If any of them is a possible reference to the
  // destination slot, then we must fail.
  CachedOriginFinder &finder = emitter.shared.cachedOriginFinder;
  for (TypedAttr origin :
       finder.findOriginsIn(argTypes, calleeSig.getCaptureOrigins())) {
    // If an operand is reading from the origin, there will be an immcast in
    // the way.  Look through it and any field sensitivity.
    origin = OriginMutCastAttr::strip(origin);
    while (auto fieldAttr = dyn_cast<OriginFieldAttr>(origin))
      origin = fieldAttr.getBase();

    // If the destination set includes this origin, then we can't use the
    // destination.
    if (destOrigins.count(origin))
      return false;
  }

  // If no problems are found, it is safe!
  return true;
}

/// This function emits the specified pre-emitted argument into a single MLIR
/// Value suitable for passing to the callee with the specified convention.
Value CallEmitter::emitPreemittedArgumentAsDynamicValue(
    ASTExprAnd<AnyValue> argValAndExpr, ArgConvention convention,
    Type declaredArgType, ArrayRef<Value> callArgsSoFar) {
  assert(emitter.builder && "Should only be called in dynamic context");

  // This checks any returned MValue argument convention for validity.
  auto checkMValueAddrSpace = [&](AnyValue someMValue) -> Value {
    // Propagate errors.
    if (!someMValue)
      return {};
    assert(someMValue.isMValue() && "Not an MValue");

    // All argument conventions take things in the default address space, so any
    // use of references in other address spaces need to do a copyinit.  Right
    // now __copyinit__ requires the source to be borrowed (it doesn't allow a
    // `ref ` existing so there is no way to define a
    // non-@register_passable("trivial") type in another address space. Diagnose
    // this error with a specific message, and copy RPTrivial types into address
    // space 0 implicitly.
    auto refType = someMValue.getMValueType();

    if (!refType.isDefaultAddrSpace()) {
      auto *expr = argValAndExpr.expr;
      // Non-trivial types cannot be copied.
      // TODO: If there is a reason to, we could generalize copyinit.
      if (!ASTType(refType.getElementType())
               .isTrivial(expr->getLoc(), emitter.shared)) {
        emitter.emitError(expr->getLoc(),
                          "non-trivial value cannot be copied from a "
                          "non-default address space")
            << expr->getRange();
        return {};
      }

      // If this is a trivial value, then we can do a copy by doing a load.
      auto srVal = emitter.emitSRValue({someMValue, expr}, EC_CallArgValue);
      someMValue = emitter.emitMRValue({srVal, expr}, EC_CallArgValue);
      if (!someMValue)
        return {};
    }
    return someMValue.getMValueReference();
  };

  switch (convention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::OwnedMem:
    // Promote PValue's if needed.
    return checkMValueAddrSpace(
        emitter.emitMRValue(argValAndExpr, EC_CallArgValue));
    break;
  case ArgConvention::ReadReg:
    if (auto pVal = argValAndExpr.ir.getIfPValue())
      return emitter.emitSRValue(argValAndExpr, EC_CallArgValue);

    // If this is an MBValue, the element must be register passable but not
    // loaded.
    if (argValAndExpr.ir.isMValue()) {
      auto refVal = argValAndExpr.ir.getMValueReference();
      return emitter.builder->create<RefLoadOp>(
          argValAndExpr.expr->getLocation(emitter), refVal);
    }
    assert(argValAndExpr.ir.isSValue() && "unknown irvalue");
    return argValAndExpr.ir.getSValueRegister();

  case ArgConvention::ReadMem: {
    // Promote PValue's if needed.
    Value result = checkMValueAddrSpace(
        emitter.emitMBValue(argValAndExpr, EC_CallArgValue));

    // Drop mutability for a MBValue.
    if (result && !cast<RefType>(result.getType()).isMutableKnown(false))
      result = emitter.builder->create<RefImmutOp>(
          argValAndExpr.expr->getLocation(emitter), result);
    return result;
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef:
    assert(argValAndExpr.ir.isMValue() &&
           "Ref args are already emitted to boxes during overload resolution");
    // These can be in any address space.
    return argValAndExpr.ir.getMValueReference();

  case ArgConvention::ByRefError: {
    // If the callee throws and is not async, we pass the contextual error
    // slot.
    MLValue errSlot = emitter.findNearestErrorSlot();
    if (!errSlot) {
      auto diag = emitter.emitError(callExpr->getLoc())
                  << "cannot call function that may raise in a context that "
                     "cannot raise"
                  << callExpr->getRange();
      diag.attachNote(callExpr->getLoc())
          << "try surrounding the call in a 'try' block";
      if (auto func = getBlockParentOfType<FnOp>(
              emitter.builder->getInsertionBlock())) {
        diag.attachNote(func.getLoc())
            << "or mark surrounding function as 'raises'";
      }
      return {};
    }
    return errSlot;
  }

  case ArgConvention::ByRefResult:
  case ArgConvention::Mut: {
    // byref_result can have a placeholder when there is no specified
    // destination, but can also have a destination specified directly.
    if (!argValAndExpr.ir) {
      auto resultRValueType = cast<RefType>(declaredArgType).getElementType();

      // Often the result of the call will be directly assigned into a
      // user-defined var or other location with existing storage.  In these
      // cases, we really want to assign directly into the existing slot.
      //
      // However, we cannot do that if the destination slot is also being
      // passed into the call as an input value, as in: `x = foo(x)` or `x = x
      // + 1`. In these cases we really do need a temporary+copy in the var
      // slot. At this point we've got enough information about the arguments
      // to make that assessment in a correct way.
      Value resultSlotVal;
      if (isSafeToUseValueDestForDirectResult(resultRValueType,
                                              callArgsSoFar)) {
        // Use the preferred location of the destination slot.
        resultSlotVal = dest.getMLValueForResult(callExpr->getLoc(),
                                                 resultRValueType, emitter);
      } else {
        auto loc = argValAndExpr.expr->getLocation(emitter);
        resultSlotVal =
            emitter.emitVarDecl("__call_result_tmp__", resultRValueType, loc,
                                VarDeclKind::Synthesized);
      }
      argValAndExpr.ir = MLValue(resultSlotVal);
    }

    // We know that the operand is an LValue, but it might be
    // dynamic/computed.
    LValue lv = argValAndExpr.ir.getIfLValue();
    assert(lv && "type checking ensures we will have an lvalue");

    // If this the first mutation to a def box, make sure to emit the box to
    // a local variable (materializing the value) before we emit a load.
    // Otherwise we'll load from the original argument and store back to the
    // box.  This is functionally correct, but weird and causes an additional
    // temporary to be generated because there is no MLValue.
    if (auto dlv = lv.getIfDLValue()) {
      lv = dlv->prepareForMutAccess(argValAndExpr.expr->getLoc(), emitter);
      if (!lv)
        return {};
    }

    // If this is already an MLValue in the default address space, we can pass
    // in the reference directly.
    if (auto ref = lv.getIfMLValue()) {
      if (lv.getMValueType().isDefaultAddrSpace())
        return checkMValueAddrSpace(ref);
    }

    // If dynamic, we need to generate a temporary slot, emit a 'get' into
    // that slot, pass the address, then write it back when we're done.
    ValueDest dlvBuffer(lv, EC_CallArgValue);
    MLValue mlvBuffer = dlvBuffer.getMLValueForResult(
        argValAndExpr.expr->getLoc(), lv.getRValueType(), emitter);
    // Emit the 'get' into the buffer.
    ValueDest bufferDest(mlvBuffer, EC_CallArgValue);
    if (!emitter.emitLoadOfLValue({lv, argValAndExpr.expr}, bufferDest)) {
      bufferDest.resetForError();
      dlvBuffer.resetForError();
      return {};
    }
    afterCallActions.lvalueWritebacks.push_back(
        {std::move(dlvBuffer), mlvBuffer});
    return checkMValueAddrSpace(mlvBuffer);
  }
  }

  llvm_unreachable("unexpected argument convention");
}

/// This function drops `byref_result` result slots from an argument list,
/// leaving only the formal arguments. This logic is valid for parameter calls
/// only.
static ArrayRef<ASTExprAnd<AnyValue>>
dropResultSlots(ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                FnTypeGeneratorType sig) {
  // TODO: What about throwing functions?
  if (sig.hasMemoryOnlyResult() &&
      sig.getNumArguments() == argumentValues.size())
    return argumentValues.drop_back();
  return argumentValues;
}

namespace {
/// Replace all dangling ("free") implicit origin references with the empty
/// origin union. This is used for emitting calls in the param context.
struct DanglingImplicitOriginRefEraser
    : IndexParameterReplacer<DanglingImplicitOriginRefEraser> {
  Type tryReplace(Type, size_t) { return {}; }
  Attribute tryReplace(Attribute attr, size_t depth) {
    auto ref = ::dyn_cast<ImplicitOriginRefAttr>(attr);
    if (!ref || ref.getDepth() < depth)
      return nullptr;
    return OriginUnionAttr::get({}, ref.getType());
  }

  /// Get this signature with all the implicit origins bound to the empty union.
  FnTypeGeneratorType replaceSignature(FnTypeGeneratorType sig) {
    FunctionType newFnType = replace(sig.getValues());
    return sig.getWithBody(FuncType::get(newFnType, sig.getArgConventions(),
                                         sig.getFnEffects(),
                                         sig.getFnMetadata()));
  }
};
} // namespace

TypedAttr CallEmitter::emitCallInParamContext(
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues) {
  assert(!emitter.builder && "not in parameter context");

  // TODO: We can support throwing parameter calls by inserting a 'force to
  // normal value' check which aborts (at compile time) if interpretation
  // throws an error.
  if (calleeSig.isThrows()) {
    return emitter.emitErrorForDynamicValueInParameter(
        callExpr, "TODO: cannot call potentially raising function");
  }
  if (calleeSig.isAsync()) {
    return emitter.emitErrorForDynamicValueInParameter(
        callExpr, "cannot call async function");
  }

  // Emitting a call in a parameter context. Generate an apply operator.
  SmallVector<TypedAttr> operands({callee.getIfPValue().get()});

  // If the callee has implicit origins, we need to bind them to immortal
  // references and rebind the callee.
  // FIXME: Extend apply to handle implicit origins directly, this makes it
  // super hard to read the generated IR because of redundant signatures.
  FnTypeGeneratorType boundSigType = calleeSig;
  std::optional<DanglingImplicitOriginRefEraser> implicitOriginRefEraser;
  if (calleeSig.getNumImplicitOriginDecls()) {
    implicitOriginRefEraser.emplace();
    boundSigType = implicitOriginRefEraser->replaceSignature(calleeSig);
    operands[0] =
        ParamOperatorAttr::get(POC::Rebind, operands[0], boundSigType);
  }

  auto argTypes = boundSigType.getArguments();
  auto argConventions = boundSigType.getArgConventions();
  // TODO: What about throwing functions?
  if (boundSigType.hasMemoryOnlyResult()) {
    argTypes = argTypes.drop_back();
    argConventions = argConventions.drop_back();
  }

  for (auto [argValAndExpr, calleeArgType, convention] :
       llvm::zip(argumentValues, argTypes, argConventions)) {
    PValue pValue = argValAndExpr.ir.getIfPValue();
    if (!pValue)
      return emitter.emitErrorForDynamicValueInParameter(argValAndExpr.expr);
    TypedAttr arg = pValue.get();
    if (implicitOriginRefEraser)
      arg = implicitOriginRefEraser->replace(arg);

    // Put memory-only arguments into memory ("PRValue" to "PLValue"
    // conversion).
    if (hasAddress(convention)) {
      auto immortal = OriginUnionAttr::get(arg.getContext());
      arg = StoreToMemAttr::get(arg, RefType::get(arg.getType(), immortal));
    }

    // Emit a rebind if the refined type does not match the callee arg type.
    if (arg.getType() != calleeArgType)
      arg = ParamOperatorAttr::get(POC::Rebind, arg, calleeArgType);
    operands.push_back(arg);
  }

  // Check to see if this is a call to a @always_inline("builtin") function,
  // like Int.__add__ etc.  If so, we need to inlined the body instead of making
  // an apply operator attr. We can only tell the inline level by finding the
  // lit.fn of the callee.  We require knowing the inline level because we have
  // to recursively resolve the body of the function, which we don't want to do
  // unilaterally.
  if (auto result = emitter.shared.foldInlineBuiltinFunction(operands, loc,
                                                             /*isError*/ false))
    return result;

  TypedAttr result;
  if (!boundSigType.hasMemoryOnlyResult()) {
    Type resultType = boundSigType.getResults().front();
    result = ParamOperatorAttr::get(POC::Apply, operands, resultType);
  } else {
    Type resultType =
        ASTType(boundSigType.getArguments().back()).getReferenceElementType();
    // ByRefResult uses ApplyResultSlot.
    result = ParamOperatorAttr::get(POC::ApplyResultSlot, operands, resultType);
  }

  // If the result was a returned reference, load it before returning it.
  if (boundSigType.isRefResult()) {
    result = ParamOperatorAttr::get(
        POC::LoadFromMem, result,
        cast<RefType>(result.getType()).getElementType());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ExprEmitter::emitCallUnchecked
//===----------------------------------------------------------------------===//

/// The results of calls to async functions are always bound to a `Coroutine`
/// type, or `RaisingCoroutine` type in the case of a raising function. This
/// function looks up the corresponding coroutine type and binds its result
/// type.
static ASTType getBoundCoroutineType(ASTDecl &declScope, const ExprNode *expr,
                                     FnTypeGeneratorType sig,
                                     TypedAttr origin) {
  auto &shared = declScope.getShared();
  SMLoc loc = expr->getLoc();
  ASTDecl *decl = sig.isThrows() ? shared.getBuiltinRaisingCoroutineType(loc)
                                 : shared.getBuiltinCoroutineType(loc);
  if (!decl) {
    shared.emitError(loc,
                     "internal error: could not find builtin 'Coroutine' type");
    return {};
  }
  // If the async function throws, extract the normal result type.
  ASTType resultType = ASTType(sig.getUserResultType());

  // Bind the result type to the base coroutine type.
  ParamBindings paramBinds(declScope);
  paramBinds.add(expr, PValue(resultType));
  paramBinds.add(expr, origin);

  auto structOp = cast<StructDeclOp>(decl);
  ParameterExprArrayAttr bindings = paramBinds.verifyBindings(
      structOp, structOp.getSignature(), expr->getLoc(), /*partial=*/false);
  if (!bindings)
    return {};

  return structOp.bindReference(bindings);
}

/// Emit warnings about incorrect code in a direct call.  This is invoked after
/// the full IR for the call is emitted, so we know that it was a valid call.
void CallEmitter::emitDirectCallWarnings(LIT::CallOp call,
                                         const CallOperands &callOperands) {
  // Check for a known callee.
  auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
  if (!symbol)
    return;

  // Figure out what is getting called.
  ASTDecl *calleeDecl =
      emitter.getDeclResolver().getDeclForFuncSymbol(symbol.getSymbol());
  if (!calleeDecl)
    return;
  auto calleeFunc = cast<FnOp>(*calleeDecl);

  // The __del__ special function takes its operand as an owning reference,
  // and destroys it.  It is a bit silly, but you can call it directly on an
  // RValue and it will destroy the RValue explicitly.  However, some folks
  // will call it on a local variable (or other !RValue reference) which will
  // actually cause a COPY of the source value, and then explicitly destroy
  // this copy of the value.  Emit a warning in this case.
  if (calleeFunc.getSpecialFunctionKind() == SpecialFunctionKind::kDel &&
      callOperands.size() == 1 && // defensive.
      callOperands[0].ir.getIfRValue().isNull()) {
    emitter.emitWarning(loc) << "explicit call to '__del__' destroys a copy of "
                                "the value; consider removing this call"
                             << callOperands[0].expr->getRange();
    return;
  }
}

// As we emit the arguments, we check to see if there are any exclusivity
// violations provided by the argument.
namespace {
struct ExclusivityChecker : public SharedStateUser {
  ExclusivityChecker(RValue callee, const ExprNode *callExpr, CallSyntax syntax,
                     ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                     ExprEmitter &emitter)
      : SharedStateUser(emitter.shared), callee(callee), callExpr(callExpr),
        syntax(syntax), argumentValues(argumentValues),
        builder(emitter.builder) {

    // Handle __unsafe_disable_nested_origin_exclusivity.
    isNestedOriginExclusivityCheckingDisabled =
        cast<FnTypeGeneratorType>(callee.getRValueType())
            .getIsNestedOriginExclusivityCheckingDisabled();

    // Check capture origins first so we know if argument values may overlap.
    checkCaptureOrigins();
  }

  /// As each argument is emitted, check against previous arguments for
  /// exclusivity violations.
  void checkArgument(Value val, unsigned argIdx, FnTypeGeneratorType signature);

private:
  RValue callee;
  const ExprNode *callExpr;
  CallSyntax syntax;
  /// These are the arguments that are being emitted.
  ArrayRef<ASTExprAnd<AnyValue>> argumentValues;
  std::optional<OpBuilder> builder;

  /// True if the __unsafe_disable_nested_origin_exclusivity decorator is
  /// on the callee.
  bool isNestedOriginExclusivityCheckingDisabled = false;

  /// For each origin that is referenced, we keep track of what argIdx it came
  /// from, and whether it was potentially mutated.
  struct OriginInfo {
    /// The argument that accessed this origin, or the capture set if null.
    std::optional<unsigned> argIdx;
    bool isImmut;
    /// True if this is the leaf of a nested access (e.g. "a.x.y.z"), false if
    /// this is the parent origin of a leaf access (e.g. "a.x" from that
    /// reference).
    bool isLeaf;
  };
  SmallDenseMap<TypedAttr, OriginInfo, 8> originAccesses;

  /// Look at the capture origin set on the callee and register uses of them.
  /// The capture origins are considered accessed as a single unit, so they
  /// never conflict with themselves, but they may conflict with argument
  /// accesses.
  void checkCaptureOrigins();

  void checkOriginAccess(Value val, std::optional<ArgConvention> convention,
                         std::optional<unsigned> argIdx, TypedAttr origin);

  void diagViolation(Value val, ArgConvention convention, unsigned argIdx,
                     TypedAttr origin, const OriginInfo &previousAccess);
};
} // end anonymous namespace

void ExclusivityChecker::checkCaptureOrigins() {
  TypedAttr captureOrigins =
      cast<FnTypeGeneratorType>(callee.getRValueType()).getCaptureOrigins();
  for (TypedAttr origin : shared.cachedOriginFinder.findOriginsIn(
           /*types=*/{}, captureOrigins))
    checkOriginAccess(Value(), /*convention=*/{}, /*argIdx=*/{}, origin);
}

/// Given an argument value being passed with a specified convention, check to
/// see if the following origin (which may be part of the argument convention,
/// or buried in the type) is a legal access given the other things we've
/// already seen.
void ExclusivityChecker::checkOriginAccess(
    Value val, std::optional<ArgConvention> convention,
    std::optional<unsigned> argIdx, TypedAttr origin) {
  // Determine whether the access was immutable.
  bool isImmut = cast<OriginType>(origin.getType()).isMutableKnown(false);

  // Look through immcasts to determine the accessed origin.
  origin = OriginMutCastAttr::strip(origin);

  // Accesses to the global origin never conflict.
  if (isa<AnyOriginAttr>(origin))
    return;

  // Determine whether we've seen this leaf origin before.
  auto [it, isNew] =
      originAccesses.insert({origin, {argIdx, isImmut, /*isLeaf=*/true}});
  if (!isNew) {
    assert(val && "capture origins cannot self-conflict");
    // If so, check to see if this access and the previous one were both
    // immutable.  Read/read aliasing is fine, but write/write and read/write
    // are not.
    if (!it->second.isImmut || !isImmut) {
      // If not, we have a problem!
      diagViolation(val, *convention, *argIdx, origin, it->second);
      return;
    }

    // Ok, this is a read/read conflict.  If this origin was previously seen
    // as a non-leaf, upgrade it to a leaf access, so any subsequent subfield
    // modifications are known to conflict.
    it->second.isLeaf = true;
  }

  // Ok, there is no direct conflict: scan up the parent structs to see if there
  // are conflicts for them.
  while (auto fieldAttr = dyn_cast<OriginFieldAttr>(origin)) {
    origin = fieldAttr.getBase();
    auto [it, isNew] =
        originAccesses.insert({origin, {argIdx, isImmut, /*isLeaf=*/false}});

    // If we have seen this parent origin before, check to see if it is ok.
    if (isNew)
      continue;

    // If the other access is a leaf access, then we are a subfield of it -
    // the access conflicts if either is a store.
    if (it->second.isLeaf && (!isImmut || !it->second.isImmut)) {
      assert(val && "capture origins cannot self-conflict");
      diagViolation(val, *convention, *argIdx, origin, it->second);
      return;
    }

    // Otherwise we have a non-conflicting access.  This can be because we
    // have a read of a subfield of another read, or because we have a
    // write/write or read/write of different subfields (e.g. 'a.x' vs 'a.y').
    // Either way it is fine, just make sure that we upgrade the interior
    // access to a write if our access is a write.
    it->second.isImmut &= isImmut;
  }
}

/// As each argument is emitted, check against previous arguments for
/// exclusivity violations.
void ExclusivityChecker::checkArgument(Value argVal, unsigned argIdx,
                                       FnTypeGeneratorType signature) {
  // We get passed the MLIR representation for the dynamic argument, which
  // includes variadic and pack constructions.  Make sure to handle each
  // variadic argument separately.
  auto checkArg = [&](Value argVal, ArgConvention convention) {
    // We sometimes get rebinds for downcasts of origins, e.g. to AnyOrigin.
    // Ignore those so we can see the actual incoming value's origin.
    if (auto rebind = argVal.getDefiningOp<RebindOp>())
      argVal = rebind.getOperand();

    // If this is a result argument, then we only look at the origin of the
    // destination that we're storing into, not any nested references that may
    // be in the result. This returned value is derived from the other arguments
    // passed to the function, it doesn't conflict with them.
    if (convention == ArgConvention::ByRefResult ||
        convention == ArgConvention::ByRefError) {
      checkOriginAccess(argVal, convention, argIdx,
                        cast<RefType>(argVal.getType()).getOrigin());
      return;
    }

    // Don't look at nested origins if checking for them has been explicitly
    // disabled.
    if (isNestedOriginExclusivityCheckingDisabled) {
      // DO check the origin of any in-memory arguments, we only ignore nested
      // origins.
      if (hasAddress(convention))
        checkOriginAccess(argVal, convention, argIdx,
                          cast<RefType>(argVal.getType()).getOrigin());
      return;
    }

    // Find all the of the origins that are buried in the specified type.
    for (TypedAttr origin :
         shared.cachedOriginFinder.findOriginsIn(argVal.getType()))
      checkOriginAccess(argVal, convention, argIdx, origin);
  };

  // Handle positional/homogenous variadics.
  if (signature.isPosVarArg(argIdx)) {
    // There are two ways to form a pos vararg:
    // VariadicSplatOp/VariadicCreateOp.  Unfurl these.
    SmallVector<Value> unpackedArgs;
    if (auto splat = argVal.getDefiningOp<POP::VariadicSplatOp>()) {
      // We know these are only created by the parser, so will have a concrete
      // element count.
      auto numElements = cast<IntegerAttr>(splat.getNumElements()).getInt();
      unpackedArgs.resize(numElements, splat.getOperand());
    } else if (auto vararg = argVal.getDefiningOp<POP::VariadicCreateOp>()) {
      assert(vararg && "only two ways to create a variadic list");
      unpackedArgs.append(vararg.getOperands().begin(),
                          vararg.getOperands().end());
    } else {
      // Zero elements
      assert(argVal.getDefiningOp<ParamConstantOp>() &&
             "Unknown way to create variadic list");
    }

    auto conv = cast<VariadicType>(argVal.getType()).getConvention();
    for (auto elt : unpackedArgs)
      checkArg(elt, conv);
    return;
  }

  // Normal arguments.
  if (!signature.isPack(argIdx)) {
    checkArg(argVal, signature.getArgConvention(argIdx));
    return;
  }

  // Handle variadic packs.
  auto packVal = RefPackCreateOp::findRefPackCreate(argVal);
  assert(packVal && "couldn't decode variadic pack information!");

  /// Zero argument packs are kgen.param.constant but they have no
  /// references anyway.
  if (packVal.getDefiningOp<ParamConstantOp>())
    return;

  auto pack = packVal.getDefiningOp<RefPackCreateOp>();
  assert(pack && "unknown variadic pack processing logic");
  auto conv = signature.getPackVarArgConvention(argIdx);
  for (auto packOperand : pack.getOperands())
    checkArg(packOperand, conv);
}

/// Emit an error about an access to a conflicting origin after a previous
/// access was seen.
void ExclusivityChecker::diagViolation(Value val, ArgConvention convention,
                                       unsigned argIdx, TypedAttr origin,
                                       const OriginInfo &previousAccess) {
  bool isImmut = cast<OriginType>(origin.getType()).isMutableKnown(false);
  InflightDiag diag = emitError(callExpr->getLoc());

  diag << "argument of ";

  switch (syntax) {
  default:
    // If the callee is a direct call, dig out the source name.
    if (PValue pv = callee.getIfPValue()) {
      if (auto symbol = dyn_cast<SymbolConstantAttr>(pv.get())) {
        // Figure out what is getting called and include it.
        if (ASTDecl *calleeDecl =
                getDeclResolver().getDeclForFuncSymbol(symbol.getSymbol())) {
          auto calleeFunc = cast<FnOp>(*calleeDecl);

          if (auto sourceName = calleeFunc.getSourceNameAttr())
            diag << sourceName << ' ';
        }
      }
    }
    diag << "call ";
    break;
  case CallSyntax::kImplicitConvert:
    diag << "implicit conversion ";
    break;
  case CallSyntax::kImplicitCopyInit:
    diag << "implicit __copyinit__ call ";
    break;
  case CallSyntax::kImplicitMoveInit:
    diag << "implicit __moveinit__ call ";
    break;
  }

  diag << "allows ";
  diag << (isImmut ? "reading" : "writing");
  diag << " a memory location previously ";
  diag << (previousAccess.isImmut ? "readable" : "writable");
  diag << " through ";

  // Add ranges for the two arguments.
  diag << argumentValues[argIdx].expr->getRange();
  if (std::optional<unsigned> prevIdx = previousAccess.argIdx) {
    diag << "another aliased argument";
    diag << argumentValues[*prevIdx].expr->getRange();
  } else {
    // TODO: Dig into the closure to get better error messages.
    diag << "implicit closure captures";
  }

  // Attach a note to explain what is going on in more detail.
  diag.attachNote(callExpr->getLoc());
  origin = OriginMutCastAttr::strip(origin);

  // If the origin in question is because of the top-level ref binding, then
  // we have a common problem where something is passed both mutable and
  // borrowed.
  if (hasAddress(convention) &&
      OriginMutCastAttr::strip(cast<RefType>(val.getType()).getOrigin()) ==
          origin) {
    diag << origin << " value is passed through aliasing '"
         << getUserSyntax(convention) << "' argument"
         << argumentValues[argIdx].expr->getRange();
    return;
  }

  ASTType argType = val.getType();
  if (hasAddress(convention))
    argType = argType.getReferenceElementType();

  // Otherwise, it is a more complicated buried origin in a type like a
  // Reference or Span.
  diag << origin
       << " memory accessed through reference embedded in value of type "
       << argType;

  // Diagnostics can be very confusing when generated by synthesized functions
  // like trait stubs.  Generate a note when this happens to make it more
  // obvious what is going on.
  if (builder && builder->getBlock()) {
    Block &block = *builder->getBlock();
    FnOp parentFunc = dyn_cast<FnOp>(block.getParentOp());
    if (!parentFunc)
      parentFunc = block.getParentOp()->getParentOfType<FnOp>();
    if (parentFunc && parentFunc.isSynthetic()) {
      // sourceName
      diag.attachNote(parentFunc.getLoc()) << "in synthesized method";
      if (auto sourceName = parentFunc.getSourceNameAttr())
        diag << ' ' << sourceName;
    }
  }
}

/// When emitting a call where all of the arguments are PValues, and the callee
/// is @always_inline("builtin"), we can safely emit the call in a parameter
/// context.  We know it doesn't have side effects because of the checks that
/// @always_inline("builtin") performs.
static bool shouldEmitParameterCall(RValue callee,
                                    ArrayRef<ASTExprAnd<AnyValue>> argValues,
                                    SharedState &shared) {
  auto calleeSig = cast<FnTypeGeneratorType>(callee.getRValueType());
  argValues = dropResultSlots(argValues, calleeSig);

  // We cannot inline this if any of the arguments are dynamic.
  auto isPValue = [](ASTExprAnd<AnyValue> arg) { return arg.ir.getIfPValue(); };
  if (!callee.getIfPValue() || !llvm::all_of(argValues, isPValue))
    return false;

  // If this is an @always_inline("builtin") function, we must emit its body
  // inline.
  if (auto calleeSymbolCst = dyn_cast<SymbolConstantAttr>(
          ParamOperatorAttr::stripRebind(callee.getIfPValue()))) {
    if (ASTDecl *calleeDecl = shared.getDeclResolver().getDeclForFuncSymbol(
            calleeSymbolCst.getSymbol())) {
      if (cast<FnOp>(*calleeDecl).getInlineLevel() ==
          InlineLevel::AlwaysBuiltin)
        return true;
    }
  }
  return false;
}

/// Compute the union of all references origins in a set of function call
/// arguments.
static TypedAttr computeArgumentsOrigin(AsyncCallOp call,
                                        CachedOriginFinder &originFinder) {
  SmallVector<std::pair<Value, OperandEffect>> operands;
  SmallVector<ResultEffect> results;
  SmallVector<TypedAttr> origins;
  // Check origin accesses on the types. We need to forward this to the
  // coroutine since it is a transitive capture.
  LIT::getOperationEffects(*call, operands, results, origins, originFinder);
  // Collect the implicit origins of the arguments.
  for (Value value : call.getOperands())
    if (auto ref = dyn_cast<RefType>(value.getType()))
      origins.push_back(ref.getOrigin());
  return OriginSetAttr::get(call.getContext(), origins);
}

CValue ExprEmitter::emitCallUnchecked(RValue callee,
                                      const CallOperands &callOperands,
                                      ValueDest &dest, CallSyntax syntax,
                                      const ExprNode *callExpr) {
  CallEmitter callEmitter(callee, callExpr, *this, dest);
  auto calleeSig = cast<FnTypeGeneratorType>(callee.getRValueType());

  // We first emit all the arguments.
  FailureOr<SmallVector<ASTExprAnd<AnyValue>>> argumentValuesOr =
      callEmitter.emitArgValues(callOperands);
  if (failed(argumentValuesOr)) {
    dest.resetForError();
    return {};
  }
  ArrayRef<ASTExprAnd<AnyValue>> argumentValues = *argumentValuesOr;

  if (!builder || shouldEmitParameterCall(callee, argumentValues, shared)) {
    TypedAttr paramCallResult;
    {
      llvm::SaveAndRestore savedBuilder(builder, {});
      assert(dest.getContext() != EC_InvalidContext &&
             "parametric emitCallUnchecked must include an ExprContext");
      llvm::SaveAndRestore savedContext(paramContext, dest.getContext());
      argumentValues = dropResultSlots(argumentValues, calleeSig);
      paramCallResult = callEmitter.emitCallInParamContext(argumentValues);
    }
    // The dest might force further calls.  We delay calling it until after
    // restoring the builder so that it is NOT forced to be in the parameter
    // context.  In particular, dest may cause a call to set the paramCallResult
    // into a DLValue.
    CValue result = emitCResult(paramCallResult, callExpr, dest);
    if (!result)
      dest.resetForError();
    return result;
  }

  // Otherwise, materialize PValue and DLValue's as SSA values for emission.
  Location loc = translateLocation(callExpr->getLoc());

  // As we emit the arguments, we check to see if there are any exclusivity
  // violations provided by the argument.
  ExclusivityChecker exclusivityChecker(callee, callExpr, syntax,
                                        argumentValues, *this);

  SmallVector<Value> callArgs;
  SmallVector<TypedAttr> implicitOrigins;
  ArrayRef<ArgConvention> conventions = calleeSig.getArgConventions();

  for (auto [argIdx, argValAndExpr, conventionX, declaredArgTypeX] :
       llvm::enumerate(argumentValues, conventions, calleeSig.getArguments())) {
    ArgConvention convention = conventionX;
    Type declaredArgType = declaredArgTypeX;

    // If this is a variadic operation, the N operands have already been emitted
    // together and consolidated into a pop.variadic.create/pop.variadic.attr,
    // which is emitted as an SRValue instead of whatever the underlying type
    // is.
    if (calleeSig.isPosVarArg(argIdx))
      convention = ArgConvention::ReadReg;

    // Owned and borrowed packs are passed as expected, but mut and read
    // are passed borrowed.
    if (calleeSig.isPack(argIdx) && convention != ArgConvention::OwnedMem)
      convention = ArgConvention::ReadMem;

    if (isResultSlot(convention)) {
      // Async function signatures have results slots even though they are not
      // actually provided.
      // TODO: Why are these in the signature, why do they take implicit
      // origins for these things?
      if (calleeSig.isAsync()) {
        implicitOrigins.push_back(OriginUnionAttr::get(getContext()));
        continue;
      }

      // 'ref' results can have origins derived from implicit origins of
      // earlier arguments, and can be passed ByRefResult in throwing functions.
      // Make sure to remap the implicit origins into place.  This works
      // because ByRefResult is at the end of the list.
      if (convention == ArgConvention::ByRefResult) {
        implicitOrigins.push_back(
            AnyOriginAttr::get(getContext(), /*isMutable=*/true));
        FunctionType remappedCalleeType =
            calleeSig.substituteImplicitOriginsIntoValues(
                implicitOrigins, [&]() -> InFlightDiagnostic {
                  llvm_unreachable("substitution should always succeed");
                });
        implicitOrigins.pop_back();
        declaredArgType = remappedCalleeType.getInput(argIdx);
      }
    }

    Value arg = callEmitter.emitPreemittedArgumentAsDynamicValue(
        argValAndExpr, convention, declaredArgType, callArgs);
    if (!arg) {
      dest.resetForError();
      return {};
    }

    // VariadicPack also includes the implicit origin for the elements, which
    // is different than the origin for the pack itself (when passed through
    // memory).
    if (ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx)) {
      ASTType argRVType = arg.getType();
      if (hasAddress(convention))
        argRVType = argRVType.getReferenceElementType();

      // Include the union origin that covers all the values.
      implicitOrigins.push_back(
          argRVType.getVariadicPackInfo(shared).getOrigin());
    }

    // See if we have an implicit origin bound for this argument.
    if (hasImplicitOrigin(convention)) {
      implicitOrigins.push_back(cast<RefType>(arg.getType()).getOrigin());
    } else if (calleeSig.isPosVarArg(argIdx)) {
      // If this is a variadic, it will have a wrapper around the ref.
      auto eltType = ASTType(arg.getType()).getVariadicElementType();
      if (auto refType = dyn_cast<RefType>(eltType))
        implicitOrigins.push_back(refType.getOrigin());
    }

    // The argument looks good on its own, check to see if it is an exclusivity
    // violation with a previous argument.
    exclusivityChecker.checkArgument(arg, argIdx, calleeSig);

    // All looks good!
    callArgs.push_back(arg);
  }

  // Now that we have the origins for the arguments, we can calculate what the
  // substituted signature should be.
  FunctionType expectedCalleeType =
      calleeSig.substituteImplicitOriginsIntoValues(
          implicitOrigins, [&]() -> InFlightDiagnostic {
            llvm_unreachable("substitution should always succeed");
          });

  // Now that all of the arguments have been emitted, coerce them to the
  // expected type if needed.  We do this after the first pass above, because
  // there can be forward references from the result slot to the later
  // arguments' origins.
  for (auto [arg, expectedType] :
       llvm::zip(callArgs, expectedCalleeType.getInputs())) {
    // Make sure the parameters of an argument line up by emitting a rebind
    // operation.
    if (arg.getType() == expectedType)
      continue;

    // Use rebindValue to do this so we get assertions and checks.
    arg = rebindValue({SRValue(arg), callExpr}, expectedType).getIfSRValue();
    assert(arg && "rebindValue always succeeds");
  }

  assert(expectedCalleeType.getResults().size() == 1 &&
         "All mojo functions return one value");
  Type resultType = expectedCalleeType.getResults()[0];
  CValue callResult;
  if (auto target = callee.getIfPValue()) {
    if (calleeSig.isAsync()) {
      // If the callee is an async function, emit an async call. Then wrap the
      // `!co.routine<T>` result in a `Coroutine[T]` object.
      auto call = builder->create<AsyncCallOp>(loc, target.get(),
                                               implicitOrigins, callArgs);
      ASTType coroType = getBoundCoroutineType(
          getDeclScope(), callExpr, calleeSig,
          computeArgumentsOrigin(call, shared.cachedOriginFinder));

      if (!coroType) {
        dest.resetForError();
        return {};
      }
      // Emit the implicit conversion to Coroutine[T].  We emit into the call's
      // destination to avoid an extra copy/move of the Coroutine object.
      callResult =
          emitConstructorCall(coroType, {{{SRValue(call), callExpr}}}, callExpr,
                              CallSyntax::kImplicitConvert, dest);
      if (!callResult) {
        dest.resetForError();
        return {};
      }
    } else {
      auto call = builder->create<CallOp>(loc, resultType, target.get(),
                                          implicitOrigins, callArgs);
      callResult = SRValue(call.getResult(0));

      // If there are any callee-specific warnings to emit, do so after
      // successfully emitting the call.
      callEmitter.emitDirectCallWarnings(call, callOperands);
    }
  } else {
    // TODO(MOCO-788): We need a `lit.async.call_indirect` to model indirect
    // async calls.
    if (calleeSig.isAsync()) {
      emitError(callExpr->getLoc())
          << "TODO: indirect calls to async functions not yet supported";
      return {};
    }
    // If the callee isn't a PValue, it must be a dynamic callee.
    Value calleeVal = emitSRValue({callee, callExpr}, EC_CallCalleeValue);
    assert(calleeVal && "don't have a callee of expected type");
    auto call = builder->create<CallIndirectOp>(loc, resultType, calleeVal,
                                                implicitOrigins, callArgs);
    callResult = SRValue(call.getResult(0));
  }

  // If there were any writebacks to handle, emit them before handling raised
  // errors.
  callEmitter.emitAfterCallActions();

  // If there is a memory result slot, the value we filled in is our MRValue
  // result and we've already handled the ValueDest by emitting into it.
  if (calleeSig.hasMemoryOnlyResult() && !calleeSig.isAsync()) {
    callResult = MRValue(callArgs.back());
  }

  // If returning a reference, we need to convert to an MValue from
  // the SRValue we've got.
  if (calleeSig.isRefResult()) {
    auto resultVal = emitSRValue({callResult, callExpr}, EC_CallCalleeValue);
    if (!resultVal) {
      dest.resetForError();
      return {};
    }

    // Use the appropriate classification for the value based on its mutability.
    callResult = CValue::getMValueForRef(resultVal);
  }

  // Otherwise, register-passable results are the call result which may need to
  // be emitted into a ValueDest.
  return emitCResult(callResult, callExpr, dest);
}
