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

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "MojoUtils.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
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

/// This helper function emits a call to VariadicPack(refPackValue, isOwned)
/// and returns the result value.  'variadicPackType' is the fully bound
/// VariadicPack type per the function signature.
static CValue emitVariadicPackConstructor(
    ASTType variadicPackType, ArgConvention declaredArgConvention,
    TypedAttr lifetimeToUse, const ExprNode *expr, ExprEmitter &emitter,
    std::function<CValue(RefPackType)> refPackBuilder) {
  RefPackType packType = variadicPackType.getVariadicPackInfo();

  // If there was no lifetime specified, use an immortal one with the same
  // mutability.
  if (!lifetimeToUse)
    lifetimeToUse = LifetimeAttr::get(packType.getLifetime().getType());

  // Rebind the !lit.ref.pack with the common lifetime.
  packType = RefPackType::get(packType.getVariadic(), lifetimeToUse,
                              packType.getAddressSpace());

  // Build the !lit.ref.pack or #lit.ref.pack value with the adjusted lifetime.
  CValue refPackValue = refPackBuilder(packType);

  auto isOwned = declaredArgConvention == ArgConvention::OwnedInMem;
  auto isOwnedAttr = BoolAttr::get(emitter.getContext(), isOwned);

  // Emit a VariadicPack constructor call taking the #lit.ref.pack and a
  // bool indicating whether the argument is owned.
  CallOperands operands;
  operands.add({refPackValue, expr});
  operands.add({isOwnedAttr, expr});

  ValueDest packDest(ExprContext::EC_PackArgument);

  // Construct the pack type without parameters so we reinfer the lifetime which
  // is different on the caller side (the union of the argument lifetimes) than
  // the declared callee side (a parameter).
  variadicPackType = variadicPackType.getWithoutParameters(emitter.shared);

  auto callResult =
      emitter.emitConstructorCall(variadicPackType, std::move(operands), expr,
                                  CallSyntax::kTypeCall, packDest);

  if (isOwned)
    return callResult;
  // RValue->BValue decay if we're passing VariadicPack as an SBValue.
  return emitter.emitBValue({callResult, expr}, ExprContext::EC_PackArgument);
}

//===----------------------------------------------------------------------===//
// CallEmitter
//===----------------------------------------------------------------------===//

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

  // Emit all the lit.ownership.use ops.
  OpBuilder &b = *emitter.builder;
  for (auto [value, alloc] : valuesToKeepAlive) {
    b.create<OwnershipUseOp>(callEmitter.loc, value);
    b.create<POP::StackAllocLifetimeEndOp>(callEmitter.loc, alloc);
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
    RefPackType packType = variadicPackType.getVariadicPackInfo();

    // Operands being applied to a concrete pack type argument must be
    // converted to the pack element type at that index.  The calleeSig has the
    // pack type resolved to a concrete list of types it is expecting.
    expectedType =
        ASTType(packType.getVariadicIfResolved().getValues()[sequenceIndex]);
    // Get the !lit.ref with the lifetime and other paraphernalia.
    expectedType = packType.getElementRefTypeFor(expectedType);
    convention = calleeSig.getPackVarArgConvention(argIdx);
  }

  switch (convention) {
  case ArgConvention::InOut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
  case ArgConvention::InitSelf:
    // By-ref arguments, must be lvalues.
    assert(operand.ir.getIfLValue() && "Call should already be type checked");
    return operand.ir;
  case ArgConvention::OwnedInReg:
  case ArgConvention::OwnedInMem:
    // Owned conventions pass rvalues.
    if (convention == ArgConvention::OwnedInMem)
      expectedType = cast<RefType>(expectedType).getElementType();
    return emitter.emitRValue(operand, EC_CallArgValue, expectedType);

  case ArgConvention::Ref: {
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

    // Lifetimes must be convertible, this is checked by OverloadFitness.
    // The destination may be less mutable because of canConvertWithRebind.
    // This also lazy materializes cast to immutable that MBValue avoided.
    if (!refValueType.isMutableKnown(false) &&
        expectedRefType.isMutableKnown(false)) {
      refValue = emitter.builder->create<RefImmutOp>(
          operand.expr->getLocation(emitter), refValue);
      refValueType = cast<RefType>(refValue.getType());
    }

    // The lifetimes may disagree if we're converting a value to a
    // superset lifetime, e.g. "immortal -> X" or "X -> X|y".
    if (refValueType.getLifetime() != expectedRefType.getLifetime()) {
      refValue = emitter.builder->create<RebindOp>(
          operand.expr->getLocation(emitter),
          refValueType.getWithLifetime(expectedRefType.getLifetime()),
          refValue);
      refValueType = cast<RefType>(refValue.getType());
    }
    // The element types may disagree if we're dealing with ParamRef types
    // downcast from a trait type to AnyType.
    if (refValueType.getElementType() != expectedRefType.getElementType()) {
      assert(isa<ParamRefType>(refValueType.getElementType()) &&
             isa<ParamRefType>(expectedRefType.getElementType()) &&
             "Unknown element type mismatch in ref binding");
      refValue = emitter.builder->create<RebindOp>(
          operand.expr->getLocation(emitter),
          refValueType.getWithElement(expectedRefType.getElementType()),
          refValue);
      refValueType = cast<RefType>(refValue.getType());
    }

    assert(refValueType == expectedType && "Should have exact match now");
    return CValue::getMValueForRef(refValue);
  }

  case ArgConvention::BorrowedInReg:
  case ArgConvention::BorrowedInMem:
    // by-ref arguments are converted to the expected r-value type.
    if (convention == ArgConvention::BorrowedInMem)
      expectedType = cast<RefType>(expectedType).getElementType();

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
      if (SignatureType::hasAddress(convention)) {
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
          variadicPackType, convention, /*lifetime*/ {}, callExpr, emitter,
          [&](RefPackType adjustedPackType) -> CValue {
            // RefPack elements are passed through memory.  Use adjustedPackType
            // to get the proper (immortal) lifetime installed.
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
    Value argVal =
        emitPreemittedArgumentAsDynamicValue(operand, convention, expectedType);
    if (!argVal)
      return failure();
    args.push_back(argVal);

    // Variadic and pack arguments are always passed through memory. An
    // exception was carved out for trivial register-passable values, which
    // don't require lifetime tracking.
    // TODO(MOCO-726): Make variadics always pass through memory.
    if (SignatureType::hasAddress(convention) ||
        ASTType(argVal.getType()).isTrivial(callExpr->getLoc(), emitter.shared))
      continue;
    emitter.shared.emitError(
        operand.expr->getLoc(),
        "cannot bind non-trivial value to trivial variadic argument");
  }

  // If there are lifetimes on anything, create a uniform representation and
  // cast to a common reference type.
  if (!args.empty() && isa<RefType>(args.back().getType())) {
    // If one arg is a reference, then they all are.
    SmallVector<TypedAttr> refLifetimes;
    for (auto arg : args)
      refLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());

    // All the lifetimes will have the same LifetimeType, indicating the
    // reference mutability that the callee expected.
    LifetimeType commonLifetimeType =
        cast<LifetimeType>(refLifetimes.back().getType());

    // If there is more than one element, they probably have different
    // lifetimes, and thus need to be rebound into a common union of them.
    auto commonLifetime =
        LifetimeUnionAttr::get(refLifetimes, commonLifetimeType);
    for (auto &arg : args) {
      auto argType = cast<RefType>(arg.getType());
      if (argType.getLifetime() == commonLifetime)
        continue; // Already the right lifetime.
      // Cast to common lifetime with a rebind.
      arg = emitter.builder->create<RebindOp>(
          loc, argType.getWithLifetime(commonLifetime), arg);
    }
  }

  // Given a reference type for a variadic list of pack element, return the same
  // type updated to the common lifetime of the elements.
  auto getCommonLifetime = [&]() -> TypedAttr {
    if (!args.empty())
      return cast<RefType>(args.back().getType()).getLifetime();
    return {};
  };

  CValue argVal;
  if (isPosVarArg) { // Positional homogenous varargs
    // Rebind the lifetime of the argument to the expected lifetime if needed.
    auto expectedVararg = cast<VariadicType>(expectedType);
    if (auto refType = dyn_cast<RefType>(expectedVararg.getElementType())) {
      auto lifetime = getCommonLifetime();
      if (!lifetime) // No arguments, use immortal with same mutability.
        lifetime = LifetimeAttr::get(refType.getLifetime().getType());

      refType = refType.getWithLifetime(getCommonLifetime());
      expectedType = VariadicType::get(refType, expectedVararg.getConvention());
    }

    // Check for a splat.
    if (!args.empty() &&
        llvm::all_of(args, [&](Value operand) { return operand == args[0]; })) {
      argVal = SBValue(emitter.builder->create<POP::VariadicSplatOp>(
          loc, expectedType, args[0], args.size()));
    } else {
      argVal = SBValue(emitter.builder->create<POP::VariadicCreateOp>(
          loc, expectedType, args));
    }
  } else {
    // Bundle them up into a VariadicPack instance.
    ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx);
    assert(variadicPackType && "Must be a VariadicPack");
    argVal = emitVariadicPackConstructor(
        variadicPackType, convention, getCommonLifetime(), callExpr, emitter,
        [&](RefPackType adjustedPackType) -> CValue {
          return SBValue(emitter.builder->create<RefPackCreateOp>(
              loc, adjustedPackType, args));
        });
    if (!argVal)
      return failure();
  }
  argumentValues.push_back({argVal, remainingOperands[0].expr});
  return success();
}

/// Return true if this operand is a placeholder initself that is produced for
/// constructor calls.  Return false it if it something explicitly specified,
/// e.g. when a constructor is invoked with `self.__init__()`
static bool isPlaceholderInitSelf(const AnyValue &value) {
  if (!value)
    return true;
  if (auto pv = value.getIfPValue())
    return isa<UnknownAttr>(pv.get());
  return false;
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
  for (auto [argIdx, expectedTypeX, convention, pogAttr] :
       llvm::enumerate(calleeSig.getArguments(), calleeSig.getArgConventions(),
                       argListAttr.getPogs())) {
    // Use a ParserParamEvaluator to fold only 'apply' expressions. Emit a
    // rebind if the refined type is different than the expected type.
    Type expectedType = evaluator.refine(expectedTypeX);

    // If this is the return slot for a call, we need a temporary to emit into,
    // but don't know the type until the arguments (and their lifetimes) are all
    // emitted. Just skip over it for now.
    if (SignatureType::isResultSlot(convention)) {
      assert(calleeSig.hasMemoryOnlyResult() ||
             (calleeSig.isThrows() &&
              pogAttr.getPassingKind() == PassingKind::Implicit));
      argumentValues.push_back({AnyValue(), callExpr});
      continue;
    }

    // If this is an `init_self` slot that has not been explicitly provided by
    // the user, we will have to find a slot later.
    if (convention == ArgConvention::InitSelf &&
        isPlaceholderInitSelf(operands.values.front().ir)) {
      ++posOperandIdx;
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
      if (!calleeSig.isPosVarArg(argIdx) && !calleeSig.isPackVarArg(argIdx)) {
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
      assert(variadicPackType.getVariadicPackInfo()
                 .getVariadicIfResolved()
                 .getValues()
                 .empty() &&
             "pack type already checked against operand count");
      auto argConv = calleeSig.getPackVarArgConvention(argIdx);
      // Emit a VariadicPack constructor call.
      auto variadicPack = emitVariadicPackConstructor(
          variadicPackType, argConv, /*lifetime*/ {}, callExpr, emitter,
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
          CallSyntax::kImplicitConvert, dictDest,
          /*allowImplicitConversion=*/false);
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
    assert(convention != ArgConvention::InOut &&
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

    // We first construct a String key from the operand name.
    ASTType stringLiteralType =
        emitter.shared.getBuiltinStringLiteralType(emitter.declScope, loc);
    auto nameAttr = StringAttr::get(operand.keyword.strref(),
                                    StringType::get(emitter.getContext()));
    CValue literalKey = emitter.emitConstructorCall(
        stringLiteralType, CallOperands({{PValue(nameAttr), operand.expr}}),
        callExpr, CallSyntax::kImplicitConvert, kwargsDest);

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

  // See if the destination buffer is something that ownership can track.  If
  // not, we cannot make reliable determinations about aliasing.
  Value underlyingDest =
      LifetimeTrackable::findUnderlyingValueFromField(destBuffer);
  if (!underlyingDest)
    return false;

  // If this is a throwing function, then we cannot write to a field of a
  // lifetime tracked value.  Consider:
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
  if (calleeSig.isThrows() && underlyingDest != destBuffer)
    return false;

  // Collect all of the types of all the arguments so we can collect the
  // lifetimes they may reference.
  SmallVector<Type> argTypes;
  for (auto [value, convention] :
       llvm::zip(argValues, calleeSig.getArgConventions())) {
    if (SignatureType::isResultSlot(convention) ||
        convention == ArgConvention::InitSelf)
      continue;

    argTypes.push_back(value.getType());
  }

  TypedAttr destLifetime =
      cast<RefType>(underlyingDest.getType()).getLifetime();

  // Check to see if any of the the lifetimes they may be accessing are the
  // lifetime in question.  If any of them is a possible reference to the
  // destination slot, then we must fail.
  CachedTypeLifetimeFinder &finder = emitter.shared.cachedLifetimeFinder;
  for (TypedAttr lifetime : finder.findLifetimesInTypes(argTypes)) {
    // If an operand is reading from the lifetime, there will be an immcast in
    // the way.  Look through it.
    lifetime = LifetimeMutCastAttr::strip(lifetime);
    if (lifetime == destLifetime)
      return false;
  }

  // If no problems are found, it is safe!
  return true;
}

Value CallEmitter::emitPreemittedArgumentAsDynamicValue(
    ASTExprAnd<AnyValue> argValAndExpr, ArgConvention convention,
    Type declaredArgType) {
  assert(emitter.builder && "Should only be called in dynamic context");

  Value arg;
  switch (convention) {
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    llvm_unreachable("this is handled specially during call emission");
  case ArgConvention::OwnedInReg:
    // Promote PValue's if needed.
    return emitter.emitSRValue(argValAndExpr, EC_CallArgValue);
  case ArgConvention::OwnedInMem:
    // Promote PValue's if needed.
    return emitter.emitMRValue(argValAndExpr, EC_CallArgValue);
  case ArgConvention::BorrowedInReg:
    if (auto pVal = argValAndExpr.ir.getIfPValue())
      return emitter.emitSRValue(argValAndExpr, EC_CallArgValue);

    // If this is an MBValue, the element must be register passable but not
    // loaded.
    if (argValAndExpr.ir.isMValue()) {
      auto refVal = argValAndExpr.ir.getMValueReference();
      auto load = emitter.builder->create<RefLoadOp>(
          argValAndExpr.expr->getLocation(emitter), refVal);
      argValAndExpr.ir = SBValue(load);
    }

    arg = argValAndExpr.ir.getIfSBValue();
    assert(arg && "unknown BValue");
    return arg;

  case ArgConvention::BorrowedInMem: {
    if (SBValue sbValue = argValAndExpr.ir.getIfSBValue()) {
      // "Convert" an SBValue to an MBValue by performing a bitcopy of the value
      // into an untracked stack allocation.
      // FIXME(MOCO-725): This doesn't work in async functions, because the
      // borrowed argument is captured.
      if (calleeSig.isAsync()) {
        emitter.emitError(argValAndExpr.expr->getLoc())
            << "TODO: cannot bind non-trivial register-passable value to "
               "borrowed generic argument yet";
      }
      const ExprNode *expr = argValAndExpr.expr;
      Location argLoc = expr->getLocation(emitter);
      Value ptr = emitter.builder->create<POP::StackAllocationOp>(
          argLoc, PointerType::get(sbValue.getType()), 1,
          /*markedLifetimes=*/true);
      emitter.builder->create<POP::StackAllocLifetimeStartOp>(argLoc, ptr);
      emitter.builder->create<POP::StoreOp>(argLoc, sbValue, ptr);
      auto immortal = emitter.builder->getAttr<LifetimeAttr>(/*isMut=*/false);
      auto ref =
          emitter.builder->create<RefFromPointerOp>(argLoc, ptr, immortal,
                                                    /*startUninit=*/false,
                                                    /*endUninit=*/false);
      // Because the result of StackAllocationOp is not a lifetime trackable,
      // StoreOp will not transfer ownership and we must manually extend the
      // lifetime of the SBValue.
      afterCallActions.valuesToKeepAlive.emplace_back(sbValue, ptr);
      return MBValue(ref);
    }
    // Promote PValue's if needed.
    Value result = emitter.emitMBValue(argValAndExpr, EC_CallArgValue);
    // Drop mutability for a MBValue.
    if (result && !cast<RefType>(result.getType()).isMutableKnown(false))
      result = emitter.builder->create<RefImmutOp>(
          argValAndExpr.expr->getLocation(emitter), result);
    return result;
  }
  case ArgConvention::Ref:
    assert(argValAndExpr.ir.isMValue() &&
           "Ref args are already emitted to boxes during overload resolution");
    return argValAndExpr.ir.getMValueReference();

  case ArgConvention::InOut:
  case ArgConvention::InitSelf: {
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
      lv = dlv->prepareForInoutAccess(argValAndExpr.expr->getLoc(), emitter);
      if (!lv)
        return {};
    }

    // If this is already an MLValue in the default address space, we can pass
    // in the reference directly.
    if (auto ref = lv.getIfMLValue()) {
      if (lv.getMValueType().isDefaultAddrSpace())
        return ref;
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
    return mlvBuffer;
  }
  }
  llvm_unreachable("unexpected argument convention");
}

/// This function drops `init_self` or `byref_result` result slots from an
/// argument list, leaving only the formal arguments. This logic is valid for
/// parameter calls only.
template <typename T>
static ArrayRef<T> dropResultSlots(ArrayRef<T> argumentValues,
                                   SignatureType sig) {
  if (sig.hasInitSelfArg() && sig.getNumArguments() == argumentValues.size())
    return argumentValues.drop_front();
  if (sig.hasMemoryOnlyResult() &&
      sig.getNumArguments() == argumentValues.size())
    return argumentValues.drop_back();
  return argumentValues;
}

FailureOr<CValue> CallEmitter::inlineFunctionCallIntoPValueIfPossible(
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues) {
  if (calleeSig.isThrows() || calleeSig.isAsync())
    return failure();
  auto calleePR = callee.getIfPValue();
  if (!calleePR)
    return failure();
  auto calleeSymbolCst = evaluator.findDirectCallee(calleePR.get());
  if (!calleeSymbolCst)
    return failure();

  // When emitting a call in a dynamic context to function with an `init_self`
  // or `byref_result` argument, the caller sets up an MLValue destination or
  // may pass in a placeholder value. Make sure to drop them before calling into
  // the interpreter.
  argumentValues = dropResultSlots(argumentValues, calleeSig);
  ArrayRef<ArgConvention> conventions =
      dropResultSlots(calleeSig.getArgConventions(), calleeSig);
  ArrayRef<Type> types = dropResultSlots(calleeSig.getArguments(), calleeSig);

  SmallVector<Attribute> arguments;
  for (auto [argValue, conv, type] :
       llvm::zip(argumentValues, conventions, types)) {
    auto pValue = argValue.ir.getIfPValue();
    if (!pValue || !ParameterAttr::isSimpleConstant(pValue.get()))
      return failure();
    arguments.push_back(SignatureType::hasAddress(conv)
                            ? StoreToMemAttr::get(pValue, type)
                            : pValue);
  }

  FailureOr<TypedAttr> res =
      evaluator.evaluateFunctionCall(calleeSymbolCst.getSymbol(), arguments);
  if (failed(res))
    return failure();
  TypedAttr resultValue = *res;

  // If the result was a returned reference, load it before returning it.
  if (calleeSig.isRefResult()) {
    resultValue = ParamOperatorAttr::get(
        POC::LoadFromMem, resultValue,
        cast<RefType>(resultValue.getType()).getElementType());
  }

  return emitter.emitCResult(resultValue, callExpr, dest);
}

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

  // If the callee has implicit lifetimes, we need to bind them to immortal
  // references and rebind the callee.
  LITSignatureType boundSigType = calleeSig;
  if (calleeSig.getNumImplicitLifetimeDecls()) {
    boundSigType = calleeSig.getWithImplicitLifetimesBoundImmortal();
    operands[0] =
        ParamOperatorAttr::get(POC::Rebind, operands[0], boundSigType);
  }

  auto argTypes = boundSigType.getArguments();
  auto argConventions = boundSigType.getArgConventions();
  if (boundSigType.hasMemoryOnlyResult()) {
    argTypes = argTypes.drop_back();
    argConventions = argConventions.drop_back();
  }
  if (boundSigType.hasInitSelfArg()) {
    argTypes = argTypes.drop_front();
    argConventions = argConventions.drop_front();
  }

  for (auto [argValAndExpr, calleeArgType, convention] :
       llvm::zip(argumentValues, argTypes, argConventions)) {
    PValue pValue = argValAndExpr.ir.getIfPValue();
    if (!pValue)
      return emitter.emitErrorForDynamicValueInParameter(argValAndExpr.expr);
    TypedAttr arg = pValue.get();
    // Put memory-only arguments into memory ("PRValue" to "PLValue"
    // conversion).
    if (SignatureType::hasAddress(convention)) {
      arg = StoreToMemAttr::get(
          arg, RefType::getImmortal(arg.getType(), /*isMut=*/true));
    }

    // Emit a rebind if the refined type does not match the callee arg type.
    if (arg.getType() != calleeArgType)
      arg = ParamOperatorAttr::get(POC::Rebind, arg, calleeArgType);
    operands.push_back(arg);
  }

  TypedAttr result;
  if (!boundSigType.hasMemoryOnlyResult() && !boundSigType.hasInitSelfArg()) {
    Type resultType = boundSigType.getResults().front();
    result = ParamOperatorAttr::get(POC::Apply, operands, resultType);
  } else {
    Type resultType;
    if (boundSigType.hasMemoryOnlyResult())
      resultType =
          ASTType(boundSigType.getArguments().back()).getReferenceElementType();
    else {
      assert(boundSigType.hasInitSelfArg());
      resultType = ASTType(boundSigType.getArguments().front())
                       .getReferenceElementType();
    }
    // ByRefResult and InitSelf use ApplyResultSlot.
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
static ASTType getBoundCoroutineType(const TypeCheckScopeInfo &scopeInfo,
                                     const ExprNode *expr, SignatureType sig,
                                     TypedAttr lifetime) {
  auto [declScope, shared] = scopeInfo;
  SMLoc loc = expr->getLoc();
  ASTDecl *decl = sig.isThrows()
                      ? shared.getBuiltinRaisingCoroutineType(declScope, loc)
                      : shared.getBuiltinCoroutineType(declScope, loc);
  if (!decl) {
    shared.emitError(loc,
                     "internal error: could not find builtin 'Coroutine' type");
    return {};
  }
  // If the async function throws, extract the normal result type.
  ASTType resultType = ASTType(sig).getSignatureUserResultType();

  // Bind the result type to the base coroutine type.
  ParamBindings paramBinds(scopeInfo);
  paramBinds.add(expr, PValue(resultType));
  paramBinds.add(expr, lifetime);

  auto structOp = cast<StructDeclOp>(decl);
  ASTType coroType = structOp.bindReference();
  ParameterExprArrayAttr bindings = paramBinds.verifyBindings(
      structOp, cast<AnyStructType>(coroType.getMetaType()).getSignature(),
      expr->getLoc(), /*partial=*/false);
  if (!bindings)
    return {};

  return BindTypeAttr::get(PValue(coroType), bindings);
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
  auto calleeFunc = cast<LIT::FuncOp>(*calleeDecl);

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
struct ExclusivityChecker {
  ExclusivityChecker(const ExprNode *callExpr, CallSyntax syntax,
                     ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                     SharedState &shared)
      : callExpr(callExpr), syntax(syntax), argumentValues(argumentValues),
        shared(shared) {}

  /// As each argument is emitted, check against previous arguments for
  /// exclusivity violations.
  void checkArgument(Value val, ArgConvention convention, size_t argIdx);

private:
  const ExprNode *callExpr;
  CallSyntax syntax;
  /// These are the arguments that are being emitted.
  ArrayRef<ASTExprAnd<AnyValue>> argumentValues;
  SharedState &shared;

  /// For each lifetime that is referenced, we keep track of what argIdx it came
  /// from, and whether it was potentially mutated.
  struct LifetimeInfo {
    unsigned argIdx;
    bool isImmut;
  };
  SmallDenseMap<TypedAttr, LifetimeInfo, 8> lifetimeAccesses;

  void diagViolation(Value val, ArgConvention convention, size_t argIdx,
                     TypedAttr lifetime, const LifetimeInfo &previousAccess);
};
} // end anonymous namespace

/// As each argument is emitted, check against previous arguments for
/// exclusivity violations.
void ExclusivityChecker::checkArgument(Value val, ArgConvention convention,
                                       size_t argIdx) {
  auto checkLifetimeAccess = [&](TypedAttr lifetime) {
    // Determine whether the access was immutable.
    bool isImmut = cast<LifetimeType>(lifetime.getType()).isMutableKnown(false);

    // Look through immcasts to determine the accessed lifetime.
    lifetime = LifetimeMutCastAttr::strip(lifetime);

    // Accesses to the global lifetime never conflict.
    if (isa<LifetimeAttr>(lifetime))
      return;

    // Determine whether we've seen this lifetime before.
    auto [iter, isNew] =
        lifetimeAccesses.insert({lifetime, {unsigned(argIdx), isImmut}});
    if (isNew) // If not, then it isn't a conflict.
      return;

    // If so, check to see if this access and the previous one were both
    // immutable.  Read/read aliasing is fine.
    if (iter->second.isImmut && isImmut)
      return;

    // If not, we have a problem!
    diagViolation(val, convention, argIdx, lifetime, iter->second);
  };

  // If this is a result argument, then we only look at the lifetime of the
  // destination that we're storing into, not any nested references that may
  // be in the result. This returned value is derived from the other arguments
  // passed to the function, it doesn't conflict with them.
  if (convention == ArgConvention::InitSelf ||
      convention == ArgConvention::ByRefResult ||
      convention == ArgConvention::ByRefError) {
    checkLifetimeAccess(cast<RefType>(val.getType()).getLifetime());
    return;
  }

  // Find all the of the lifetimes that are buried in the specified type.
  for (TypedAttr lifetime :
       shared.cachedLifetimeFinder.findLifetimesInTypes(val.getType()))
    checkLifetimeAccess(lifetime);
}

/// Emit an error about an access to a conflicting lifetime after a previous
/// access was seen.
void ExclusivityChecker::diagViolation(Value val, ArgConvention convention,
                                       size_t argIdx, TypedAttr lifetime,
                                       const LifetimeInfo &previousAccess) {
  bool isImmut = cast<LifetimeType>(lifetime.getType()).isMutableKnown(false);
  auto diag = shared.emitWarning(callExpr->getLoc());

  switch (syntax) {
  default:
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

  diag << "argument allows ";
  diag << (isImmut ? "reading" : "writing");
  diag << " a memory location previously ";
  diag << (previousAccess.isImmut ? "readable" : "writable");
  diag << " through another aliased argument";

  // Add ranges for the two arguments.
  diag << argumentValues[argIdx].expr->getRange()
       << argumentValues[previousAccess.argIdx].expr->getRange();

  // Attach a note to explain what is going on in more detail.
  diag.attachNote(callExpr->getLoc());
  lifetime = LifetimeMutCastAttr::strip(lifetime);

  // If the lifetime in question is because of the top-level ref binding, then
  // we have a common problem where something is passed both mutable and
  // borrowed.
  if (SignatureType::hasAddress(convention) &&
      LifetimeMutCastAttr::strip(cast<RefType>(val.getType()).getLifetime()) ==
          lifetime) {
    diag << lifetime << " value is passed through aliasing '"
         << getUserSyntax(convention) << "' argument"
         << argumentValues[argIdx].expr->getRange();
    return;
  }

  ASTType argType = val.getType();
  if (SignatureType::hasAddress(convention))
    argType = argType.getReferenceElementType();

  // Otherwise, it is a more complicated buried lifetime in a type like a
  // Reference or Span.
  diag << lifetime
       << " memory accessed through reference embedded in value of type "
       << argType;
}

/// When emitting a call where any of the arguments are nonmaterializable, we
/// know the type lacks a runtime representation and that it must be
/// interpretable. If all other arguments are PValues, we can safely emit the
/// call in a parameter context.
static bool
shouldEmitParameterCall(LITSignatureType calleeSig,
                        ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                        SharedState &shared) {
  argumentValues = dropResultSlots(argumentValues, calleeSig);
  auto isPValue = [](ASTExprAnd<AnyValue> arg) { return arg.ir.getIfPValue(); };
  auto isNonMaterializable = [&](ASTExprAnd<AnyValue> arg) {
    return arg.ir.getIfPValue().getType().getNonmaterializableTarget(shared);
  };
  return llvm::all_of(argumentValues, isPValue) &&
         (llvm::any_of(argumentValues, isNonMaterializable) ||
          ASTType(calleeSig.getUserResultType())
              .getNonmaterializableTarget(shared));
}

/// Compute the union of all references lifetimes in a set of function call
/// arguments.
static TypedAttr
computeArgumentsLifetime(AsyncCallOp call,
                         CachedTypeLifetimeFinder &lifetimeFinder) {
  SmallVector<std::pair<Value, OperandEffect>> operands;
  SmallVector<ResultEffect> results;
  SmallVector<TypedAttr> lifetimes;
  // Check lifetime accesses on the types. We need to forward this to the
  // coroutine since it is a transitive capture.
  LIT::getOperationEffects(*call, operands, results, lifetimes, lifetimeFinder);
  // Collect the implicit lifetimes of the arguments.
  for (Value value : call.getOperands())
    if (auto ref = dyn_cast<RefType>(value.getType()))
      lifetimes.push_back(ref.getLifetime());
  return LifetimeSetAttr::get(call.getContext(), lifetimes);
}

CValue ExprEmitter::emitCallUnchecked(RValue callee,
                                      const CallOperands &callOperands,
                                      ValueDest &dest, CallSyntax syntax,
                                      const ExprNode *callExpr) {
  CallEmitter callEmitter(callee, callExpr, *this, dest);
  auto calleeSig = cast<LITSignatureType>(callee.getRValueType());

  // We first emit all the arguments.
  FailureOr<SmallVector<ASTExprAnd<AnyValue>>> argumentValuesOr =
      callEmitter.emitArgValues(callOperands);
  if (failed(argumentValuesOr)) {
    dest.resetForError();
    return {};
  }
  ArrayRef<ASTExprAnd<AnyValue>> argumentValues = *argumentValuesOr;

  // Folding into PValue can fail for a number of reasons, in which case we
  // fall back to emitting normally.
  if (FailureOr<CValue> resCValue =
          callEmitter.inlineFunctionCallIntoPValueIfPossible(argumentValues);
      succeeded(resCValue))
    return *resCValue;

  if (!builder || shouldEmitParameterCall(calleeSig, argumentValues, shared)) {
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
  ExclusivityChecker exclusivityChecker(callExpr, syntax, argumentValues,
                                        shared);

  SmallVector<Value> callArgs;
  SmallVector<TypedAttr> implicitLifetimes;
  ArrayRef<ArgConvention> conventions = calleeSig.getArgConventions();
  bool needInitSelfSlot = false;
  for (auto [argIdx, argValAndExpr, conventionX, declaredArgType] :
       llvm::enumerate(argumentValues, conventions, calleeSig.getArguments())) {
    ArgConvention convention = conventionX;

    // If this is a byref_result slot, we will have emitted a null value for
    // this.  We can't know the type of it until we emit all the operands and
    // collect their lifetimes.
    bool isInitSelf = convention == ArgConvention::InitSelf;
    if (SignatureType::isResultSlot(convention) ||
        (isInitSelf && isPlaceholderInitSelf(argValAndExpr.ir))) {
      needInitSelfSlot |= isInitSelf;
      // Don't know the right thing yet, use a placeholder.
      callArgs.push_back(Value());
      implicitLifetimes.push_back(
          LifetimeAttr::get(getContext(), /*isMutable=*/true));
      continue;
    }

    // If this is a variadic operation, the N operands have already been emitted
    // together and consolidated into a pop.variadic.create/pop.variadic.attr,
    // which is emitted as an SRValue instead of whatever the underlying type
    // is.
    if (calleeSig.isPosVarArg(argIdx))
      convention = ArgConvention::BorrowedInReg;

    // Owned and borrowed packs are passed as expected, but inout is passed
    // borrowed.
    if (calleeSig.isPackVarArg(argIdx)) {
      if (convention == ArgConvention::InOut ||
          convention == ArgConvention::BorrowedInMem)
        convention = ArgConvention::BorrowedInReg;
      else if (convention == ArgConvention::OwnedInMem)
        convention = ArgConvention::OwnedInReg;
    }

    Value arg = callEmitter.emitPreemittedArgumentAsDynamicValue(
        argValAndExpr, convention, declaredArgType);
    if (!arg)
      return {};

    // See if we have an implicit lifetime bound for this argument.
    if (SignatureType::hasImplicitLifetime(convention)) {
      implicitLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());
    } else if (calleeSig.isPosVarArg(argIdx)) {
      // If this is a variadic, it will have a wrapper around the ref.
      auto eltType = ASTType(arg.getType()).getVariadicElementType();
      if (auto refType = dyn_cast<RefType>(eltType))
        implicitLifetimes.push_back(refType.getLifetime());
    } else if (ASTType variadicPackType = calleeSig.getIfVariadicPack(argIdx)) {
      // Use the union lifetime that covers all the values.
      RefPackType packType = ASTType(arg.getType()).getVariadicPackInfo();
      implicitLifetimes.push_back(packType.getLifetime());
    }

    // If the address space of a by-ref argument mismatches, then we need to
    // throw an error.  This can happen when non-default address space lvalues
    // are passed to borrowed or inout arguments.
    if (auto refType = dyn_cast<RefType>(arg.getType()))
      if (refType.getAddressSpace() !=
          cast<RefType>(declaredArgType).getAddressSpace()) {
        emitError(argValAndExpr.expr->getLoc(),
                  "value cannot be passed from a non-default address space")
            << argValAndExpr.expr->getRange();
        dest.resetForError();
        return {};
      }

    // The argument looks good on its own, check to see if it is an exclusivity
    // violation with a previous argument.
    exclusivityChecker.checkArgument(arg, convention, argIdx);

    // All looks good!
    callArgs.push_back(arg);
  }

  // Now that we have the lifetimes for the arguments, we can calculate what the
  // substituted signature should be.  This will take in the wrong inout-result
  // lifetime as an input, but we know that it cannot be referenced in the type
  // anyway, because there is no way to name it in the Mojo program.
  FunctionType expectedCalleeType =
      calleeSig.substituteImplicitLifetimesIntoValues(
          implicitLifetimes, [&]() -> InFlightDiagnostic {
            llvm_unreachable("substitution should always succeed");
          });

  // With that done, we can know what type the inout result slot should have.
  // We see if we can emit directly into the ValueDest slot, and if not, we
  // create a VarDecl temporary and allow emitResult to copy it over to the
  // destination. Calls to async functions don't need a result slot provided.
  if ((calleeSig.hasMemoryOnlyResult() || needInitSelfSlot) &&
      !calleeSig.isAsync()) {
    Type refType = needInitSelfSlot ? expectedCalleeType.getInputs().front()
                                    : expectedCalleeType.getInputs().back();
    auto resultRValueType = cast<RefType>(refType).getElementType();

    // Often the result of the call will be directly assigned into a
    // user-defined var or other location with existing storage.  In these
    // cases, we really want to assign directly into the existing slot.
    //
    // However, we cannot do that if the destination slot is also being passed
    // into the call as an input value, as in: `x = foo(x)` or `x = x + 1`.
    // In these cases we really do need a temporary+copy in the var slot.
    // At this point we've got enough information about the arguments to make
    // that assessment in a correct way.
    Value resultSlotVal;
    if (needInitSelfSlot || callEmitter.isSafeToUseValueDestForDirectResult(
                                resultRValueType, callArgs)) {
      // Use the preferred location of the destination slot.
      resultSlotVal = dest.getMLValueForResult(
          callExpr->getLoc(), resultRValueType, callEmitter.emitter);
    } else {
      // Create a temporary.
      resultSlotVal = callEmitter.emitter.emitVarDecl("__call_result_tmp__",
                                                      resultRValueType, loc,
                                                      VarDeclKind::Synthesized);
    }
    // Now that we know the result slot, we can set it and its lifetime.
    if (calleeSig.hasMemoryOnlyResult()) {
      assert(!callArgs.back() && "byref_result slot is always last");
      callArgs.back() = resultSlotVal;
      implicitLifetimes.back() =
          cast<RefType>(resultSlotVal.getType()).getLifetime();
    } else {
      assert(!callArgs.front() && "init_self slot is always first");
      callArgs.front() = resultSlotVal;
      implicitLifetimes.front() =
          cast<RefType>(resultSlotVal.getType()).getLifetime();
    }
  }

  // If the callee throws and is not async, we can now also resolve the lifetime
  // and value of the contextual error slot to provide the callee.
  if (calleeSig.isThrows() && !calleeSig.isAsync()) {
    // The error slot is always the second last argument.
    MLValue errSlot = findNearestErrorSlot();
    if (!errSlot) {
      InflightDiag diag =
          emitError(callExpr->getLoc(), "cannot call function that may raise "
                                        "in a context that cannot raise")
          << callExpr->getRange();
      diag.attachNote(callExpr->getLoc())
          << "try surrounding the call in a 'try' block";
      if (auto func =
              getBlockParentOfType<LIT::FuncOp>(builder->getInsertionBlock())) {
        diag.attachNote(func.getLoc())
            << "or mark surrounding function as 'raises'";
      }
      return {};
    }
    unsigned errSlotOffset = calleeSig.getErrorSlotOffset();
    callArgs[callArgs.size() - errSlotOffset] = errSlot;
    implicitLifetimes[implicitLifetimes.size() - errSlotOffset] =
        cast<RefType>(errSlot.getType()).getLifetime();
  }

  // Now that the implicit lifetimes of result slots are set, recompute the
  // expected types.
  expectedCalleeType = calleeSig.substituteImplicitLifetimesIntoValues(
      implicitLifetimes, [&]() -> InFlightDiagnostic {
        llvm_unreachable("substitution should always succeed");
      });

  // If the function is async, we won't have provided any values for the return
  // slots. Remove them from the call arguments.
  if (calleeSig.isAsync()) {
    callArgs.erase(callArgs.end() - calleeSig.getNumAsyncReturnSlots(),
                   callArgs.end());
  }

  // Now that all of the arguments have been emitted, coerce them to the
  // expected type if needed.  We do this after the first pass above, because
  // there can be forward refefences from the result slot to the later
  // arguments' lifetimes.
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
                                               implicitLifetimes, callArgs);
      ASTType coroType = getBoundCoroutineType(
          getScopeInfo(), callExpr, calleeSig,
          computeArgumentsLifetime(call, shared.cachedLifetimeFinder));
      if (!coroType) {
        dest.resetForError();
        return {};
      }
      // Emit the implicit conversion to Coroutine[T].  We emit into the call's
      // destination to avoid an extra copy/move of the Coroutine object.
      callResult =
          emitConstructorCall(coroType, {{{SBValue(call), callExpr}}}, callExpr,
                              CallSyntax::kImplicitConvert, dest);
      if (!callResult) {
        dest.resetForError();
        return {};
      }
    } else {
      auto call = builder->create<CallOp>(loc, resultType, target.get(),
                                          implicitLifetimes, callArgs);
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
                                                implicitLifetimes, callArgs);
    callResult = SRValue(call.getResult(0));
  }

  // If there were any writebacks to handle, emit them before handling raised
  // errors.
  callEmitter.emitAfterCallActions();

  // If there is a memory result slot, the value we filled in is our MRValue
  // result and we've already handled the ValueDest by emitting into it.
  if (calleeSig.hasMemoryOnlyResult() && !calleeSig.isAsync()) {
    callResult = MRValue(callArgs.back());
  } else if (calleeSig.hasInitSelfArg()) {
    // If this is a constructor call with an implicit `init_self` argument, we
    // need to pass it on to the destination.
    if (needInitSelfSlot)
      callResult = MRValue(callArgs.front());
    else
      // Otherwise, always return `None` when directly invoking a constructor.
      // Raising constructors have an ABI result of `i1`.
      callResult = PValue(shared.getNoneAttr());
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
