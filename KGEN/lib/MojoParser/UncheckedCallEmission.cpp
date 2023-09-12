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

#include "ASTDecl.h"
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ParserParamEvaluator.h"
#include "Utils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/Compiler/OperationUtils.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// CallEmitter (implementation detail)
//===----------------------------------------------------------------------===//

class CallEmitter {
public:
  CallEmitter(CRValue callee, const ExprNode *callExpr, ExprEmitter &emitter,
              ValueDest &dest)
      : callee(callee), callExpr(callExpr), emitter(emitter),
        loc(emitter.translateLocation(callExpr->getLoc())),
        evaluator(emitter.getDeclResolver()), dest(dest),
        calleeSig(cast<SignatureType>(callee.getType().mlirType)),
        afterCallActions(*this){};

  /// Emit IR for a single argument, according to its convention.
  AnyValue emitOneArgVal(ASTExprAnd<AnyValue> operand, unsigned argIdx,
                         ValueInputConvention convention, Type expectedType,
                         size_t sequenceIndex = 0);

  /// Emit all arguments. This function iterates by expected arguments since
  /// we're building the argument list of the call. Default arguments and
  /// variadics are also handled.
  LogicalResult emitArgValues(const CallOperands &operands);

  /// This function emits the specified pre-emitted argument into a single MLIR
  /// Value suitable for passing to the callee with the specified convention.
  /// This handles promotion of PValues to dynamic values as needed.
  Value emitPreemittedArgumentAsDynamicValue(ASTExprAnd<AnyValue> argValAndExpr,
                                             ValueInputConvention convention);

  /// If this is a call to a @always_inline function (and there's only one
  /// possible callee), this method tries to fold the entire function body into
  /// an PValue.
  FailureOr<CValue> inlineFunctionCallIntoPValueIfPossible();

  /// Emit a function call in a parameter context.
  CValue emitCallInParamContext();

  /// Emit any after-call actions collected during call emission.
  void emitAfterCallActions() { afterCallActions.emit(); }

  /// Return the emitted pre-emitted-argument values. Fails if `emitArgValues`
  /// weren't called before.
  ArrayRef<ASTExprAnd<AnyValue>> getEmittedArgValues() {
    assert(argumentValues.has_value());
    return *argumentValues;
  }

private:
  /// The (type-checked and resolved) callee we are emitting the call to.
  CRValue callee;
  /// The call's expression node.
  const ExprNode *callExpr;
  /// The underlying expression emitter instance.
  ExprEmitter &emitter;
  /// The mlir location of the call expression above, stored for convenience.
  Location loc;
  /// The argument values emitted by calling `emitArgValues`.
  std::optional<SmallVector<ASTExprAnd<AnyValue>>> argumentValues;
  /// A parameter evaluator used to simplify parameter expression and fold the
  /// callee if possible.
  ParserParamEvaluator evaluator;
  /// The destination context we're emitting into.
  ValueDest &dest;
  /// The signature type of the callee, stored for convenience.
  SignatureType calleeSig;

  /// This struct accumulates information about IR to emit after the call, e.g.
  /// writebacks for computed inout lvalues, and lifetime markers.
  struct AfterCallActions {
    CallEmitter &callEmitter;

    // The first entry of this is a ValueDest for a DLValue that we can invoke
    // for the setter.
    SmallVector<std::pair<ValueDest, SLValue>> lvalueWritebacks;

    /// This is a list of values that we need to keep alive across the duration
    /// of the call.  They will get lit.ownership.use operations at the end of
    /// the call.
    SmallVector<Value> valuesToKeepAlive;

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
  bool isSafeToUseValueDestForDirectResult(ASTType destRValueType);

  /// Emit the given (remaining) operands as a variadic or pack sequence.
  LogicalResult emitRemainingPosOperands(
      size_t argIdx, MutableArrayRef<ASTExprAnd<AnyValue>> remainingOperands,
      ValueInputConvention convention, Type expectedType);
};

void CallEmitter::AfterCallActions::emit() {
  // Emit the elements and clear the writebacks so the ValueDest's get
  // destroyed when they are emitted into.
  while (!lvalueWritebacks.empty()) {
    auto [dest, lValue] = lvalueWritebacks.pop_back_val();
    if (!callEmitter.emitter.emitResult(MRValue(lValue), callEmitter.callExpr,
                                        dest))
      dest.resetForError();
  }

  // Emit all the lit.ownership.use ops.
  for (Value value : valuesToKeepAlive)
    callEmitter.emitter.builder->create<OwnershipUseOp>(callEmitter.loc, value);
}

AnyValue CallEmitter::emitOneArgVal(ASTExprAnd<AnyValue> operand,
                                    unsigned argIdx,
                                    ValueInputConvention convention,
                                    Type expectedType, size_t sequenceIndex) {
  switch (convention) {
  case ValueInputConvention::ByRef:
  case ValueInputConvention::ByRefResult:
  case ValueInputConvention::InitSelf:
    // By-ref arguments, must be lvalues.
    assert(operand.ir.getIfLValue() && "Call should already be type checked");
    return operand.ir;
  case ValueInputConvention::OwnedInReg:
  case ValueInputConvention::OwnedInMem:
  case ValueInputConvention::BorrowedInReg:
  case ValueInputConvention::BorrowedInMem:
    // by-val arguments are converted to the expected r-value type.
    ASTType expectedArgType = expectedType;
    if (calleeSig.isVararg(argIdx))
      // In the case of a variadic argument, we need to remove the
      // !pop.variadic<> wrapper to get the type to convert to.
      expectedArgType = expectedArgType.getVariadicElementType();
    else if (auto packType = getIfPackType(calleeSig, argIdx))
      // Operands being applied to a concrete pack type argument must be
      // converted to the pack element type at that index.
      expectedArgType = packType.getVariadicAttr().getValues()[sequenceIndex];

    if (convention == ValueInputConvention::OwnedInMem ||
        convention == ValueInputConvention::BorrowedInMem)
      expectedArgType = expectedArgType.getReferenceElementType();

    if (convention == ValueInputConvention::OwnedInReg ||
        convention == ValueInputConvention::OwnedInMem)
      return emitter.emitRValue(operand, EC_CallArgValue, expectedArgType);
    return emitter.emitBValue(operand, EC_CallArgValue, expectedArgType);
  }
  llvm_unreachable("unknown value input convention");
}

LogicalResult CallEmitter::emitRemainingPosOperands(
    size_t argIdx, MutableArrayRef<ASTExprAnd<AnyValue>> remainingOperands,
    ValueInputConvention convention, Type expectedType) {
  // Emit all of the remaining values to make sure they're converted to the
  // right type.
  for (auto [idx, operand] : llvm::enumerate(remainingOperands)) {
    auto emittedArg =
        emitOneArgVal(operand, argIdx, convention, expectedType, idx);
    if (!emittedArg)
      return failure();
    operand.ir = emittedArg;
  }

  // If all of the remaining operands are compile-time values, then we can
  // represent the sequence as a variadic or pack attribute.
  if (std::all_of(remainingOperands.begin(), remainingOperands.end(),
                  [](ASTExprAnd<AnyValue> operand) {
                    return operand.ir.getIfPValue();
                  })) {
    SmallVector<TypedAttr> args;
    for (ASTExprAnd<AnyValue> operand : remainingOperands)
      args.push_back(operand.ir.getIfPValue().get());
    Attribute attr;
    if (calleeSig.isVararg(argIdx))
      attr = VariadicAttr::get(args, expectedType.cast<VariadicType>());
    else
      attr = POP::PackAttr::get(args, expectedType.cast<POP::PackType>());
    argumentValues->push_back({PValue(attr), remainingOperands[0].expr});
    return success();
  }

  // If not all remaining operands are compile-time values, use an operation to
  // create a variadic or pack sequence.
  SmallVector<Value> args;
  for (auto &operand : remainingOperands) {
    Value argVal = emitPreemittedArgumentAsDynamicValue(operand, convention);
    if (!argVal)
      return failure();
    args.push_back(argVal);

    // Make sure the values in the pack stay live across the entire call,
    // not just the pop.variadic.create op.
    bool isTrivial = false;
    if (auto cv = operand.ir.getIfCValue())
      isTrivial =
          cv.getRValueType().isTrivial(callExpr->getLoc(), emitter.shared);
    if (!isTrivial)
      afterCallActions.valuesToKeepAlive.emplace_back(argVal);
  }

  Value argVal;
  if (calleeSig.isVararg(argIdx))
    argVal =
        emitter.builder->create<POP::VariadicCreateOp>(loc, expectedType, args);
  else
    argVal =
        emitter.builder->create<POP::PackCreateOp>(loc, expectedType, args);
  argumentValues->push_back({SRValue(argVal), remainingOperands[0].expr});
  return success();
}

LogicalResult CallEmitter::emitArgValues(const CallOperands &operands) {
  assert(!operands.hasKwOperands() && "keyword arguments not yet supported");
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = operands.posOperands;
  size_t nextOperandIdx = 0;
  size_t nextDefaultIdx = 0;

  assert(!argumentValues.has_value());
  argumentValues = SmallVector<ASTExprAnd<AnyValue>>();
  argumentValues->reserve(calleeSig.getNumInputs());
  for (auto [argIdx, argName, expectedTypeX, convention] :
       llvm::enumerate(calleeSig.getArgNames(), calleeSig.getValueInputs(),
                       calleeSig.getValueInputConventions())) {
    // Use a ParserParamEvaluator to fold only 'apply' expressions. Emit a
    // rebind if the refined type is different than the expected type.
    Type expectedType = evaluator.refineType(expectedTypeX);

    std::optional<OpBuilder> &builder = emitter.builder;

    // If this is the return slot for a call, we want to propagate the
    // ValueDest into this, but we need information about each argument being
    // emitted before we can do that.  As such, we just use a var decl and
    // replace it opportunistically later if we can.
    if (convention == ValueInputConvention::ByRefResult && builder) {
      assert(argIdx == 0 && calleeSig.hasMemoryOnlyResult());
      assert(argName.empty());
      auto resultTmp = builder->create<VarLetDeclOp>(
          loc, expectedType, "__call_result_tmp__", /*isVar=*/true,
          /*isSynth=*/true);
      argumentValues->push_back({SLValue(resultTmp), callExpr});
      continue;
    }

    // Memory-primary result slots are allocated automatically by the apply
    // operator.
    if (!builder && llvm::is_contained({ValueInputConvention::ByRefResult,
                                        ValueInputConvention::InitSelf},
                                       convention))
      continue;

    // If we ran out of operands, fulfill this with a default value, empty
    // variadic list, or empty pack.
    if (nextOperandIdx == posOperands.size()) {
      Attribute argAttr;
      if (calleeSig.isVararg(argIdx)) {
        // Varargs arguments are fulfilled with an empty !kgen.variadic list.
        argAttr = VariadicAttr::get(ArrayRef<TypedAttr>(),
                                    expectedType.cast<VariadicType>());
      } else if (auto packType = getIfPackType(calleeSig, argIdx)) {
        // Pack arguments are fulfilled with an empty !pop.pack sequence.
        assert(packType.isEmpty() &&
               "pack type already checked against operand count");
        argAttr = POP::PackAttr::get(ArrayRef<TypedAttr>(), packType);
      } else {
        // Otherwise, apply the default argument. We've ensured above that we
        // have a default argument for each missing operand.
        argAttr = calleeSig.getDefaultArguments()[nextDefaultIdx++];
      }
      argumentValues->push_back({PValue(argAttr), callExpr});
      continue;
    } else if (argIdx >= calleeSig.getNumInputs() -
                             calleeSig.getDefaultArguments().size()) {
      // If we provided a value for an argument with a default value, advance
      // the index.
      ++nextDefaultIdx;
    }

    // Otherwise, we're applying one or more arguments to this.
    // For a normal (not a vararg or a pack) argument, we just emit it and add
    // it to our list.
    if (!calleeSig.isVararg(argIdx) && !isa<POP::PackType>(expectedType)) {
      auto operand = posOperands[nextOperandIdx++];
      AnyValue argVal =
          emitOneArgVal(operand, argIdx, convention, expectedType);
      if (!argVal)
        return failure();
      argumentValues->push_back({argVal, operand.expr});
      continue;
    }

    // At this point, we must be dealing with variadic or pack arguments. We
    // handle these all at once (or fail).
    SmallVector<ASTExprAnd<AnyValue>> remainingOperands(
        posOperands.begin() + nextOperandIdx, posOperands.end());
    nextOperandIdx = posOperands.size();

    if (succeeded(emitRemainingPosOperands(argIdx, remainingOperands,
                                           convention, expectedType)))
      break;

    return failure();
  }

  assert(nextOperandIdx == posOperands.size() &&
         "typechecking confirmed that we would use up all operands");
  return success();
}

/// Given a call to a function with a memory only result and the desired value
/// destination, decide if it is safe to directly emit into the slot.  Doing so
/// requires a form of alias analysis to determine whether any input arguments
/// could alias the result slot.  We cannot emit into the result slot when
/// passing the value as an argument like 'x = foo(x)' or 'x = x + 1'.
///
/// At this point, we've already applied implicit conversions and converted
/// things to RValues or BValues as required by the argument convention, but
/// things may still be in parameter space.
bool CallEmitter::isSafeToUseValueDestForDirectResult(ASTType destRValueType) {
  // Drop the first argument which is the return slot.
  ArrayRef<ValueInputConvention> argConventions =
      calleeSig.getValueInputConventions();
  assert(argConventions[0] == ValueInputConvention::ByRefResult);
  argConventions = argConventions.drop_front();
  ArrayRef<ASTExprAnd<AnyValue>> argValues = getEmittedArgValues().drop_front();

  // Check to see if the destination provides a buffer.  If not, it is safe to
  // emit into it, but it doesn't actually matter.
  Value destBuffer = dest.getDefinedSLValueIfExists(destRValueType, emitter);
  if (!destBuffer)
    return true;

  // See if the destination buffer is something that ownership can track.  If
  // not, we cannot make reliable determinations about aliasing.
  Value underlyingDest =
      LifetimeTrackable::findUnderlyingValueFromField(destBuffer);
  if (!underlyingDest)
    return false;

  // Check to see if the specified argument value pointer could alias with the
  // destination buffer, returning true if it might.  We can only disambiguate
  // this safely when we can prove that the pointer points to a different
  // distinguishable object than the result slot.
  // TODO: This will need to be extended to support lifetimes.
  auto ptrGuaranteedNoAlias = [&](Value ptrVal) -> bool {
    Value underlyingPtr =
        LifetimeTrackable::findUnderlyingValueFromField(ptrVal);
    return underlyingPtr && underlyingPtr != underlyingDest;
  };

  // If any of the arguments might alias, then we need to use a temporary
  // buffer.
  for (auto [value, convention] : llvm::zip(argValues, argConventions)) {
    switch (convention) {
    case ValueInputConvention::OwnedInReg:
    case ValueInputConvention::BorrowedInReg:
      // Register conventions can never alias the result.
      continue;

    case ValueInputConvention::OwnedInMem:
    case ValueInputConvention::BorrowedInMem:
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::ByRef:
    case ValueInputConvention::InitSelf:
      // Parameter values will never alias.
      if (value.ir.getIfPValue())
        continue;
      if (auto sl = value.ir.getIfSLValue()) {
        if (ptrGuaranteedNoAlias(sl))
          continue;
        return false;
      }
      if (auto mb = value.ir.getIfMBValue()) {
        if (ptrGuaranteedNoAlias(mb))
          continue;
        return false;
      }
      if (auto mb = value.ir.getIfMRValue()) {
        if (ptrGuaranteedNoAlias(mb))
          continue;
        return false;
      }
      // Dynamic variadic memory values are passed with a pop.variadic.create,
      // check each field.
      if (auto sr = value.ir.getIfSRValue()) {
        if (auto variadic = sr.getDefiningOp<POP::VariadicCreateOp>()) {
          for (auto operand : variadic.getOperands()) {
            if (!ptrGuaranteedNoAlias(operand))
              return false;
          }
          continue;
        }
      }
      llvm_unreachable("Unknown value kind for memory convention");
    }
  }

  // If no problems are found, it is safe!
  return true;
}

Value CallEmitter::emitPreemittedArgumentAsDynamicValue(
    ASTExprAnd<AnyValue> argValAndExpr, ValueInputConvention convention) {
  Value arg;
  switch (convention) {
  case ValueInputConvention::OwnedInReg:
    // Promote PValue's if needed.
    return emitter.emitSRValue(argValAndExpr, EC_CallArgValue);
  case ValueInputConvention::OwnedInMem:
    // Promote PValue's if needed.
    return emitter.emitMRValue(argValAndExpr, EC_CallArgValue);
  case ValueInputConvention::BorrowedInReg:
    if (auto pVal = argValAndExpr.ir.getIfPValue())
      return arg = emitter.emitSRValue(argValAndExpr, EC_CallArgValue);

    // If this is an MBValue, the element must be register passable but not
    // loaded.
    if (auto mbVal = argValAndExpr.ir.getIfMBValue()) {
      const ExprNode *expr = argValAndExpr.expr;
      // TODO: Factor this into a helper.
      std::optional<OpBuilder> builder = emitter.builder;
      if (!builder) {
        emitter.emitErrorForDynamicValueInParameter(expr);
        return {};
      }
      auto load =
          builder->create<POP::LoadOp>(expr->getLocation(emitter), mbVal,
                                       /*alignment=*/std::nullopt);
      argValAndExpr.ir = SBValue(load);
    }

    arg = argValAndExpr.ir.getIfSBValue();
    break;
  case ValueInputConvention::BorrowedInMem:
    // Promote PValue's if needed.
    return emitter.emitMBValue(argValAndExpr, EC_CallArgValue);
  case ValueInputConvention::ByRefResult: {
    auto tmpSlotAddr = argValAndExpr.ir.getIfSLValue();
    assert(tmpSlotAddr && "byref_result value start in a temp slot");
    auto rvalueType = ASTType(tmpSlotAddr.getType()).getReferenceElementType();

    // Often the result of the call will be directly assigned into a
    // user-defined var or other location with existing storage.  In these
    // cases, we really want to assign directly into the existing slot.
    //
    // However, we cannot do that if the destination slot is also being passed
    // into the call as an input value, as in: `x = foo(x)` or `x = x + 1`.
    // In these cases we really do need a temporary+copy in the var slot.
    // At this point we've got enough information about the arguments to make
    // that assessment in a correct way.
    if (!isSafeToUseValueDestForDirectResult(rvalueType))
      return tmpSlotAddr;

    // Okay it is safe to use, so remove the temporary allocation we aren't
    // going to use.
    tmpSlotAddr.getDefiningOp<VarLetDeclOp>()->erase();
    // Get the SLValue of the destination slot.
    return dest.getSLValueForResult(callExpr->getLoc(), rvalueType, emitter);
  }
  case ValueInputConvention::ByRef:
  case ValueInputConvention::InitSelf: {
    // We know that the operand is an LValue, but it might be
    // dynamic/computed.
    LValue lv = argValAndExpr.ir.getIfLValue();
    assert(lv && "type checking ensures we will have an lvalue");
    if (auto sl = lv.getIfSLValue())
      return sl;

    // If dynamic, we need to generate a temporary slot, emit a 'get' into
    // that slot, pass the address, then write it back when we're done.
    ValueDest dlvBuffer(lv, EC_CallArgValue);
    SLValue slvBuffer = dlvBuffer.getSLValueForResult(
        argValAndExpr.expr->getLoc(), lv.getRValueType(), emitter);
    // Emit the 'get' into the buffer.
    ValueDest bufferDest(slvBuffer, EC_CallArgValue);
    if (!emitter.emitLoadOfLValue({lv, argValAndExpr.expr}, bufferDest)) {
      bufferDest.resetForError();
      dlvBuffer.resetForError();
      return {};
    }
    afterCallActions.lvalueWritebacks.push_back(
        {std::move(dlvBuffer), slvBuffer});
    return slvBuffer;
  }
  }
  if (!arg) {
    llvm::errs() << "CALL ARG MISMATCH: " << int(convention) << " ";
    argValAndExpr.ir.dump();
    llvm_unreachable("didn't get a value as expected");
  }
  return arg;
}

FailureOr<CValue> CallEmitter::inlineFunctionCallIntoPValueIfPossible() {
  if (calleeSig.isThrows())
    return failure();
  auto calleePR = callee.getIfPValue();
  if (!calleePR)
    return failure();
  auto calleeSymbolCst = dyn_cast<SymbolConstantAttr>(calleePR.get());
  if (!calleeSymbolCst)
    return failure();

  SmallVector<Attribute> arguments;
  for (ASTExprAnd<AnyValue> argValue : getEmittedArgValues()) {
    auto mValue = argValue.ir.getIfPValue();
    if (!mValue || !ParameterAttr::isSimpleConstant(mValue.get()))
      return failure();
    arguments.push_back(mValue.get());
  }

  auto res =
      evaluator.evaluateFunctionCall(calleeSymbolCst.getSymbol(), arguments);
  if (failed(res))
    return failure();
  return emitter.emitCResult(*res, callExpr, dest);
}

CValue CallEmitter::emitCallInParamContext() {
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
  bool dropFirst =
      calleeSig.hasMemoryOnlyResult() || calleeSig.hasInitSelfResult();
  for (auto [argValAndExpr, calleeArgType, convention] :
       llvm::zip(getEmittedArgValues(),
                 calleeSig.getValueInputs().drop_front(dropFirst),
                 calleeSig.getValueInputConventions().drop_front(dropFirst))) {
    PValue pValue = argValAndExpr.ir.getIfPValue();
    if (!pValue) {
      return emitter.emitErrorForDynamicValueInParameter(
          argValAndExpr.expr,
          "cannot use a dynamic value in parameter context");
    }
    TypedAttr arg = pValue.get();
    // Put memory-only arguments into memory ("PRValue" to "PLValue"
    // conversion).
    if (!llvm::is_contained({ValueInputConvention::BorrowedInReg,
                             ValueInputConvention::OwnedInReg},
                            convention) &&
        !isa<StoreToMemAttr>(arg))
      arg = StoreToMemAttr::get(arg, PointerType::get(arg.getType()));
    // Emit a rebind if the refined type does not match the callee arg type.
    if (arg.getType() != calleeArgType)
      arg = ParamOperatorAttr::get(POC::Rebind, arg, calleeArgType);
    operands.push_back(arg);
  }

  bool hasResultSlot =
      calleeSig.hasMemoryOnlyResult() || calleeSig.hasInitSelfResult();
  Type resultType =
      hasResultSlot
          ? ASTType(calleeSig.getValueInputs().front()).getPointerElementType()
          : ASTType(calleeSig.getValueResults().front());
  TypedAttr result = ParamOperatorAttr::get(
      hasResultSlot ? POC::ApplyResultSlot : POC::Apply, operands, resultType);
  return emitter.emitCResult(result, callExpr, dest);
}

//===----------------------------------------------------------------------===//
// ExprEmitter::emitCallUnchecked
//===----------------------------------------------------------------------===//

CValue ExprEmitter::emitCallUnchecked(CRValue callee,
                                      const CallOperands &callOperands,
                                      ArrayRef<ParamDeclAttr> resultParams,
                                      ValueDest &dest,
                                      const ExprNode *callExpr) {
  CallEmitter callEmitter(callee, callExpr, *this, dest);

  auto calleeSig = cast<SignatureType>(callee.getType().mlirType);
  assert(calleeSig.getNumResultParams() == resultParams.size() &&
         "Type checking should be done");

  // We first emit all the arguments.
  if (failed(callEmitter.emitArgValues(callOperands)))
    return {};

  // Folding into PValue can fail for a number of reasons, in which case we
  // fall back to emitting normally.
  if (FailureOr<CValue> resCValue =
          callEmitter.inlineFunctionCallIntoPValueIfPossible();
      succeeded(resCValue))
    return *resCValue;

  // If we are in a parameter context, we can now emit the call.
  if (!builder)
    return callEmitter.emitCallInParamContext();

  Location loc = translateLocation(callExpr->getLoc());

  // Otherwise, materialize PValue and DLValue's as SSA values for emission.
  SmallVector<Value> callArgs;
  SmallVector<Value, 1> byRefResults;
  for (auto [argValAndExpr, conventionX, calleeArgTypeAndIdx] :
       llvm::zip(callEmitter.getEmittedArgValues(),
                 calleeSig.getValueInputConventions(),
                 llvm::enumerate(calleeSig.getValueInputs()))) {
    auto calleeArgType = calleeArgTypeAndIdx.value();
    auto argIdx = calleeArgTypeAndIdx.index();
    ValueInputConvention convention = conventionX;

    // If this is a variadic operation, the N operands have already been emitted
    // together and consolidated into a pop.variadic.create/pop.variadic.attr,
    // which is emitted as an SRValue instead of whatever the underlying type
    // is.
    if (calleeSig.isVararg(argIdx) || isa<POP::PackType>(calleeArgType))
      convention = ValueInputConvention::OwnedInReg;

    Value arg = callEmitter.emitPreemittedArgumentAsDynamicValue(argValAndExpr,
                                                                 convention);
    if (!arg)
      return {};
    if (arg.getType() != calleeArgType)
      arg = builder->create<RebindOp>(loc, calleeArgType, arg);
    if (conventionX == ValueInputConvention::ByRefResult ||
        conventionX == ValueInputConvention::InitSelf)
      byRefResults.push_back(arg);
    callArgs.push_back(arg);
  }

  ArrayRef<Type> resultTypes = calleeSig.getValueResults();
  Value callResult;
  if (auto target = callee.getIfPValue()) {
    if (auto sig = dyn_cast<SignatureType>(target.getType().mlirType);
        sig && sig.isAsync()) {
      // If the callee is an async function, emit an async call. Then wrap the
      // `!pop.coroutine<() -> T>` result in a `Coroutine[T]` object.
      auto call = builder->create<AsyncCallOp>(loc, target.get(), resultParams,
                                               callArgs);
      ASTType coroType =
          shared.getBuiltinCoroutineType(declScope, callExpr->getLoc());
      if (!coroType) {
        emitError(callExpr->getLoc(),
                  "internal error: could not find builtin 'Coroutine' type");
        return {};
      }
      // Bind the result type to the base coroutine type.
      coroType = DeclRefType::get(
          cast<DeclRefType>(coroType.mlirType).getSymbol(),
          ParamBindArrayAttr::get(
              getContext(),
              {ParamBindAttr::get(
                  "type", TypeConstantAttr::get(resultTypes.front()))}));
      ValueDest dest;
      // Emit the implicit conversion.
      callResult =
          emitConstructorCall(coroType, {{{SBValue(call), callExpr}}}, callExpr,
                              CallSyntax::kImplicitConvert, dest)
              .getIfSRValue();
    } else if (auto symbol = dyn_cast<SymbolConstantAttr>(target.get())) {
      // If the callee is a symbol constant, directly emit a call.
      auto call = builder->create<CallOp>(loc, resultTypes, symbol,
                                          resultParams, callArgs);
      callResult = call.getResult(0);
    } else {
      auto call = builder->create<CallParamOp>(loc, resultTypes, target.get(),
                                               resultParams, callArgs);
      callResult = call.getResult(0);
    }
  } else {
    auto call = builder->create<CallSignatureOp>(
        loc, resultTypes, callee.getIfSRValue(), callArgs);
    callResult = call.getResult(0);
  }

  // If there were any writebacks to handle, emit them before handling raised
  // errors.
  callEmitter.emitAfterCallActions();

  // If the callee can raise an error, it will be represented as a variant: try
  // to unwrap it.
  if (calleeSig.isThrows()) {
    // Put the insertion point back after we're done building the 'if'.
    OpBuilder::InsertionGuard builderGuard(*builder);
    auto callResultTy = cast<POP::VariantType>(callResult.getType());
    Type successType = callResultTy.getType(1);
    auto handleVariant = builder->create<LIT::HandleVariantOp>(
        loc, successType, callResult, ValueRange(byRefResults));
    Block *successBlock =
        builder->createBlock(&handleVariant.getSuccessRegion());
    builder->setInsertionPointToStart(successBlock);
    Value value =
        builder->create<POP::VariantGetOp>(loc, successType, callResult);
    builder->create<LIT::YieldOp>(loc, value);

    Block *errorBlock = builder->createBlock(&handleVariant.getErrorRegion());
    builder->setInsertionPointToStart(errorBlock);
    Value error = builder->create<POP::VariantGetOp>(
        loc, callResultTy.getType(0), callResult);
    if (failed(emitRaise(error, loc))) {
      InflightDiag diag =
          emitError(callExpr->getLoc(), "cannot call function that may raise "
                                        "in a context that cannot raise")
          << callExpr->getRange();
      diag.attachNote(callExpr->getLoc())
          << "try surrounding the call in a 'try' block";
      if (auto func =
              getBlockParentOfType<LIT::FuncOp>(builder->getInsertionBlock()))
        diag.attachNote(func.getLoc())
            << "or mark surrounding function as 'raises'";
      return {};
    }
    builder->create<UnreachableOp>(loc);
    callResult = handleVariant.getResult(0);
  }

  // If there is a memory result slot, the value we filled in is our MRValue
  // result and we've already handled the ValueDest by emitting into it.
  if (calleeSig.hasMemoryOnlyResult()) {
    // Re-emit the value in case a conversion was required or if the result was
    // a dynamic-lvalue.  In both case we will have emitted into a temporary
    // slot and 'dest' will have the ultimate location to write to.
    return emitCResult(MRValue(callArgs[0]), callExpr, dest);
  }

  // Otherwise, register-passable results are the call result which may need to
  // be emitted into a ValueDest.
  return emitCResult(SRValue(callResult), callExpr, dest);
}
