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
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "MojoUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"

#include "llvm/Support/SaveAndRestore.h"

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
      : emitter(emitter), callee(callee), callExpr(callExpr),
        loc(emitter.translateLocation(callExpr->getLoc())),
        evaluator(emitter.getDeclResolver()), dest(dest),
        calleeSig(cast<SignatureType>(callee.getType().mlirType)),
        afterCallActions(*this){};

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
  /// This handles promotion of PValues to dynamic values as needed. It needs
  /// the list of pre-emitted argument values to check aliasing with the result
  /// slot.
  Value emitPreemittedArgumentAsDynamicValue(
      ASTExprAnd<AnyValue> argValAndExpr, ArgConvention convention,
      ArrayRef<ASTExprAnd<AnyValue>> argumentValues);

  /// If this is a call to a @always_inline function (and there's only one
  /// possible callee), this method tries to fold the entire function body into
  /// an PValue.
  FailureOr<CValue> inlineFunctionCallIntoPValueIfPossible(
      ArrayRef<ASTExprAnd<AnyValue>> argumentValues);

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
  bool isSafeToUseValueDestForDirectResult(
      ASTType destRValueType, ArrayRef<ASTExprAnd<AnyValue>> argumentValues);

private:
  /// The (type-checked and resolved) callee we are emitting the call to.
  CRValue callee;
  /// The call's expression node.
  const ExprNode *callExpr;
  /// The mlir location of the call expression above, stored for convenience.
  Location loc;
  /// A parameter evaluator used to simplify parameter expression and fold the
  /// callee if possible.
  ParserParamEvaluator evaluator;
  /// The destination context we're emitting into.
  ValueDest &dest;
  /// The signature type of the callee, stored for convenience.
  LITSignatureType calleeSig;

  /// This struct accumulates information about IR to emit after the call, e.g.
  /// writebacks for computed inout lvalues, and lifetime markers.
  struct AfterCallActions {
    CallEmitter &callEmitter;

    // The first entry of this is a ValueDest for a DLValue that we can invoke
    // for the setter.
    SmallVector<std::pair<ValueDest, MLValue>> lvalueWritebacks;

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

  /// Emit the given (remaining) operands as a variadic or pack sequence,
  /// appending to the given argument value vector.
  LogicalResult emitRemainingPosOperands(
      size_t argIdx, MutableArrayRef<ASTExprAnd<AnyValue>> remainingOperands,
      ArgConvention convention, Type expectedType,
      SmallVectorImpl<ASTExprAnd<AnyValue>> &argumentValues);
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
                                    unsigned argIdx, ArgConvention convention,
                                    Type expectedType, size_t sequenceIndex) {
  if (calleeSig.isVarArg(argIdx)) {
    auto variadic = cast<VariadicType>(expectedType);
    // In the case of a variadic argument, we need to remove the
    // !pop.variadic<> wrapper to get the type to convert to.
    expectedType = variadic.getElementType();
    convention = variadic.getConvention();
  } else if (auto packType = getIfPackType(calleeSig, argIdx)) {
    // Operands being applied to a concrete pack type argument must be
    // converted to the pack element type at that index.
    expectedType =
        ASTType(packType.getVariadicAttr().getValues()[sequenceIndex]);
  }

  switch (convention) {
  case ArgConvention::ByRef:
  case ArgConvention::ByRefResult:
  case ArgConvention::InitSelf:
    // By-ref arguments, must be lvalues.
    assert(operand.ir.getIfLValue() && "Call should already be type checked");
    return operand.ir;
  case ArgConvention::OwnedInReg:
  case ArgConvention::OwnedInMem:
  case ArgConvention::BorrowedInReg:
  case ArgConvention::BorrowedInMem: {
    // by-val arguments are converted to the expected r-value type.
    if (SignatureType::hasAddress(convention))
      expectedType = cast<RefType>(expectedType).getElementType();

    if (convention == ArgConvention::OwnedInReg ||
        convention == ArgConvention::OwnedInMem)
      return emitter.emitRValue(operand, EC_CallArgValue, expectedType);

    // If this is a low-level !lit.ref passed by value, we can bind either to a
    // !lit.ref SBValue of the same type, or an MValue of the element type.
    // TODO(references): If we had lifetimeof(self) and inout/borrowed
    // overloading, we could get rid of this special case.
    if (auto expectedRef = dyn_cast<RefType>(expectedType)) {
      if (auto cValue = operand.ir.getIfCValue())
        if (convention == ArgConvention::BorrowedInReg &&
            ASTType(cValue.getRValueType())
                .isEqualCanon(expectedRef.getElementType())) {
          // We don't currently support non-MValues.  We could dump them into
          // memory with an MBValue conversion if there is a need to.
          if (!cValue.isMValue()) {
            emitter.emitError(
                operand.expr->getLoc(),
                "cannot pass non-memory value through implicit reference");
            return {};
          }
          // Lifetimes must be convertible, this is checked by OverloadFitness.
          Value refValue = cValue.getMValueReference();
          auto refValueType = cast<RefType>(refValue.getType());
          // The destination may be less mutable.
          if (!refValueType.isMutableKnown(false) &&
              expectedRef.isMutableKnown(false))
            refValue = emitter.builder->create<RefImmutOp>(
                operand.expr->getLocation(emitter), refValue);
          assert(refValue.getType() == expectedType &&
                 "Should have exact match now");
          return SBValue(refValue);
        }
    }

    return emitter.emitBValue(operand, EC_CallArgValue, expectedType);
  }
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
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
  bool isVarArg = calleeSig.isVarArg(argIdx);
  if (isVarArg)
    convention = cast<VariadicType>(expectedType).getConvention();

  // If we are emitting in a compile-time context and all of the remaining
  // operands are compile-time values, then emit the sequence as an attribute.
  if (!emitter.builder &&
      std::all_of(remainingOperands.begin(), remainingOperands.end(),
                  [](ASTExprAnd<AnyValue> operand) {
                    return operand.ir.getIfPValue();
                  })) {
    SmallVector<TypedAttr> args;
    for (ASTExprAnd<AnyValue> operand : remainingOperands)
      args.push_back(operand.ir.getIfPValue().get());
    Attribute attr;
    if (isVarArg) {
      auto varType = cast<VariadicType>(expectedType);
      Type varElType = varType.getElementType();

      // If dealing with a memory-only type, remove the pointer.
      if (SignatureType::hasAddress(convention))
        varElType = ASTType(varElType).getReferenceElementType();
      auto newVarType = VariadicType::get(varElType, varType.getConvention());
      attr = VariadicAttr::get(args, newVarType);
    } else {
      attr = PackAttr::get(args, cast<PackType>(expectedType));
    }
    argumentValues.push_back({PValue(attr), remainingOperands[0].expr});
    return success();
  }

  // If not all remaining operands are compile-time values, use an operation to
  // create a variadic or pack sequence.
  SmallVector<Value> args;
  for (auto &operand : remainingOperands) {
    Value argVal = emitPreemittedArgumentAsDynamicValue(operand, convention,
                                                        argumentValues);
    if (!argVal)
      return failure();
    args.push_back(argVal);

    // Make sure any register values in the pack stay live across the entire
    // call, not just the pop.variadic.create op.  Memory values are kept alive
    // by their lifetime.
    bool isTrivial = false;
    if (auto cv = operand.ir.getIfCValue())
      isTrivial =
          cv.getRValueType().isTrivial(callExpr->getLoc(), emitter.shared);

    // Memory values are kept alive by their lifetime so they don't need these
    // ops.
    // TODO: Expand this to packs as well.
    if (isVarArg && SignatureType::hasAddress(convention))
      isTrivial = true;

    if (!isTrivial)
      afterCallActions.valuesToKeepAlive.emplace_back(argVal);
  }

  // If there are lifetimes on anything, create a uniform representation and
  // cast to a common reference type.
  if (isVarArg && !args.empty() && isa<RefType>(args.back().getType())) {
    auto expectedVararg = cast<VariadicType>(expectedType);
    // If one arg is a reference, then they all are.
    auto expectedRefType = cast<RefType>(expectedVararg.getElementType());
    SmallVector<TypedAttr> refLifetimes;
    for (auto arg : args)
      refLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());

    // If there is more than one element, they probably have different
    // lifetimes, and thus need to be rebound into a common union of them.
    LifetimeType commonLifetimeType = expectedRefType.getLifetimeType();
    auto commonLifetime =
        LifetimeUnionAttr::get(refLifetimes, commonLifetimeType);
    expectedRefType = expectedRefType.getWithLifetime(commonLifetime);
    for (auto &arg : args)
      // Cast to common lifetime.
      if (arg.getType() != expectedRefType)
        arg = emitter.builder->create<RebindOp>(loc, expectedRefType, arg);

    // TODO(metatypes): preserve metatype?  These don't seem to do anything.
    expectedType =
        VariadicType::get(expectedRefType, expectedVararg.getConvention());
  }

  Value argVal;
  if (isVarArg) {
    // Check for a splat.
    if (!args.empty() &&
        llvm::all_of(args, [&](Value operand) { return operand == args[0]; })) {
      argVal = emitter.builder->create<POP::VariadicSplatOp>(
          loc, expectedType, args[0], args.size());
    } else {
      argVal = emitter.builder->create<POP::VariadicCreateOp>(loc, expectedType,
                                                              args);
    }
  } else
    argVal = emitter.builder->create<PackCreateOp>(loc, expectedType, args);
  argumentValues.push_back({SRValue(argVal), remainingOperands[0].expr});
  return success();
}

FailureOr<SmallVector<ASTExprAnd<AnyValue>>>
CallEmitter::emitArgValues(const CallOperands &operands) {
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = operands.posOperands;
  size_t posOperandIdx = 0;

  SmallVector<ASTExprAnd<AnyValue>> argumentValues;
  argumentValues.reserve(calleeSig.getNumArguments());

  auto defaultHandler = DefaultValueHandler::getDefaultArgHandler(calleeSig);
  for (auto [argIdx, argName, expectedTypeX, convention, passingKind] :
       llvm::enumerate(calleeSig.getArgNames(), calleeSig.getArguments(),
                       calleeSig.getArgConventions(),
                       calleeSig.getArgPassingKinds())) {
    // Use a ParserParamEvaluator to fold only 'apply' expressions. Emit a
    // rebind if the refined type is different than the expected type.
    Type expectedType = evaluator.refineType(expectedTypeX);

    std::optional<OpBuilder> &builder = emitter.builder;

    // Memory-only result slots are allocated automatically by the apply
    // operator.
    if (!builder && (convention == ArgConvention::ByRefResult ||
                     convention == ArgConvention::InitSelf))
      continue;

    // If this is the return slot for a call, we need a temporary to emit into,
    // but don't know the type until the arguments (and their lifetimes) are all
    // emitted. Just skip over it for now.
    if (convention == ArgConvention::ByRefResult && builder) {
      assert(argIdx == 0 && calleeSig.hasMemoryOnlyResult());
      assert(passingKind == PassingKind::PosOnly);
      argumentValues.push_back({AnyValue(), callExpr});
      continue;
    }

    // If we ran out of operands, fulfill this with a keyword argument, default
    // value, empty variadic list, or empty pack.
    if (posOperandIdx == posOperands.size()) {
      if (calleeSig.isVarArg(argIdx)) {
        // VarArgs arguments are fulfilled with an empty !kgen.variadic list.
        auto argAttr = VariadicAttr::get(ArrayRef<TypedAttr>(),
                                         expectedType.cast<VariadicType>());
        argumentValues.push_back({PValue(argAttr), callExpr});
        continue;
      }
      if (auto packType = getIfPackType(calleeSig, argIdx)) {
        // Pack arguments are fulfilled with an empty !kgen.pack sequence.
        assert(packType.isEmpty() &&
               "pack type already checked against operand count");
        auto argAttr = PackAttr::get(ArrayRef<TypedAttr>(), packType);
        argumentValues.push_back({PValue(argAttr), callExpr});
        continue;
      }
      if (auto kwOperandOr = operands.findKwArg(argName);
          kwOperandOr.has_value()) {
        // The argument is passed as a keyword operand.
        AnyValue argVal =
            emitOneArgVal(*kwOperandOr, argIdx, convention, expectedType);
        if (!argVal)
          return failure();
        argumentValues.push_back({argVal, kwOperandOr->expr});
        continue;
      }

      // Otherwise, apply the default argument. We've ensured before that we
      // have a default argument for each missing operand.
      TypedAttr defaultOr = defaultHandler.getDefault(argIdx);
      assert(defaultOr);
      assert(convention != ArgConvention::ByRef &&
             "by_ref argument cannot have defaults");
      argumentValues.push_back({PValue(defaultOr), callExpr});
      continue;
    }

    // Otherwise, we're applying one or more arguments to this.
    // For a normal (not a vararg or a pack) argument, we just emit it and add
    // it to our list.
    if (!calleeSig.isVarArg(argIdx) && !calleeSig.isPackVarArg(argIdx)) {
      ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx++];
      AnyValue argVal =
          emitOneArgVal(operand, argIdx, convention, expectedType);
      if (!argVal)
        return failure();
      argumentValues.push_back({argVal, operand.expr});
      continue;
    }

    // At this point, we must be dealing with variadic or pack arguments. We
    // handle these all at once (or fail).
    SmallVector<ASTExprAnd<AnyValue>> remainingOperands(
        posOperands.begin() + posOperandIdx, posOperands.end());
    posOperandIdx = posOperands.size();

    if (succeeded(emitRemainingPosOperands(argIdx, remainingOperands,
                                           convention, expectedType,
                                           argumentValues)))
      break;

    return failure();
  }

  assert(posOperandIdx == posOperands.size() &&
         "typechecking confirmed that we would use up all operands");
  return argumentValues;
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
bool CallEmitter::isSafeToUseValueDestForDirectResult(
    ASTType destRValueType, ArrayRef<ASTExprAnd<AnyValue>> argumentValues) {
  // Drop the first argument which is the return slot.
  ArrayRef<ArgConvention> argConventions = calleeSig.getArgConventions();
  assert(argConventions[0] == ArgConvention::ByRefResult);
  argConventions = argConventions.drop_front();
  argumentValues = argumentValues.drop_front();

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
  for (auto [value, convention] : llvm::zip(argumentValues, argConventions)) {
    if (SignatureType::hasAddress(convention)) {
      // Parameter values will never alias.
      if (value.ir.getIfPValue())
        continue;
      if (value.ir.isMValue()) {
        if (ptrGuaranteedNoAlias(value.ir.getMValueReference()))
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
        if (auto variadic = sr.getDefiningOp<POP::VariadicSplatOp>()) {
          if (!ptrGuaranteedNoAlias(variadic.getOperand()))
            return false;
          continue;
        }
      }

      // Otherwise, this may be a scalar value being passed through a borrowed
      // convention (e.g. for trait-bound value).  These will get anonymous
      // memory locations so they'll never alias.
      if (value.ir.isSValue() || value.ir.getIfPValue())
        continue;

      llvm_unreachable("Unknown value kind for memory convention");
    }
  }

  // If no problems are found, it is safe!
  return true;
}

Value CallEmitter::emitPreemittedArgumentAsDynamicValue(
    ASTExprAnd<AnyValue> argValAndExpr, ArgConvention convention,
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues) {

  Value arg;
  switch (convention) {
  case ArgConvention::ByRefResult:
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
    if (auto refVal = argValAndExpr.ir.getIfMBValue()) {
      auto load = emitter.builder->create<RefLoadOp>(
          argValAndExpr.expr->getLocation(emitter), refVal);
      argValAndExpr.ir = SBValue(load);
    }

    arg = argValAndExpr.ir.getIfSBValue();
    assert(arg && "unknown BValue");
    break;
  case ArgConvention::BorrowedInMem: {
    if (SBValue sbValue = argValAndExpr.ir.getIfSBValue()) {
      const ExprNode *expr = argValAndExpr.expr;
      Location argLoc = expr->getLocation(emitter);
      Value ptr = emitter.builder->create<POP::StackAllocationOp>(
          argLoc, PointerType::get(sbValue.getType()), 1);
      emitter.builder->create<POP::StoreOp>(argLoc, sbValue, ptr);
      // Given a legacy pointer, get it to a reference.
      // TODO(references): RefFromPointerOp should take a novel lifetime.
      auto immortal = emitter.builder->getAttr<LifetimeAttr>(/*isMut=*/false);
      auto ref =
          emitter.builder->create<RefFromPointerOp>(argLoc, ptr, immortal,
                                                    /*startUninit=*/false,
                                                    /*endUninit=*/false);
      // Because the result of StackAllocationOp is not a lifetime trackable,
      // StoreOp will not transfer ownership and we must manually extend the
      // lifetime of the SBValue.
      afterCallActions.valuesToKeepAlive.push_back(sbValue);
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
  case ArgConvention::ByRef:
  case ArgConvention::InitSelf: {
    // We know that the operand is an LValue, but it might be
    // dynamic/computed.
    LValue lv = argValAndExpr.ir.getIfLValue();
    assert(lv && "type checking ensures we will have an lvalue");

    if (auto ref = lv.getIfMLValue())
      return ref;

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
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }
  if (!arg) {
    llvm::errs() << "CALL ARG MISMATCH: " << int(convention) << " ";
    argValAndExpr.ir.dump();
    llvm_unreachable("didn't get a value as expected");
  }
  return arg;
}

FailureOr<CValue> CallEmitter::inlineFunctionCallIntoPValueIfPossible(
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues) {
  if (calleeSig.isThrows())
    return failure();
  auto calleePR = callee.getIfPValue();
  if (!calleePR)
    return failure();
  auto calleeSymbolCst = dyn_cast<SymbolConstantAttr>(calleePR.get());
  if (!calleeSymbolCst)
    return failure();

  SmallVector<Attribute> arguments;
  for (ASTExprAnd<AnyValue> argValue : argumentValues) {
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

  bool dropFirst =
      boundSigType.hasMemoryOnlyResult() || boundSigType.hasInitSelfResult();
  for (auto [argIdx, argValAndExpr, calleeArgType, convention] :
       llvm::enumerate(
           argumentValues, boundSigType.getArguments().drop_front(dropFirst),
           boundSigType.getArgConventions().drop_front(dropFirst))) {
    PValue pValue = argValAndExpr.ir.getIfPValue();
    if (!pValue)
      return emitter.emitErrorForDynamicValueInParameter(argValAndExpr.expr);
    TypedAttr arg = pValue.get();
    // Put memory-only arguments into memory ("PRValue" to "PLValue"
    // conversion).
    if (SignatureType::hasAddress(convention)) {
      arg = StoreToMemAttr::get(
          arg, RefType::getImmortal(arg.getType(), /*isMut=*/true));
    } else if (boundSigType.isVarArg(argIdx)) {
      // If handling a variadic memory argument, put each element into memory.
      auto varType = cast<VariadicType>(arg.getType());
      if (SignatureType::hasAddress(varType.getConvention())) {
        bool isMut = varType.getConvention() != ArgConvention::BorrowedInMem;
        auto varElType = RefType::getImmortal(varType.getElementType(), isMut);
        SmallVector<TypedAttr> storedAttrs;
        for (TypedAttr var : cast<VariadicAttr>(arg).getValues())
          storedAttrs.push_back(StoreToMemAttr::get(var, varElType));
        auto newVarType = VariadicType::get(varElType, varType.getConvention());
        arg = VariadicAttr::get(storedAttrs, newVarType);
      }
    }
    // Emit a rebind if the refined type does not match the callee arg type.
    if (arg.getType() != calleeArgType)
      arg = ParamOperatorAttr::get(POC::Rebind, arg, calleeArgType);
    operands.push_back(arg);
  }

  bool hasResultSlot =
      boundSigType.hasMemoryOnlyResult() || boundSigType.hasInitSelfResult();
  Type resultType = hasResultSlot ? ASTType(boundSigType.getArguments().front())
                                        .getReferenceElementType()
                                  : ASTType(boundSigType.getResults().front());
  TypedAttr result = ParamOperatorAttr::get(
      hasResultSlot ? POC::ApplyResultSlot : POC::Apply, operands, resultType);
  return result;
}

//===----------------------------------------------------------------------===//
// ExprEmitter::emitCallUnchecked
//===----------------------------------------------------------------------===//

/// The results of calls to async functions are always bound to a `Coroutine`
/// type, or `RaisingCoroutine` type in the case of a raising function. This
/// function looks up the corresponding coroutine type and binds its result
/// type.
static ASTType getBoundCoroutineType(SharedState &shared, ASTDecl &declScope,
                                     SMLoc loc, SignatureType sig,
                                     Type resultType) {
  ASTType coroType = sig.isThrows()
                         ? shared.getBuiltinRaisingCoroutineType(declScope, loc)
                         : shared.getBuiltinCoroutineType(declScope, loc);
  if (!coroType) {
    shared.emitError(loc,
                     "internal error: could not find builtin 'Coroutine' type");
    return {};
  }
  // If the async function throws, extract the normal result type.
  if (sig.isThrows())
    resultType = cast<VariantType>(resultType).getTypes().back();

  // Bind the result type to the base coroutine type.
  auto typeExpr =
      TypeConstantAttr::get(resultType, TypeType::get(shared.getContext()));
  return BindTypeAttr::get(PValue(coroType), typeExpr);
}

/// Emit warnings about incorrect code in a direct call.  This is invoked after
/// the full IR for the call is emitted, so we know that it was a valid call.
void CallEmitter::emitDirectCallWarnings(LIT::CallOp call,
                                         const CallOperands &callOperands) {
  SymbolConstantAttr symbol = call.getCallee();

  // Figure out what is getting called.
  ASTDecl *calleeDecl =
      emitter.getDeclResolver().getDeclForFuncSymbol(symbol.getSymbol());
  if (!calleeDecl)
    return;
  auto calleeFunc = cast<LIT::FuncOp>(*calleeDecl);

  // Check to see if this is a self-recursive function call.
  if (ASTDecl *callerDecl =
          emitter.declScope.getNearestDeclOfType<LIT::FuncOp>()) {
    if (calleeDecl == callerDecl) {
      auto callerFunc = cast<LIT::FuncOp>(*callerDecl);

      // We only diagnose self-recursive calls with obviously identical
      // arguments or parameters.  Note that we don't need to check argument
      // conventions here because you don't need to pass
      bool allIdentical = true;
      assert(call.getNumOperands() == callerFunc.getNumArguments() &&
             "argument mismatch");
      for (auto [argValue, argDecl] :
           llvm::zip(call.getOperands(), callerFunc.getArguments())) {
        if (argValue != argDecl) {
          allIdentical = false;
          break;
        }
      }
      // Check to see if all parameters match.
      if (allIdentical) {
        SmallVector<ParamDeclAttr> paramDecls =
            callerFunc.collectAllParams(/*includeImplLifetimes=*/false);
        assert(symbol.getParamValues().size() == paramDecls.size() &&
               "parameter mismatch");
        for (auto [paramValue, paramDecl] :
             llvm::zip(symbol.getParamValues(), paramDecls)) {
          auto valueRef = dyn_cast<ParamDeclRefAttr>(paramValue);
          if (!valueRef || valueRef.getName() != paramDecl.getName()) {
            allIdentical = false;
            break;
          }
        }
      }

      if (allIdentical) {
        emitter.emitWarning(loc)
            << "self recursive call will cause an infinite loop"
            << callExpr->getRange();
        return;
      }
    }
  }

  // The __del__ special function takes its operand as an owning reference,
  // and destroys it.  It is a bit silly, but you can call it directly on an
  // RValue and it will destroy the RValue explicitly.  However, some folks
  // will call it on a local variable (or other !RValue reference) which will
  // actually cause a COPY of the source value, and then explicitly destroy
  // this copy of the value.  Emit a warning in this case.
  if (calleeFunc.getSpecialFunctionKind() == SpecialFunctionKind::kDel &&
      callOperands.posOperands.size() == 1 && // defensive.
      callOperands.posOperands[0].ir.getIfRValue().isNull()) {
    emitter.emitWarning(loc) << "explicit call to '__del__' destroys a copy of "
                                "the value; consider removing this call"
                             << callOperands.posOperands[0].expr->getRange();
    return;
  }
}

CValue ExprEmitter::emitCallUnchecked(CRValue callee,
                                      const CallOperands &callOperands,
                                      ValueDest &dest,
                                      const ExprNode *callExpr) {
  CallEmitter callEmitter(callee, callExpr, *this, dest);
  auto calleeSig = cast<LITSignatureType>(callee.getType());

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

  // HACK: If any of the arguments are nonmaterializable and all arguments are
  // PValues, then emit the call in the parameter context.
  auto isPValue = [](ASTExprAnd<AnyValue> arg) { return arg.ir.getIfPValue(); };
  auto isNonMaterializable = [&](ASTExprAnd<AnyValue> arg) {
    return arg.ir.getIfPValue().getType().getNonmaterializableTarget(shared);
  };
  bool forceParameterCall = llvm::all_of(argumentValues, isPValue) &&
                            llvm::any_of(argumentValues, isNonMaterializable);
  if (!builder || forceParameterCall) {
    TypedAttr paramCallResult;
    {
      llvm::SaveAndRestore savedBuilder(builder, {});
      assert(dest.getContext() != EC_Unknown &&
             "parametric emitCallUnchecked must include an ExprContext");
      llvm::SaveAndRestore savedContext(paramContext, dest.getContext());
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

  Location loc = translateLocation(callExpr->getLoc());

  // Otherwise, materialize PValue and DLValue's as SSA values for emission.
  SmallVector<Value> callArgs;
  SmallVector<TypedAttr> implicitLifetimes;
  for (auto [argIdx, argValAndExpr, conventionX] :
       llvm::enumerate(argumentValues, calleeSig.getArgConventions())) {
    ArgConvention convention = conventionX;

    // If this is a byref_result slot, we will have emitted a null value for
    // this.  We can't know the type of it until we emit all the operands and
    // collect their lifetimes.
    // TODO(#30134): This should be at the end of the argument list instead of
    // the beginning!
    if (convention == ArgConvention::ByRefResult) {
      // Don't know the right thing yet, use a placeholder.
      callArgs.push_back(Value());
      implicitLifetimes.push_back(
          LifetimeAttr::get(getContext(), /*isMutable*/ true));
      continue;
    }

    // If this is a variadic operation, the N operands have already been emitted
    // together and consolidated into a pop.variadic.create/pop.variadic.attr,
    // which is emitted as an SRValue instead of whatever the underlying type
    // is.
    if (calleeSig.isVarArg(argIdx) || calleeSig.isPackVarArg(argIdx))
      convention = ArgConvention::OwnedInReg;

    Value arg = callEmitter.emitPreemittedArgumentAsDynamicValue(
        argValAndExpr, convention, argumentValues);
    if (!arg)
      return {};

    // See if we have an implicit lifetime bound for this argument.
    if (SignatureType::hasAddress(convention)) {
      implicitLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());
    } else if (calleeSig.isVarArg(argIdx)) {
      // If this is a variadic, it will have a wrapper around the ref.
      auto eltType = ASTType(arg.getType()).getVariadicElementType();
      if (auto refType = dyn_cast<RefType>(eltType))
        implicitLifetimes.push_back(refType.getLifetime());
    }

    // TODO: What about pack types?
    callArgs.push_back(arg);
  }

  // Now that we have the lifetimes for the arguments, we can calculate what the
  // substituted signature should be.  This will take in the wrong byref-result
  // lifetime as an input, but we know that it cannot be referenced in the type
  // anyway, because there is no way to name it in the Mojo program.
  FunctionType expectedCalleeType =
      calleeSig.substituteImplicitLifetimesIntoValues(
          implicitLifetimes,
          [&]() -> InFlightDiagnostic { llvm_unreachable("bad call"); });

  // With that done, we can know what type the byref result slot should have.
  // We see if we can emit directly into the ValueDest slot, and if not, we
  // create a VarDecl temporary and allow emitResult to copy it over to the
  // destination.
  if (calleeSig.hasMemoryOnlyResult()) {
    auto resultRValueType =
        cast<RefType>(expectedCalleeType.getInput(0)).getElementType();

    // Often the result of the call will be directly assigned into a
    // user-defined var or other location with existing storage.  In these
    // cases, we really want to assign directly into the existing slot.
    //
    // However, we cannot do that if the destination slot is also being passed
    // into the call as an input value, as in: `x = foo(x)` or `x = x + 1`.
    // In these cases we really do need a temporary+copy in the var slot.
    // At this point we've got enough information about the arguments to make
    // that assessment in a correct way.
    if (callEmitter.isSafeToUseValueDestForDirectResult(resultRValueType,
                                                        argumentValues)) {
      // Use the preferred location of the destination slot.
      callArgs[0] = dest.getMLValueForResult(
          callExpr->getLoc(), resultRValueType, callEmitter.emitter);
    } else {
      // Create a temporary.
      callArgs[0] = callEmitter.emitter.emitVarLetDecl(
          "__call_result_tmp__", resultRValueType, loc,
          VarLetDeclKind::Synthesized);
    }

    implicitLifetimes[0] = cast<RefType>(callArgs[0].getType()).getLifetime();
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

    // Ignore the byref result slot - we know its lifetime differs but we're ok
    // with that.
    if (arg == callArgs[0] && calleeSig.hasMemoryOnlyResult())
      continue;

    // Insert a rebind.
    arg = builder->create<RebindOp>(loc, expectedType, arg);
  }

  assert(expectedCalleeType.getResults().size() == 1 &&
         "All mojo functions return one value");
  Type resultType = expectedCalleeType.getResults()[0];
  Value callResult;
  if (auto target = callee.getIfPValue()) {
    if (auto sig = dyn_cast<SignatureType>(target.getType());
        sig && sig.isAsync()) {
      // If the callee is an async function, emit an async call. Then wrap the
      // `!pop.coroutine<() -> T>` result in a `Coroutine[T]` object.
      auto call = builder->create<AsyncCallOp>(
          loc,
          POP::CoroutineType::get(getContext(), resultType, sig.isThrows()),
          target.get(), implicitLifetimes, callArgs);
      ASTType coroType = getBoundCoroutineType(
          shared, declScope, callExpr->getLoc(), sig, resultType);
      if (!coroType) {
        dest.resetForError();
        return {};
      }
      ValueDest ctorDest(dest.getContext());
      // Emit the implicit conversion.
      callResult =
          emitConstructorCall(coroType, {{{SBValue(call), callExpr}}}, callExpr,
                              CallSyntax::kImplicitConvert, ctorDest)
              .getIfSRValue();
    } else if (auto symbol = dyn_cast<SymbolConstantAttr>(target.get())) {
      // If the callee is a symbol constant, directly emit a call.
      auto call = builder->create<CallOp>(loc, resultType, symbol,
                                          implicitLifetimes, callArgs);
      callResult = call.getResult(0);

      // If there are any callee-specific warnings to emit, do so after
      // successfully emitting the call.
      callEmitter.emitDirectCallWarnings(call, callOperands);
    } else {
      auto call = builder->create<CallParamOp>(loc, resultType, target.get(),
                                               implicitLifetimes, callArgs);
      callResult = call.getResult(0);
    }
  } else {
    Value calleeVal = callee.getIfSRValue();
    assert(calleeVal && "don't have a callee of expected type");
    auto call = builder->create<CallSignatureOp>(loc, resultType, calleeVal,
                                                 implicitLifetimes, callArgs);
    callResult = call.getResult(0);
  }

  // If there were any writebacks to handle, emit them before handling raised
  // errors.
  callEmitter.emitAfterCallActions();

  SmallVector<Value, 1> byRefResults;
  if (calleeSig.hasMemoryOnlyResult() || calleeSig.hasInitSelfResult())
    byRefResults.push_back(callArgs[0]);

  // If the callee can raise an error, it will be represented as a variant: try
  // to unwrap it. If the callee is async, the error is propagated later.
  if (calleeSig.isThrows() && !calleeSig.isAsync()) {
    // Put the insertion point back after we're done building the 'if'.
    OpBuilder::InsertionGuard builderGuard(*builder);
    auto callResultTy = cast<VariantType>(callResult.getType());
    Type successType = callResultTy.getType(1);
    auto handleVariant = builder->create<LIT::HandleVariantOp>(
        loc, successType, callResult, ValueRange(byRefResults));
    Block *successBlock =
        builder->createBlock(&handleVariant.getSuccessRegion());
    builder->setInsertionPointToStart(successBlock);
    Value value = builder->create<VariantTakeOp>(loc, callResult, 1);
    builder->create<LIT::YieldOp>(loc, value);

    Block *errorBlock = builder->createBlock(&handleVariant.getErrorRegion());
    builder->setInsertionPointToStart(errorBlock);
    Value error = builder->create<VariantTakeOp>(loc, callResult, 0);
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
