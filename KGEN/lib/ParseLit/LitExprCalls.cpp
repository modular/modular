//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "LitExprCalls.h"
#include "ASTDecl.h"
#include "LitExprEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// CallableValue Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This struct indicates whether a signature can be successfully applied to a
/// parameter binding and argument list.  If so, it keeps track of the number of
/// implicit conversions required to make the call, and if not, it indicates the
/// reason for the mismatch.
struct OverloadFitness {
  enum Kind {
    kValid,          //< This is a valid candidate.
    kParamCount,     //< Invalid due to a parameter count mismatch
    kParamWrongType, //< A parameter value cannot be converted to expected type
    kArgCount,       //< Invalid due to a argument count mismatch
    kArgNotLValue,   //< By-ref argument requires an lvalue, but got an rvalue.
    kArgWrongLVType, //< By-ref argument and provided l-value types mismatch.
    kArgWrongType,   //< An argument value cannot be converted to expected type
  } kind;

  /// The interpretation of this payload depends on the 'kind' field:
  ///  kValid:          number of implicit conversions required.
  ///  kParamCount:     Not used.
  ///  kParamWrongType: the parameter # that mismatches.
  ///  kArgCount:       number of arguments expected.
  ///  kArgNotLValue:   the argument # that mismatches.
  ///  kArgWrongLVType: the argument # that mismatches.
  ///  kArgWrongType:   the argument # that mismatches.
  size_t payload;

  /// For type mismatches, this is the actual or expected type, otherwise null.
  ASTType type;

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable` and the arguments specified in
  /// `operands`.
  static OverloadFitness evaluate(SignatureType signature,
                                  const DirectCallable &callable,
                                  ArrayRef<ASTExprAnd<AnyValue>> operands,
                                  LitSharedState &shared);

  /// Add explaination for why this candidate doesn't work to the specified
  /// diagnostic.
  void diagnose(SignatureType signature, const DirectCallable &callable,
                ArrayRef<ASTExprAnd<AnyValue>> operands, bool isMethodCall,
                Diagnostic &diag);
};
} // namespace

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `operands`.
OverloadFitness OverloadFitness::evaluate(
    SignatureType signature, const DirectCallable &callable,
    ArrayRef<ASTExprAnd<AnyValue>> operands, LitSharedState &shared) {

  // Check that the signature can be rebound with our set of bindings.
  ssize_t incorrectBindingNo = 0;
  auto newBindings =
      callable.getCheckedBindings(signature, incorrectBindingNo,
                                  /*don't emit diagnostics*/ {}, shared);

  // If there is an error, return the problem.
  if (!newBindings) {
    if (incorrectBindingNo == -1)
      return {kParamCount, 0, ASTType()};
    return {kParamWrongType, (size_t)incorrectBindingNo, ASTType()};
  }

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  if (!newBindings.empty()) {
    signature = signature.getSpecializedSignature(
        newBindings, [&]() -> InFlightDiagnostic {
          llvm_unreachable("bad bindings went undetected");
        });
    assert(signature && "bad bindings went undetected");
  }

  // Ok, the parameters all line up, check the argument list.
  size_t numArgs = signature.getValues().getNumInputs();
  if (numArgs != operands.size())
    return {kArgCount, numArgs, ASTType()};

  size_t argIdx = 0;
  size_t numImplicitConversions = 0;
  for (auto [argAnyValueAndExpr, expectedType, convention] :
       llvm::zip(operands, signature.getValueInputs(),
                 signature.getValueInputConventions())) {
    switch (convention) {
    case ValueInputConvention::ByRef: {
      // The actual value must be an lvalue if callee takes things by-ref.
      auto argVal = argAnyValueAndExpr.ir.getIfLValue();
      if (!argVal)
        return {kArgNotLValue, argIdx, argAnyValueAndExpr.ir.getType()};

      // By-ref argument types must exactly match, no conversions are allowed.
      if (!ASTType(argVal.getType()).isEqualCanon(ASTType(expectedType)))
        return {kArgWrongLVType, argIdx, expectedType};
      break;
    }
    case ValueInputConvention::ByVal:
      // Otherwise, we pass as an r-value.  If the argument types match, then
      // they are good.
      if (ASTType(argAnyValueAndExpr.ir.getRValueType())
              .isEqualCanon(ASTType(expectedType)))
        break;

      // Otherwise, check to see if we can do an implicit conversion.
      bool isErroneousDecl = false;
      CallableValue callee(expectedType, "__new__",
                           argAnyValueAndExpr.expr->getLoc(), isErroneousDecl,
                           shared);

      // Check to see if we have any viable candidates for the implicit
      // conversion.  If not, we have an argument conversion error.
      if (!callee.direct || failed(callee.direct->filterOverloadSet(
                                {argAnyValueAndExpr}, /*isMethodSyntax*/ false,
                                /*emitDiagnosticOnFailure=*/false, shared)))
        return {kArgWrongType, argIdx, expectedType};

      // If we had one, this bumps our # implicit conversions.
      ++numImplicitConversions;
    }
    ++argIdx;
  }

  return {kValid, numImplicitConversions, ASTType()};
}

/// Add explaination for why this candidate doesn't work to the specified
/// diagnostic.
void OverloadFitness::diagnose(SignatureType signature,
                               const DirectCallable &callable,
                               ArrayRef<ASTExprAnd<AnyValue>> operands,
                               bool isMethodCall, Diagnostic &diag) {
  // TODO: Would be really nice to range underline the operand in question!
  switch (kind) {
  case kValid:
    diag << "candidate is viable";
    return;
  case kParamCount:
    diag << "callee expects " << signature.getInputParams().size()
         << " input parameter" << plural(payload) << " but "
         << callable.bindings.size()
         << plural(callable.bindings.size(), " was", " were") << " provided";
    return;
  case kParamWrongType: {
    auto decl = signature.getInputParams()[payload];
    auto valueType = callable.bindings[payload].getType();
    diag << "callee parameter " << decl.getName() << " has "
         << ASTType(decl.getType()) << " type, but value has type "
         << ASTType(valueType);
    return;
  }
  case kArgCount:
    diag << "callee expects " << payload << " argument" << plural(payload);
    return;
  case kArgNotLValue:
    if (isMethodCall && payload == 0) {
      diag << "invalid use of mutating method on rvalue of type "
           << ASTType(type);
      return;
    }
    diag << "operand must be mutable in order to pass as a by-ref argument";
    return;
  case kArgWrongLVType:
    diag << "l-value of type " << operands[payload].ir.getRValueType()
         << " cannot be converted to reference to expected type "
         // TODO(QoI): Types are not attributes.
         << cast<POP::PointerType>(Type(type)).getElementType();
    return;

  case kArgWrongType:
    // If this is a method syntax call, don't count the receiver.
    if (isMethodCall) {
      // it is probably possible for this assert to fire, if it does we should
      // tailor the error message.
      assert(payload != 0 && "TODO: unexpected self mismatch");
      --payload;
    }
    diag << "in argument #" << payload << ", value of type "
         << operands[payload].ir.getRValueType()
         << " cannot be converted to expected type " << type;
    break;
  }
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings and provided
/// arguments.  If so, replace fnDecls with a single entry that works and
/// return success.  If not, generate a diagnostic and return failure.
LogicalResult DirectCallable::filterOverloadSet(
    ArrayRef<ASTExprAnd<AnyValue>> operands, bool isMethodCall,
    bool emitDiagnosticOnFailure, LitSharedState &shared) {
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto signature = cast<LIT::FuncOp>(*candidate).getFullSignature();
    evaluations.push_back(
        OverloadFitness::evaluate(signature, *this, operands, shared));
    anyValid |= evaluations.back().kind == OverloadFitness::kValid;
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure) {
      // TODO(QoI): Handle the case of zero candidates.

      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = shared.emitError(loc, "invalid call: ");
        evaluations[0].diagnose(fnDecl.getFullSignature(), *this, operands,
                                isMethodCall, *diag.getUnderlyingDiagnostic());
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return failure();
      }

      // Otherwise emit an error, and a note for what is wrong with each
      // candidate.
      auto diag = shared.emitError(loc, "no matching function in call");
      for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
        auto fnDecl = cast<LIT::FuncOp>(*candidate);
        eval.diagnose(fnDecl.getFullSignature(), *this, operands, isMethodCall,
                      diag.attachNote(fnDecl->getLoc())
                          << "candidate not viable: ");
      }
      return failure();
    }
  }

  // Ok, we have at least one valid candidate, filter the list to the ones with
  // the lowest number of implicit conversions required.
  size_t minConversions = ~0U;
  SmallVector<ASTDecl *, 1> newFnDecls;
  for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
    // Ignore failures or candidates that have more conversions.
    if (eval.kind != OverloadFitness::kValid || eval.payload > minConversions)
      continue;

    // If we found a new floor to the # conversions needed, clear the list.
    if (eval.payload < minConversions) {
      newFnDecls.clear();
      minConversions = eval.payload;
    }
    newFnDecls.push_back(candidate);
  }

  // If we found exactly one viable candidate, then we succeed.
  if (newFnDecls.size() == 1) {
    fnDecls = std::move(newFnDecls);
    return success();
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure) {
    auto diag =
        shared.emitError(loc, "ambiguous call, each candidate requires ")
        << minConversions << " implicit conversion" << plural(minConversions)
        << ", disambiguate with an explicit cast";
    for (ASTDecl *candidate : newFnDecls)
      diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
          << "candidate declared here";
  }
  return failure();
}

/// Check that our set of parameter bindings work with the specified signature
/// type, returning a checked ParamBindArrayAttr if so.  If the parameters do
/// not work, this emits an diagnostic (if `shared` is non-null) and sets
/// `incorrectBindingNo` to the bad binding (or -1 if there is a count
/// mismatch).
ParamBindArrayAttr DirectCallable::getCheckedBindings(
    SignatureType signature, ssize_t &incorrectBindingNo,
    Optional<Location> funcLoc, LitSharedState &shared) const {

  // We require an exact match for the signature right now, we don't allow
  // inference or other fancy things.
  auto expectedNumParams = signature.getInputParams().size();
  if (bindings.size() != expectedNumParams) {
    if (funcLoc) {
      auto diag = shared.emitError(loc, "function expects ")
                  << expectedNumParams << " input parameter"
                  << plural(expectedNumParams) << " but " << bindings.size()
                  << plural(bindings.size(), " was", " were") << " provided";
      diag.attachNote(*funcLoc) << "function declared here";
    }
    incorrectBindingNo = -1;
    return {};
  }

  // If we have bound parameters, type check them now and bind names to them.
  SmallVector<ParamBindAttr> newBindings;
  newBindings.reserve(bindings.size());

  for (auto [bound, decl] : llvm::zip(bindings, signature.getInputParams())) {
    // If this value was already bound and checked, use it.
    auto prebound = dyn_cast<ParamBindAttr>(bound.bindingOrValue);
    if (prebound) {
      newBindings.push_back(prebound);
      continue;
    }

    // Check the type matches what is expected.
    // TODO: Do implicit conversions when we can invoke parameter functions.
    // TODO: Handle signatures like (T, scalar<T>) where early bound
    // parameters changes the types of later ones.
    auto value = bound.getValue();
    auto valueType = value.getType();
    if (!ASTType(valueType).isEqualCanon(decl.getType())) {
      if (funcLoc) {
        auto diag = shared.emitError(bound.loc, "parameter ")
                    << decl.getName() << " has " << ASTType(decl.getType())
                    << " type, but value has type " << ASTType(valueType);
        diag.attachNote(*funcLoc) << "function declared here";
      }
      incorrectBindingNo = newBindings.size();
      return {};
    }
    newBindings.push_back(ParamBindAttr::get(decl, value));
  }

  return ParamBindArrayAttr::get(signature.getContext(), newBindings);
}

/// Generate a reference to the specified function, checking that any supplied
/// parameters are correct and match expectations..
SymbolConstantAttr
DirectCallable::getBoundConstantAttr(LitSharedState &shared) const {
  if (fnDecls.size() != 1) {
    assert(!fnDecls.empty() && "DirectCallable malformed");
    auto diag = shared.emitError(
        loc, "cannot form a reference to overloaded declaration");
    for (ASTDecl *candidate : fnDecls) {
      auto funcOp = cast<LIT::FuncOp>(*candidate);
      diag.attachNote(funcOp.getLoc()) << "candidate declared here";
    }

    return {};
  }

  auto funcOp = cast<LIT::FuncOp>(*fnDecls[0]);

  // Check that the signature can be rebound with our set of bindings.
  ssize_t incorrectBindingNo = 0;
  auto newBindings =
      getCheckedBindings(funcOp.getFullSignature(), incorrectBindingNo,
                         /*emit diagnostics*/ funcOp.getLoc(), shared);
  if (!newBindings)
    return {};

  // Now that we checked the types match, form the binding.
  return funcOp.getBoundReference(newBindings);
}

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
DirectCallable::DirectCallable(SMLoc loc, ArrayRef<ASTDecl *> fnDecls,
                               ParamBindArrayAttr bindingsAttr)
    : loc(loc), fnDecls(fnDecls.begin(), fnDecls.end()) {
  if (bindingsAttr) {
    for (ParamBindAttr bind : bindingsAttr)
      bindings.push_back({loc, bind});
  }
}

/// Get a CallableValue for a lookup of a named method on the specified type.
/// If successful, this provides a non-null CallableValue.  On failure, it
/// emits an error and returns a null CallableValue.
CallableValue::CallableValue(ASTType type, StringRef methodName, SMLoc callLoc,
                             LitSharedState &shared) {
  bool erroneousDecl = false;
  lookup(type, methodName, callLoc, /*emitErrorOnFailure=*/true, erroneousDecl,
         shared);
}

/// Get a CallableValue for a lookup of a named method on the specified type.
/// If successful, this provides a non-null CallableValue.
///
/// On failure, this returns a null CallableValue and sets 'erroneousDecl' to
/// indicate whether there was a problem with the callee that has already been
/// diagnosed (allowing the client to squish downstream error messages).  This
/// does not emit an error on failure.
CallableValue::CallableValue(ASTType type, StringRef methodName, SMLoc callLoc,
                             bool &erroneousDecl, LitSharedState &shared) {
  lookup(type, methodName, callLoc, /*emitErrorOnFailure=*/false, erroneousDecl,
         shared);
}

/// Get a CallableValue for a lookup of a named method on the specified type.
/// If successful, this provides a non-null CallableValue.
///
/// On failure, this returns a null CallableValue and sets 'erroneousDecl' to
/// indicate whether there was a problem with the callee that has already been
/// diagnosed (thus squishing downstream error messages).  If
/// emitErrorOnFailure is true an error message indicates why the call failed.
void CallableValue::lookup(ASTType type, StringRef methodName, SMLoc callLoc,
                           bool emitErrorOnFailure, bool &erroneousDecl,
                           LitSharedState &shared) {
  erroneousDecl = false;
  // First perform a lookup to see if there are any candidates.
  auto lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, type);
  ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
  if (resultDecls.empty()) {
    if (lookupResult.isErroneous())
      erroneousDecl = true;
    else if (emitErrorOnFailure)
      shared.emitError(callLoc, "") << type << " does not implement the '"
                                    << methodName << "' special method";
    return;
  }

  // If we find a vardecl or any other thing, then fail because it cannot be
  // called.
  if (!isa<LIT::FuncOp>(*resultDecls[0])) {
    if (emitErrorOnFailure)
      shared.emitError(callLoc, "member '")
          << methodName << "' of " << type << " is not a method";
    return;
  }

  // Handle method references, which might be overloaded.
  direct = DirectCallable{callLoc, resultDecls, type.getParamBindings()};
}

/// Emit this as a flattened RValue or LValue with no additional parameter
/// context.  This returns null on failure.
AnyValue CallableValue::emitAsValue(ExprEmitter &emitter) const {
  // If we have no bound symbol, return the normal lvalue or rvalue we
  // represent.
  if (!direct)
    return baseVal.ir;

  auto directSymbolAttr = direct->getBoundConstantAttr(emitter.shared);
  if (!directSymbolAttr)
    return {};

  // If we have no base value, then we are just a symbol, return it.
  if (!baseVal)
    return MValue(directSymbolAttr);

  auto loc = baseVal.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  SignatureType calleeSignature = directSymbolAttr.getType();
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  Value firstArgValue;
  switch (calleeSignature.getInputConvention(0)) {
  case ValueInputConvention::ByRef: {
    LValue baseLV = baseVal.ir.getIfLValue();
    if (!baseLV) {
      emitter.emitError(loc,
                        "invalid use of mutating method on rvalue of type ")
          << ASTType(baseVal.ir.getType());
      return {};
    }

    // TODO: Using partial application over an lvalue like this isn't
    // technically safe.  We need to extend the lifetime of the pointer captured
    // for as long as the partial application thunk is alive. This will require
    // some sort of borrow model.  In practice, this will be fine in the short
    // term of Lit bringup because the thunk cannot be emitted independently
    // anyway, it must always be canonicalized into another call.
    firstArgValue = baseLV;
    break;
  }
  case ValueInputConvention::ByVal:
    // Otherwise we can have either an lvalue or rvalue, but we need to convert
    // to an rvalue if we have an lvalue.
    firstArgValue = emitter.emitDRValue(baseVal.ir, loc);
    if (!firstArgValue)
      return {};
    break;
  }

  assert(firstArgIRType == firstArgValue.getType() &&
         "base types should always structurally line up");

  // For an instance value, we have to partially apply the callee to the first
  // argument of the reference.  Materialize callee as a DRValue for
  // partial_apply.
  auto calleeDRVal = emitter.emitDRValue(AnyValue(directSymbolAttr), loc);

  // Partial apply wants to know what operands to bind, we always bind the first
  // one.
  auto zeroAttr = emitter.builder->getAttr<mlir::DenseI64ArrayAttr>(0);
  return DRValue(emitter.builder->create<POP::PartialApplyOp>(
      emitter.translateLocation(loc), calleeDRVal,
      mlir::ValueRange(firstArgValue), zeroAttr));
}

//===----------------------------------------------------------------------===//
// Call Emission Implementation
//===----------------------------------------------------------------------===//

/// Returns true if the insertion context is valid for implicit error
/// propagation.
static bool isValidErrorContext(Block *block) {
  for (Operation *op = block->getParentOp(); op; op = op->getParentOp()) {
    if (isa<TryOp>(op))
      return true;
    if (auto func = dyn_cast<LIT::FuncOp>(op))
      return func.getRaises();
  }
  return false;
}

/// Emit a function call to the specified callee with the specified operand
/// values.  This emits an error and returns null on failure.
AnyValue
CallableValue::emitFunctionCall(ArrayRef<ASTExprAnd<AnyValue>> operands,
                                SMLoc callLoc, ExprEmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Set to true if this is a method call like `x.foo(...`.
  bool isMethodCall = false;
  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> operandsWithSelf;

  auto emitError = [&](const Twine &message) {
    return emitter.emitError(callLoc, message);
  };

  // Figure out the type of the function to call, which is either symbol or a
  // normal rvalue.
  SignatureType calleeSig;

  // This is the callee symbol constant for a direct call, or the SSA value for
  // an indirect call.
  PointerUnion<Attribute, Value> callee;
  if (direct) {
    // If we have a bound self, add it to the operand list to simplify the logic
    // below.
    if (baseVal) {
      operandsWithSelf.reserve(operands.size() + 1);
      operandsWithSelf.push_back(baseVal);
      operandsWithSelf.append(operands.begin(), operands.end());
      operands = operandsWithSelf;
      baseVal = {};
      isMethodCall = true;
    }

    // Check the direct callees to see if they can be unambiguously resolved
    // with the bindings list and specified arguments.
    if (failed(direct->filterOverloadSet(operands, isMethodCall,
                                         /*emitDiagnosticOnFailure=*/true,
                                         emitter.shared)))
      return {};
    SymbolConstantAttr symbol = direct->getBoundConstantAttr(emitter.shared);
    if (!symbol)
      return {};

    calleeSig = symbol.getType();
    callee = symbol;
  } else {
    // Otherwise we have an indirect call, emit the callee value as a DRValue so
    // we can call it with call_indirect.
    auto calleeDRVal = emitter.emitDRValue(baseVal.ir, callLoc);
    if (!calleeDRVal)
      return {};
    callee = calleeDRVal;

    calleeSig = dyn_cast<SignatureType>(calleeDRVal.getType());
    if (!calleeSig) {
      emitError("invalid function type to call ")
          << ASTType(calleeDRVal.getType());
      return {};
    }

    // Check to see if we can apply these operands to the callee signature.
    DirectCallable bindings{callLoc, {}, {}}; // No additional bound parameters.
    auto fitness = OverloadFitness::evaluate(calleeSig, bindings, operands,
                                             emitter.shared);
    if (fitness.kind != OverloadFitness::kValid) {
      // If not, diagnose it with an error.
      auto diag = emitError("invalid indirect call: ");
      fitness.diagnose(calleeSig, bindings, operands, isMethodCall,
                       *diag.getUnderlyingDiagnostic());
      return {};
    }
  }

  assert(calleeSig.getResultParamTypes().empty() &&
         "TODO: meta results not implemented yet");
  assert(calleeSig.getValues().getNumInputs() == operands.size() &&
         "Type checking should be done");

  // Emit all the arguments.
  SmallVector<Value> valueArguments;
  for (auto [argAnyValueAndExpr, expectedType, convention] :
       llvm::zip(operands, calleeSig.getValueInputs(),
                 calleeSig.getValueInputConventions())) {
    auto argLoc = argAnyValueAndExpr.expr->getLoc();
    // If the callee takes the operand as a by-ref argument, we require an
    // lvalue.
    Value argVal;
    switch (convention) {
    case ValueInputConvention::ByRef:
      argVal = argAnyValueAndExpr.ir.getIfLValue();
      assert(argVal && "Call should already be type checked");
      break;
    case ValueInputConvention::ByVal:
      // Otherwise, we pass as an r-value.
      argVal = emitter.emitDRValue(argAnyValueAndExpr.ir, argLoc);
      if (!argVal)
        return {};

      // Convert the argument to the expected type if needed.
      argVal = emitter.getAsExpectedType(argVal, argAnyValueAndExpr.expr,
                                         expectedType);
      if (!argVal)
        return {};
      break;
    }

    valueArguments.push_back(argVal);
  }

  auto &builder = emitter.builder;
  if (!builder) {
    emitError("TODO: cannot call function in parameter context");
    return {};
  }

  // If this is a call to something representable as an attribute, we can use
  // a kgen.call_param.
  Value resultVal;
  auto loc = emitter.translateLocation(callLoc);
  // FIXME: Move result type inference into CallOp/CallIndirectOp.
  auto resultTypes = calleeSig.getValueResults();
  if (auto target = dyn_cast<Attribute>(callee)) {
    resultVal =
        builder
            ->create<CallOp>(loc, resultTypes, cast<SymbolConstantAttr>(target),
                             ArrayRef<ParamDeclAttr>(), valueArguments)
            .getResult(0);
  } else {
    // Otherwise emit calls to SSA values with call_indirect.
    auto calleeDRVal = cast<Value>(callee);
    resultVal = builder
                    ->create<POP::CallIndirectOp>(loc, resultTypes, calleeDRVal,
                                                  /*operands*/ valueArguments)
                    .getResult(0);
  }

  // If the callee can raise an error, try to unwrap it.
  if (calleeSig.getFnEffects() == FnEffects::Throws) {
    if (!isValidErrorContext(builder->getInsertionBlock())) {
      emitError(
          "cannot call raising method within an 'fn' that does not raise");
      return {};
    }
    resultVal = builder->create<UnwrapOrPropagateOp>(
        loc, cast<POP::VariantType>(resultVal.getType()).getType(1), resultVal);
  }

  // Value returning call returns its result.
  return DRValue(resultVal);
}
