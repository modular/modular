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
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// InputParamBindings Implementation
//===----------------------------------------------------------------------===//

/// Check that our set of parameter bindings work with the specified input
/// parameters, returning a checked ParamBindArrayAttr if so.  If the parameters
/// do not work, this emits an diagnostic (if `declOp` is non-null) and set
/// `incorrectBindingNo/Expectedtype` to the bad binding (or -1 if there is a
/// count mismatch).
ParamBindArrayAttr InputParamBindings::verifyBindings(
    ParamDeclArrayAttr actualParamDecls, StringRef baseName, SMLoc loc,
    ssize_t &incorrectBindingNo, ASTType &incorrectBindingExpectedType,
    LitSharedState &shared, Operation *declOp) const {
  // If there are no bindings, exit early.
  // FIXME: This all or nothing thing is really weird and needs to be fixed when
  // we start infering parameter bindings from arguments etc.
  if (bindings.empty())
    return ParamBindArrayAttr::get(shared.getContext(), {});

  // We require an exact match for the actualParamDecls right now, we don't
  // allow inference or other fancy things.
  auto actualNumParams = bindings.size();
  auto expectedNumParams = actualParamDecls.size();
  if (actualNumParams != expectedNumParams) {
    if (declOp) {
      auto diag = shared.emitError(loc, "'")
                  << baseName << "' expects " << expectedNumParams
                  << " input parameter" << plural(expectedNumParams) << " but "
                  << actualNumParams << plural(actualNumParams, " was", " were")
                  << " provided";
      diag.attachNote(declOp->getLoc()) << "'" << baseName << "' declared here";
    }
    incorrectBindingNo = -1;
    return {};
  }

  // If we have bound parameters, type check them now and bind names to them.
  SmallVector<ParamBindAttr> newBindings;
  newBindings.reserve(actualNumParams);

  // This is the IR emitter we use for emitting implicit conversions when
  // needed.
  IREmitter emitter(shared, /*no builder*/ {});

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParameterEvaluator to keep track of the mapping
  // so far and remap types on demand.
  ParameterEvaluator evaluator;
  for (auto [boundX, declX] : llvm::zip(bindings, actualParamDecls)) {
    // Work around: "reference to local binding 'decl' declared in enclosing
    // function"
    auto bound = boundX;
    auto &decl = declX;

    // If this value was already bound and checked, use it.
    auto prebound = dyn_cast<ParamBindAttr>(bound.bindingOrValue);
    if (prebound) {
      evaluator.setParameterValue(prebound.getDecl(), prebound.getValue());
      newBindings.push_back(prebound);
      continue;
    }

    assert(bound.expr &&
           "should always have an expr tree for unchecked bindings");

    // Check the type matches what is expected, and perform an implicit
    // conversion if needed.
    auto expectedType = ASTType(evaluator.getReboundType(decl.getType()));
    auto argValue = emitter.getAsExpectedType(
        MValue(bound.getValue()), bound.expr, expectedType, [&]() {
          if (declOp) {
            auto diag = shared.emitError(bound.expr->getLoc(), "'")
                        << baseName << "' parameter " << decl.getName()
                        << " has " << expectedType
                        << " type, but value has type "
                        << ASTType(bound.getValue().getType())
                        << bound.expr->getRange();
            diag.attachNote(declOp->getLoc())
                << "'" << baseName << "' declared here";
          }
          incorrectBindingNo = newBindings.size();
          incorrectBindingExpectedType = expectedType;
        });
    if (!argValue)
      return {};

    auto argMValue = argValue.getIfMValue();
    assert(argMValue && "cannot emit a dynamic value in parameter context");

    // Any reference between parameters to this parameter will get our bound and
    // potentially type-converted value.
    evaluator.setParameterValue(decl, argMValue);

    // Update the decl's type if we remapped the type.
    ParamDeclAttr boundDecl = decl;
    if (decl.getType() != expectedType)
      boundDecl = ParamDeclAttr::get(decl.getName(), expectedType);

    newBindings.push_back(ParamBindAttr::get(boundDecl, argMValue));
  }

  return ParamBindArrayAttr::get(shared.getContext(), newBindings);
}

//===----------------------------------------------------------------------===//
// DirectCallable Implementation
//===----------------------------------------------------------------------===//

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
DirectCallable::DirectCallable(SMLoc nameLoc, StringRef baseName,
                               ArrayRef<ASTDecl *> fnDecls,
                               ParamBindArrayAttr bindingsAttr)
    : nameLoc(nameLoc), baseName(baseName),
      fnDecls(fnDecls.begin(), fnDecls.end()) {
  if (bindingsAttr) {
    for (ParamBindAttr bind : bindingsAttr)
      inputParamBindings.add(bind);
  }
}

namespace {
/// This struct indicates whether a signature can be successfully applied to a
/// parameter binding and argument list.  If so, it keeps track of the number of
/// implicit conversions required to make the call, and if not, it indicates the
/// reason for the mismatch.
struct OverloadFitness {
  enum Kind {
    kValid,            //< This is a valid candidate.
    kParamCount,       //< Invalid due to a parameter count mismatch
    kParamWrongType,   //< A parameter value not convertible to expected type
    kResultParamCount, //< Incorrect number of result params.
    kArgCount,         //< Invalid due to a argument count mismatch
    kArgNotLValue,     //< By-ref argument requires an lvalue, but got an rvalue
    kArgWrongLVType,   //< By-ref argument and provided l-value types mismatch.
    kArgWrongType,     //< An argument value not convertible to expected type
  } kind;

  /// The interpretation of this payload depends on the 'kind' field:
  ///  kValid:            number of implicit conversions required.
  ///  kParamCount:       not used.
  ///  kParamWrongType:   the parameter # that mismatches.
  ///  kResultParamCount: not used.
  ///  kArgCount:         number of arguments expected.
  ///  kArgNotLValue:     the argument # that mismatches.
  ///  kArgWrongLVType:   the argument # that mismatches.
  ///  kArgWrongType:     the argument # that mismatches.
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
                LitDiagnostic &diag);
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
  ASTType incorrectBindingExpectedType;
  auto newBindings = callable.inputParamBindings.verifyBindings(
      signature.getInputParams(), callable.baseName, callable.nameLoc,
      incorrectBindingNo, incorrectBindingExpectedType, shared,
      /*don't emit diagnostics*/ {});

  // If there is an error, return the problem.
  if (!newBindings) {
    if (incorrectBindingNo == -1)
      return {kParamCount, 0, ASTType()};
    return {kParamWrongType, static_cast<size_t>(incorrectBindingNo),
            incorrectBindingExpectedType};
  }

  // Check that we bound all the input parameters.  verifyBindings checks
  // bindings that are present, but doesn't check that they were all here.
  // TODO: We'll need to refactor this when infering parameter bindings from
  // arguments.
  if (newBindings.size() != signature.getInputParams().size())
    return {kParamCount, 0, ASTType()};

  // Check the result parameter count.
  if (signature.getResultParamTypes().size() != callable.resultParams.size())
    return {kResultParamCount, 0, ASTType()};

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
      auto argType = argAnyValueAndExpr.ir.getRValueType();
      // Otherwise, we pass as an r-value.  If the argument types match, then
      // they are good.
      if (argType.isEqualCanon(expectedType))
        break;

      // If we lack an exact match and conversions are disabled, this
      // candidate fails.
      if (callable.disableImplicitConversions ||
          !CallableValue::canImplicitlyConvertToType(argAnyValueAndExpr,
                                                     expectedType, shared))
        return {kArgWrongType, argIdx, expectedType};

      // If we had one, this bumps our # implicit conversions.
      ++numImplicitConversions;
    }
    ++argIdx;
  }

  return {kValid, numImplicitConversions, ASTType()};
}

/// Add explaination for why this candidate doesn't work to the specified
/// diagnostic. isMethodCall indicates whether the call was written with
/// `foo(x,y)` syntax or `x.foo(y)` syntax.
void OverloadFitness::diagnose(SignatureType signature,
                               const DirectCallable &callable,
                               ArrayRef<ASTExprAnd<AnyValue>> operands,
                               bool isMethodCall, LitDiagnostic &diag) {
  // TODO: Would be really nice to range underline the operand in question!
  switch (kind) {
  case kValid:
    diag << "candidate is viable";
    return;
  case kParamCount: {
    size_t actualNumBindings = callable.inputParamBindings.bindings.size();
    diag << "callee expects " << signature.getInputParams().size()
         << " input parameter" << plural(signature.getInputParams().size())
         << " but " << actualNumBindings
         << plural(actualNumBindings, " was", " were") << " provided";
    return;
  }
  case kParamWrongType: {
    auto decl = signature.getInputParams()[payload];
    auto binding = callable.inputParamBindings.bindings[payload];
    diag << "callee parameter " << decl.getName() << " has " << ASTType(type)
         << " type, but value has type " << ASTType(binding.getType())
         << binding.expr->getRange();
    return;
  }
  case kResultParamCount:
    diag << "callee expects " << signature.getResultParamTypes().size()
         << " result parameter"
         << plural(signature.getResultParamTypes().size()) << " but "
         << callable.resultParams.size()
         << plural(callable.resultParams.size(), " was", " were")
         << " provided";
    return;
  case kArgCount:
    diag << "callee expects " << payload << " argument" << plural(payload);
    return;
  case kArgNotLValue:
    if (isMethodCall && payload == 0) {
      diag << "invalid use of mutating method on rvalue of type "
           << ASTType(type) << operands[0].expr->getRange();
      return;
    }
    diag << "argument #" << payload
         << " must be mutable in order to pass as a by-ref argument"
         << operands[0].expr->getRange();
    return;
  case kArgWrongLVType:
    diag << "l-value of type " << operands[payload].ir.getRValueType()
         << " cannot be converted to reference to expected type "
         // TODO(QoI): Types are not attributes.  We are printing this as an
         // attr... not a sugared type.
         << cast<POP::PointerType>(Type(type)).getElementType()
         << operands[payload].expr->getRange();
    return;

  case kArgWrongType:
    // If this is a method syntax call, don't count the receiver.
    if (isMethodCall) {
      // it is probably possible for this assert to fire, if it does we should
      // tailor the error message.
      assert(payload != 0 && "TODO: unexpected self mismatch");
      diag << "method argument #" << (payload - 1);
    } else {
      diag << "argument #" << payload;
    }
    diag << " cannot be converted from " << operands[payload].ir.getRValueType()
         << " to " << type << operands[payload].expr->getRange();
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
      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = shared.emitError(nameLoc, "invalid call to '")
                    << baseName << "': ";
        evaluations[0].diagnose(fnDecl.getFullSignature(), *this, operands,
                                isMethodCall, diag);
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return failure();
      }

      // Otherwise emit an error, and a note for what is wrong with each
      // candidate.
      auto diag = shared.emitError(nameLoc, "no matching function in call to '")
                  << baseName << "': ";
      for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
        auto fnDecl = cast<LIT::FuncOp>(*candidate);
        diag.attachNote(fnDecl->getLoc()) << "candidate not viable: ";
        eval.diagnose(fnDecl.getFullSignature(), *this, operands, isMethodCall,
                      diag);
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
    auto diag = shared.emitError(nameLoc, "ambiguous call to '")
                << baseName << "', each candidate requires " << minConversions
                << " implicit conversion" << plural(minConversions)
                << ", disambiguate with an explicit cast";
    for (ASTDecl *candidate : newFnDecls)
      diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
          << "candidate declared here";
  }
  return failure();
}

/// Generate a reference to the specified function, checking that any supplied
/// parameters are correct and match expectations.
SymbolConstantAttr
DirectCallable::getBoundConstantAttr(LitSharedState &shared) const {
  if (fnDecls.size() != 1) {
    assert(!fnDecls.empty() && "DirectCallable malformed");
    auto diag =
        shared.emitError(
            nameLoc, "cannot form a reference to overloaded declaration of '")
        << baseName << "'";
    for (ASTDecl *candidate : fnDecls) {
      auto funcOp = cast<LIT::FuncOp>(*candidate);
      diag.attachNote(funcOp.getLoc()) << "candidate declared here";
    }

    return {};
  }

  auto funcOp = cast<LIT::FuncOp>(*fnDecls[0]);

  // Check that the signature can be rebound with our set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;

  auto newBindings = inputParamBindings.verifyBindings(
      funcOp.getFullSignature().getInputParams(), baseName, nameLoc,
      incorrectBindingNo, incorrectBindingExpectedType, shared,
      /*emit diagnostics*/ funcOp);
  if (!newBindings)
    return {};

  // Now that we checked the types match, form the binding.
  return funcOp.getBoundReference(newBindings);
}

/// Check declarations for the result parameters and add them to
/// resultParamDecls.  This emits and error and returns failure if an error is
/// detected.
LogicalResult DirectCallable::getResultParamDecls(
    SignatureType signature, SmallVectorImpl<ParamDeclAttr> &resultParamDecls,
    IREmitter &emitter) {
  assert(signature.getResultParamTypes().size() == resultParams.size() &&
         "We know that the callee is type checked");

  // If there is nothing to do, then we are done.
  if (resultParams.empty())
    return success();

  // Verify completion of forward declared alias declarations.  We know the
  // decl exists, but we don't know if the type is compatible or it has been
  // multiply defined.
  //
  // TODO: We don't remap input parameters types into output parameter types.
  // We surely handle this wrong: `fn x[a: type -> a]():` for example.
  for (auto [type, declAndLoc] :
       llvm::zip(signature.getResultParamTypes(), resultParams)) {
    auto forwardDecl = cast<AliasForwardDeclOp>(*declAndLoc.first);

    // Verify the types match.
    // TODO: Move this to overload resolution.
    if (!ASTType(forwardDecl.getType()).isEqualCanon(type)) {
      auto diag =
          emitter.emitError(declAndLoc.second, "result parameter returns type ")
          << type << " but forward declaration is of type "
          << ASTType(forwardDecl.getType());
      diag.attachNote(forwardDecl.getLoc()) << "alias forward declared here";
      return failure();
    }
    resultParamDecls.push_back(ParamDeclAttr::get(forwardDecl.getName(), type));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallableValue Implementation
//===----------------------------------------------------------------------===//

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
      shared.emitError(callLoc, "")
          << type << " does not implement the '" << methodName << "' method";
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
  direct =
      DirectCallable{callLoc, methodName, resultDecls, type.getParamBindings()};
}

/// Emit this as a flattened RValue or LValue with no additional parameter
/// context.  This returns null on failure.
AnyValue CallableValue::emitAsValue(IREmitter &emitter) const {
  // If we have no bound symbol, return the normal lvalue or rvalue we
  // represent.
  if (!direct)
    return baseVal.ir;

  auto directSymbolAttr = direct->getBoundConstantAttr(emitter.shared);
  if (!directSymbolAttr)
    return {};

  // Verify that the target has no result parameters.  We have no way to bind
  // these indirectly.
  SignatureType calleeSignature = directSymbolAttr.getType();
  if (!calleeSignature.getResultParamTypes().empty()) {
    emitter.emitError(direct->nameLoc,
                      "calls with result parameters must be called directly");
    return {};
  }

  // If we have no base value, then we are just a symbol, return it.
  if (!baseVal)
    return MValue(directSymbolAttr);

  auto loc = baseVal.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  Value firstArgValue;
  switch (calleeSignature.getInputConvention(0)) {
  case ValueInputConvention::ByRef: {
    LValue baseLV = baseVal.ir.getIfLValue();
    if (!baseLV) {
      emitter.emitError(loc,
                        "invalid use of mutating method on rvalue of type ")
          << ASTType(baseVal.ir.getType()) << baseVal.expr->getRange();
      return {};
    }
    firstArgValue = baseLV;

    // Using partial application over an lvalue isn't safe until we support an
    // ownership models with mutable borrows.
    emitter.emitError(loc, "TODO: partial application to mutable base isn't "
                           "supportable without a lifetime model")
        << baseVal.expr->getRange();
    return {};
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

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
bool CallableValue::canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                               ASTType requiredType,
                                               LitSharedState &shared) {
  // If it already matches, then we're done.
  if (value.ir.getRValueType().isEqualCanon(requiredType))
    return true;

  // Otherwise, check to see if we can do an implicit conversion by invoking a
  // `__new__` method on the expected type.
  bool isErroneousDecl = false;
  CallableValue callee(requiredType, "__new__", SMLoc(), isErroneousDecl,
                       shared);

  // If there are no viable candidates for the implicit conversion, we fail.
  if (!callee.direct)
    return false;

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.
  callee.direct->disableImplicitConversions = true;
  return succeeded(callee.direct->filterOverloadSet(
      {value}, /*isMethodSyntax*/ false,
      /*emitDiagnosticOnFailure=*/false, shared));
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
                                const ExprNode *callExpr, IREmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Set to true if this is a method call like `x.foo(...`.
  bool isMethodCall = false;
  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> operandsWithSelf;

  auto emitError = [&](const Twine &message) {
    return emitter.emitError(callExpr->getLoc(), message);
  };

  // Figure out the type of the function to call, which is either symbol or a
  // normal rvalue.
  SignatureType calleeSig;

  // If the callee defines parameters, this is the definitions to use.
  SmallVector<ParamDeclAttr> resultParamDecls;

  // This is the callee symbol constant for a direct call, or the SSA value for
  // an indirect call.
  AnyValue callee;
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
    callee = MValue(symbol);
    if (failed(
            direct->getResultParamDecls(calleeSig, resultParamDecls, emitter)))
      return {};

  } else {
    // Otherwise we have an indirect call. If the callee is an MValue, emit a
    // `call_param`. Otherwise, emit the callee value as a DRValue so we can
    // call it with call_indirect.
    callee = baseVal.ir.getIfMValue();
    if (!callee) {
      callee = emitter.emitDRValue(baseVal.ir, baseVal.expr->getLoc());
      if (!callee)
        return {};
    }

    calleeSig = dyn_cast<SignatureType>(callee.getType());
    if (!calleeSig) {
      emitError("invalid function type to call ")
          << ASTType(callee.getType()) << baseVal.expr->getRange();
      return {};
    }

    // Check to see if we can apply these operands to the callee signature.
    DirectCallable bindings{callExpr->getLoc(), "callee", /*params*/ {}, {}};
    auto fitness = OverloadFitness::evaluate(calleeSig, bindings, operands,
                                             emitter.shared);
    if (fitness.kind != OverloadFitness::kValid) {
      // If not, diagnose it with an error.
      auto diag = emitError("invalid indirect call: ");
      fitness.diagnose(calleeSig, bindings, operands, isMethodCall, diag);
      return {};
    }
  }

  assert(calleeSig.getResultParamTypes().size() == resultParamDecls.size() &&
         calleeSig.getValues().getNumInputs() == operands.size() &&
         "Type checking should be done");

  // Emit all the arguments.
  SmallVector<std::pair<AnyValue, llvm::SMLoc>> valueArguments;
  for (auto [argAnyValueAndExpr, expectedType, convention] :
       llvm::zip(operands, calleeSig.getValueInputs(),
                 calleeSig.getValueInputConventions())) {
    auto argLoc = argAnyValueAndExpr.expr->getLoc();
    // If the callee takes the operand as a by-ref argument, we require an
    // lvalue.
    AnyValue argVal;
    switch (convention) {
    case ValueInputConvention::ByRef:
      argVal = argAnyValueAndExpr.ir.getIfLValue();
      assert(argVal && "Call should already be type checked");
      break;
    case ValueInputConvention::ByVal:
      if (!emitter.builder) {
        argVal = argAnyValueAndExpr.ir;
        if (!argVal.getIfMValue()) {
          emitter.emitError(argAnyValueAndExpr.expr->getLoc(),
                            "cannot use a dynamic value in meta context")
              << argAnyValueAndExpr.expr->getRange();
          return {};
        }
      } else {
        argVal = emitter.emitDRValue(argAnyValueAndExpr.ir, argLoc);
      }

      // Convert the argument to the expected type if needed.
      argVal = emitter.getAsExpectedType(argVal, argAnyValueAndExpr.expr,
                                         expectedType, " in argument");
      if (!argVal)
        return {};
      break;
    }

    valueArguments.emplace_back(argVal, argLoc);
  }

  auto &builder = emitter.builder;
  if (!builder) {
    // Emitting a call in a meta context. Generate an apply operator.
    SmallVector<TypedAttr> operands({callee.getIfMValue().get()});
    for (auto [argVal, loc] : valueArguments)
      operands.push_back(argVal.getIfMValue().get());
    return MValue(ParamOperatorAttr::get(POC::Apply, operands));
  }

  // Otherwise, materialize MValue arguments as DRValues.
  SmallVector<Value> callArgs;
  for (auto [argVal, loc] : valueArguments) {
    if (auto lv = argVal.getIfLValue())
      callArgs.push_back(lv);
    else
      callArgs.push_back(emitter.emitDRValue(argVal, loc));
  }

  Operation *callOp;
  Location loc = emitter.translateLocation(callExpr->getLoc());
  ArrayRef<Type> resultTypes = calleeSig.getValueResults();
  if (auto target = callee.getIfMValue()) {
    // If the callee is a symbol constant, directly emit a call.
    if (auto symbol = dyn_cast<SymbolConstantAttr>(target.get())) {
      callOp = builder->create<CallOp>(loc, resultTypes, symbol,
                                       resultParamDecls, callArgs);
    } else {
      callOp = builder->create<CallParamOp>(loc, resultTypes, target.get(),
                                            resultParamDecls, callArgs);
    }
  } else {
    // Otherwise emit calls to SSA values with call_indirect.
    callOp = builder->create<POP::CallIndirectOp>(loc, resultTypes,
                                                  callee.getIfDRValue(),
                                                  /*operands*/ callArgs);
  }

  // If the callee can raise an error, try to unwrap it.
  Value resultVal = callOp->getResult(0);
  if (calleeSig.getFnEffects() == FnEffects::Throws) {
    if (!isValidErrorContext(builder->getInsertionBlock())) {
      emitError("cannot call function that may raise in a context that "
                "cannot raise");
      return {};
    }
    resultVal = builder->create<UnwrapOrPropagateOp>(
        loc, cast<POP::VariantType>(resultVal.getType()).getType(1), resultVal);
  }

  // Value returning call returns its result.
  return DRValue(resultVal);
}
