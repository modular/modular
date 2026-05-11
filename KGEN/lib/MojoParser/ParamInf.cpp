//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParamInf.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamBindings.h"
#include "ParamMatcher.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// File-local utils
//===----------------------------------------------------------------------===//

static Type inferInitializerType(ASTDecl &declScope, InitializerUValue &init,
                                 ASTExprAnd<AnyValue> operand,
                                 ASTType defaultType) {
  IREmitter emitter(declScope, EC_CallArgValue);
  if (!defaultType)
    return {};
  ASTType inferredType =
      defaultType.getWithUnknownParametersReplaced(declScope.getShared());

  CallOperands operands =
      init.getOperandsForInferredType(inferredType, EC_CallArgValue, emitter);

  // We expect the initializer to return the constructed type.
  // Infer the parameters of this overload candidate against the computed
  // result type of the initializer.
  FailureOr<PValue> initFn =
      OverloadSet::canConstructType(inferredType, operands, declScope);
  if (failed(initFn) || !initFn.value())
    return {};
  return FnOrFnLiteralTypeGeneratorType::get(initFn.value().getType())
      .getUserResultType();
}

/// Try to infer the type of an initializer list/dict/set/slice literal by
/// first binding it to `preferred` and, on failure, to the literal's default
/// type (e.g. `List[Int]` for a list literal).  Returns a null `Type` if
/// neither binding succeeds.
static Type tryInferInitializerType(ASTDecl &declScope, InitializerUValue &init,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType preferred) {
  if (Type result = inferInitializerType(declScope, init, operand, preferred))
    return result;
  return inferInitializerType(declScope, init, operand,
                              init.getDefaultType(declScope.getShared()));
}

//===----------------------------------------------------------------------===//
// ParameterInference
//===----------------------------------------------------------------------===//

ParamInf::ParamInf(const ParamBindings &paramBinding,
                   ArrayRef<Type> declaredParamTypes,
                   PogListAttr declaredParamPogs, bool allowImplicitConversions,
                   ASTDecl *declIfDirect, bool discardError)
    : InferenceState(paramBinding.declScope, declaredParamTypes,
                     declaredParamPogs, paramBinding.getExprLoc(),
                     discardError),
      paramBindings(paramBinding), declIfKnown(declIfDirect),
      explicitlyUnboundParams(declaredParamTypes.size(), false),
      allowImplicitConversions(allowImplicitConversions) {}

// TODO: Reconsolidate this.
namespace M::KGEN::LIT {
void printUValueTypeInfo(const AnyValue &value, MojoInflightDiag &diag);
void emitWrongTypeDiag(MojoInflightDiag &diag, ASTExprAnd<AnyValue> operand,
                       ASTType expectedType, size_t argIdx,
                       PogListAttr argListAttr, CallSyntax syntax,
                       SharedState &shared);
} // namespace M::KGEN::LIT

/// Attempt to resolve the specified operand to a CValue using the provided
/// type, checking whether any UValue's are compatible with the type and
/// inferring any parameters from it.  This emits a diagnostic and returns null
/// on failure.  On success, this makes an attempt to return a CValue, but won't
/// do so if that would require generating dynamic logic (e.g. creating an
/// instance of a value due to an initializer list).  In that case it returns
/// the inferred type of the result.
FailureOr<SmartVariant<CValue, ASTType>>
ParamInf::inferCValue(ASTExprAnd<AnyValue> operand, size_t argIdx,
                      PogListAttr argPogs, CallSyntax syntax,
                      ASTType expectedType) {
  // If this is already a CValue then we're done.
  if (auto cv = operand.ir.getIfCValue())
    return SmartVariant<CValue, ASTType>(cv);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getMojoDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, evaluator.getReboundType(expectedType),
                        argIdx, argPogs, syntax, getShared());
    return diag;
  };

  // Check to see if the expected type has an initializer with the
  // specified operands.  Remove any parameters from the expected type
  // since those are what we're inferring from the arguments.  The result
  // 'actualType' will have those newly inferred parameters.
  if (auto initValue = operand.ir.getIfInitializer()) {
    // Try binding the literal to `expectedType` (e.g. `List[$0]` becomes
    // `List[?]` so the unbound parameter is inferred), then fall back to the
    // literal's default type.
    ASTType initType = tryInferInitializerType(getDeclScope(), *initValue,
                                               operand, expectedType);

    // If there were declaration errors, assume success to not raise
    // spurious errors due to not resolving to those erroneous
    // declarations.
    if (!initType) { // TODO: Could improve this error to talk about inits.
      emitWrongTypeDiag(expectedType);
      return failure();
    }

    // If we're in a parameter binding expression, we can just emit the value as
    // a PValue and return it.  This is more powerful than the logic below,
    // because it allows implicit conversions, e.g. when we default a list
    // literal like [1, 2] to List[Int], it supports implicit conversion to
    // Span[Int, _].  The logic below does not support this.
    if (syntax == CallSyntax::kParamBindings) {
      IREmitter emitter(getDeclScope(), ExprContext::EC_ParameterList);
      auto value = emitter.emitPValue(operand, EC_ParameterList, initType);
      if (!value)
        return failure();
      return SmartVariant<CValue, ASTType>(value);
    }

    ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);

    // If we found one, we resolve our value to the inferred type.
    if (succeeded(matcher.matchTypes(initType, expectedType)))
      return SmartVariant<CValue, ASTType>(initType);

    // TODO: Could improve this to talk about initializers.
    auto &diag = emitWrongTypeDiag(expectedType);
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  auto orValue = operand.ir.getIfOverloadSet();
  assert(orValue && "Unknown UValue!");

  // If we have a reference to an overloaded method like foo(a.method),
  // then we can't resolve it.
  // TODO(partial application => closures): Given we just resolved argVal,
  // we could form the "a.method" expression with a closure.
  if (orValue->baseValue) { // Cannot merge base value.
    emitWrongTypeDiag(expectedType);
    return failure(); // TODO: Improve this.
  }

  // If the overload set has a single entry, just get it.
  if (auto pv = orValue->getIfPValue())
    return SmartVariant<CValue, ASTType>(CValue(pv));

  // If the expected type is concrete, then we can filter the overload set down
  // to a single entry and emit errors if not.
  if (!paramFinder.hasReferences(expectedType)) {
    auto emitError = [&](SMLoc loc) -> MojoInflightDiag & {
      return getMojoDiag(loc);
    };

    auto [argVal, _] = orValue->filterOverloadSetForValueType(
        expectedType, getDeclScope(), emitError);
    if (!argVal)
      return failure();
    return SmartVariant<CValue, ASTType>(CValue(argVal));
  }

  // FIXME: This emits an error unconditionally (not to getDiags) on failure.
  if (PValue result = orValue->filterOverloadSetForParamBindings())
    return SmartVariant<CValue, ASTType>(CValue(result));

  // Otherwise, we don't have a contextual error.
  emitWrongTypeDiag(expectedType);
  return failure(); // TODO: Improve this.
}

/// Core type matching logic for parameter inference, handling the expected
/// type without convention-specific processing. This function is called after
/// the expected type has been adjusted for calling conventions.
///
/// NOTE: This function performs parameter inference and error reporting,
/// while 'OverloadFitness::scoreOperandFitness' computes fitness metrics
/// after inference is complete. They serve different phases of overload
/// resolution and should remain separate.
LogicalResult ParamInf::inferFromRVType(ASTExprAnd<AnyValue> operand,
                                        size_t argIdx, ASTType expectedType,
                                        PogListAttr argPogs,
                                        CallSyntax syntax) {
  // Make sure the diagnostic machinery knows about our getDeclScope() so
  // parameter names get emitted correctly.
  DeclResolver::DiagnosticDeclContextChanger x(declIfKnown);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getMojoDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, evaluator.getReboundType(expectedType),
                        argIdx, argPogs, syntax, getShared());
    return diag;
  };

  expectedType = evaluator.getReboundType(expectedType);

  // Okay, we got a normal value argument convention and stripped off any
  // ArgConvention-related !lit.ref from the expected type.  See if we can
  // resolve the argument to a CValue.
  FailureOr<SmartVariant<CValue, ASTType>> argValOr =
      inferCValue(operand, argIdx, argPogs, syntax, expectedType);
  if (failed(argValOr))
    return failure();
  CValue argVal = dyn_cast<CValue>(*argValOr);
  if (!argVal) // Already checked the type is ok.
    return success();

  // If the argument types exactly match, then they are good.
  ASTType argType = argVal.getRValueType();
  if (argType.isEqualCanon(expectedType))
    return success();

  // We have a non-parametric expected type, and a wildcard type, we can
  // match any operand.
  if (sugarIsa<NameLookupArgWildcardType>(argType) &&
      !paramFinder.hasReferences(expectedType))
    return success();

  // TODO: Optionally compute fitness metrics (# implicit conversions,
  // convention mismatches) during inference, so they don't need to be
  // recomputed by scoreOperandFitness() afterward.
  ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);

  ParamMatcher::FailableScope simpleEqualityFailableScope(matcher);
  if (succeeded(matcher.matchTypes(argType, expectedType)))
    return success(); // Types were equal after matching.

  // Save the failure code and the bindings that were inferred so we can
  // restore them if the other attempts fail.
  auto savedFailureInfo = simpleEqualityFailableScope.saveState();
  simpleEqualityFailableScope.revert();

  // Handle values of nonmaterializable types.  These freely convert to their
  // nonmaterializable target type: even when implicit conversions are disabled.
  // We can accept this argument if that converted type is compatible with
  // our expected type.
  if (auto nonmaterializableTarget =
          argType.getNonmaterializableTarget(getShared())) {
    ParamMatcher::FailableScope failableScope(matcher);
    // Infer the parameters of this overload candidate against the computed
    // result type of the initializer.
    if (succeeded(matcher.matchTypes(nonmaterializableTarget, expectedType))) {
#if 0
      // Implicit conversion for nonmaterializable types to their target
      // type is allowed even if !allowImplicitConversions and count as half
      // as much of a mismatch as a normal implicit conversion.  This enables
      // exact matches to be more specific, and literals to be more compatible
      // than an actual conversion.
      ++numImplicitConversions;
#endif
      return success();
    }

    // Roll back any error and inferred bindings.
    failableScope.revert();
  }

  // If implicit conversions are enabled and the target type is known, then
  // we can check to see if any of the constructors for the result type can
  // work.  If disabled, then we have a failure.
  if (!allowImplicitConversions) {
    // Restore the information from the original failure so we have a simple
    // diagnostic.
    ParamMatcher::FailableScope::restore(savedFailureInfo, matcher);
    auto &diag = emitWrongTypeDiag(expectedType);
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  // If we had one, this bumps our # implicit conversions.
  numImplicitConversions += 2;

  // If the expected type has been fully resolved, check it for implicit
  // conversions using the normal type machinery.  This will handle things like
  // function pointer conversions that the code below doesn't.
  if (!paramFinder.hasReferences(expectedType)) {
    if (IREmitter::canImplicitlyConvertToType({argVal, operand.expr},
                                              expectedType, getDeclScope())) {
      return success();
    }

    // Restore the information from the original failure so we have a simple
    // diagnostic.
    ParamMatcher::FailableScope::restore(savedFailureInfo, matcher);
    auto &diag = emitWrongTypeDiag(expectedType);
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  /// When checking if an implicit conversion is possible, apply the bindings
  /// inferred so far (plus a distinct new attribute relating back to the
  /// original decls for ones that are missing) to the signature with
  /// getSpecializedSignature so we benefit from the already-fixed substitutions
  /// being applied to the input types.  This can make them more concrete and
  /// help with inferring dependent types based on already-bound parameters.  If
  /// we inferred a value for the parameter from previous arguments, substitute
  /// it into the expected types of subsequent arguments.  This allows us to
  /// handle dependent argument types like:
  ///     def foo[dt: DType](p: UnsafePointer[Scalar[dt]], v:
  ///     Scalar[p.type.type]):
  /// where the type of 'v' depends on 'dt' being inferred.

  // Determine if we can construct the requested type given the existing value
  // we have.  If so, get the type inferred signature of the init method that
  // would make it work.
  if (sugarIsa<StructType>(expectedType)) {
    // The expected type may be parameterized, and that type may both have
    // parameters that we are trying to infer as well as parameters that are
    // already known.  For example, if expectedType is known to be
    // 'SIMD[uint8, 1]', then we can infer which constructor to use when the
    // input is an IntLiteral.
    //
    // On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
    // argument is an Int8, then we need the implicit conversion to infer the
    // base element.  Our solution to this is to rip and replace parameters that
    // contain unbound parameters, replacing them with UnboundAttr so inference
    // can find them.
    auto nonParamType =
        expectedType.getWithUnknownParametersReplaced(getShared());
    CallOperands ctorOperands(CallSyntax::kImplicitConvert, operand.expr,
                              EC_TypeParamValue, {{argVal, operand.expr}});
    FailureOr<PValue> pValue = OverloadSet::canConstructType(
        nonParamType, ctorOperands, getDeclScope());
    if (failed(pValue)) {
      auto &diag = getMojoDiag(operand.expr->getLoc());
      diag << "cannot convert to type with a previously diagnosed error";
      return failure();
    }

    // If we found one, we succeed if the returned type is compatible with the
    // expected type.  Infer the parameters of this overload candidate against
    // the computed result type of the initializer.
    if (auto callee = pValue.value()) {
      auto initSig = FnOrFnLiteralTypeGeneratorType::get(callee.getType());
      ParamMatcher::FailableScope failableScope(matcher);
      if (succeeded(
              matcher.matchTypes(initSig.getUserResultType(), expectedType))) {
        return success();
      }
      failableScope.revert();
    }
  }

  // Otherwise, none of that worked. We aren't sure what to do here - it could
  // be any of these things, so we need to emit an error.  If out failure is
  // due to an uninferred parameter, and if that parameter had a default, then
  // we can bind it.
  if (savedFailureInfo.first.getIfDependentOnUnresolved()) {
    // If we're in the parameter binding list *for a call* then we can
    // re-evaluate this binding after the arguments of the call are resolved.
    //
    // For struct binding, we enforce strict left-to-right order.
    //
    // FIXME(MOCO-3300): for call binding, we need to resolve deferred parameter
    // binding before default as well (IT IS NOT THE CASE AT THE MOMENT).
    if (syntax == CallSyntax::kParamBindings && !isInferForStruct) {
      hasDeferredGivenParam = true;
      return success();
    }

    // At this point, if we still have an unresolvable dependent type, give it
    // one last shot and try to pull default parameter value
    //
    // def store[
    //     dtype: DType
    //     width: Int = 1,
    // ](
    //     self: UnsafePointer[Scalar[dtype], ...],
    //     val: SIMD[dtype, width],
    // )
    //
    // # here Int(8) need to be implicitly converted to SIMD[dtype, 1],
    // store(ptr, Int(8))
    //
    // Otherwise, check to see if this is due to an uninferred param with a
    // default value.  If so, bind the default and try again.
    size_t paramIdx =
        savedFailureInfo.first.getIfDependentOnUnresolved().value();
    if (auto value = declaredParamPogs.getDefault(paramIdx)) {
      assert(!evaluator.getIndexBindings()[paramIdx] &&
             "shouldn't have inferred this if we failed because of it");
      value = evaluator.getReboundAttribute(value);
      auto result = setInferredValue(paramIdx, value);
      assert(!failed(result) && "should always succeed");
      (void)result;
      if (failed(
              inferFromRVType(operand, argIdx, expectedType, argPogs, syntax)))
        return failure();

      TypedAttr newValue =
          evaluator.getReboundAttribute(evaluator.getIndexBindings()[paramIdx]);
      if (newValue != value) {
        result = setInferredValue(paramIdx, value);
        assert(!failed(result) && "should always succeed");
        (void)result;
      }
      return success();
    }
  }

  // Restore the information from the original failure so we have a simple
  // diagnostic.
  ParamMatcher::FailableScope::restore(savedFailureInfo, matcher);

  if (matcher.failureReason->isUnboundButInferrable()) {
    // Be more specific about this case.
    auto &diag = getMojoDiag(operand.expr->getLoc());
    diag << "failed to infer from type " << argType;
    matcher.failureReason->addExplanation(diag);
    return failure();
  }

  auto &diag = emitWrongTypeDiag(expectedType);
  matcher.failureReason->addExplanation(diag);
  return failure();
}

/// Infer and emit a single value for a parameter binding. This returns
/// failure if it emits a diagnostic, otherwise is returns a parameter value
/// if resolved, or null if deferred.
FailureOr<TypedAttr>
ParamInf::inferAndEmitOneParam(ASTExprAnd<AnyValue> binding,
                               ASTType expectedType, size_t paramIdx) {
  IREmitter emitter(getDeclScope(), EC_ParameterList);

  // We don't typecheck the '_' magic parameter, we propagate it.
  //
  // NOTE: we have to return a `_` here to mark the parameter has been
  // explicitly unbound instead of `nullptr` (maybe unless we know this is not a
  // partial binding?). Consider the following cases
  //
  // struct T[a : Int = 1] : pass
  // comptime T1 = T[_]
  // comptime T2 = T[]
  //
  // if we return nullptr here, ParamInf can not distinguish between T1 and T2,
  // and in both cases, `a` will be bound with the default value.
  //
  // NOTE: in a non-partial binding context, `_` can be also used as a place
  // holder, in this case we don't infer it to `_`
  if (isa_and_nonnull<UnboundAttr>(binding.ir.getIfPValue().get())) {
    // `_` means different things when used in struct binding or call bindings,
    // for struct binding, it is a concrete unknown value; for call binding, it
    // means something to be inferred.
    // We can not simply return and bind a `_` value here either, because it
    // could be dependent by other parameters/default values, which need to be
    // handled properly.
    if (isInferForStruct)
      explicitlyUnboundParams.set(paramIdx);

    return TypedAttr(); // Deferred
  }

  // If the expected type has unresolved bindings, try to infer them from the
  // argument.  This is a non-trivial operation because we support inferring
  // from the value directly, but also inferring as a result of implicit
  // conversions.
  if (paramFinder.hasReferences(expectedType)) {
    if (failed(inferFromRVType(binding, paramIdx, expectedType,
                               declaredParamPogs, CallSyntax::kParamBindings)))
      return failure();
  }

  // We might have inferred more parameter after `inferOneOperand`.
  expectedType = evaluator.getReboundType(expectedType);

  if (paramFinder.hasReferences(expectedType)) {
    hasDeferredGivenParam = true;
    return TypedAttr(); // Deferred.
  }

  // If we have a UValue or something else, convert it to a PValue now that we
  // know the expected type.
  TypedAttr bindingVal = binding.ir.getIfPValue();
  if (!bindingVal) {
    FailureOr<SmartVariant<CValue, ASTType>> cvOr =
        inferCValue(binding, paramIdx, declaredParamPogs,
                    CallSyntax::kParamBindings, expectedType);
    if (failed(cvOr))
      return failure();

    CValue argVal = dyn_cast<CValue>(*cvOr);
    // If we had an initializer list this will succeed but not actually create
    // the instance because the logic is shared with the dynamic argument
    // checking logic that can't create an instance.  We don't have that problem
    // so just do it.  We need to use the returned type as the expected type
    // because the expected type might be something like Span, and the inferred
    // type might be List (e.g. as the default type for a list literal).
    if (!argVal) {
      argVal =
          emitter.emitPValue(binding, EC_ParameterList, cast<ASTType>(*cvOr));
      assert(argVal && "This should always succeed; it was checked");
    }

    // Finally, check that this CValue is a PValue.
    bindingVal = argVal.getIfPValue();
    if (!bindingVal) {
      getMojoDiag(binding.expr->getLoc())
          << "cannot use a dynamic value in a parameter list"
          << binding.expr->getRange();
      return failure();
    }
  }

  // Reject invalid *'s, varargs will have been already handled.
  if (sugarIsa<UnpackedAttr>(bindingVal)) {
    getMojoDiag(binding.expr->getLoc())
        << "invalid unpack in non-variadic parameter binding"
        << binding.expr->getRange();
    return failure();
  }

  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  if (expectedType.isEqualCanon(bindingVal.getType()))
    // Align sugar if necessary.
    return ParamOperatorAttr::getRebind(bindingVal, expectedType);

  // If the parameter can be implicitly converted, do so.
  if (IREmitter::canImplicitlyConvertToType(
          {bindingVal, binding.expr}, expectedType, emitter.getDeclScope())) {
    return emitter
        .emitPValue({bindingVal, binding.expr}, EC_ParameterList, expectedType)
        .get();
  }

  // Otherwise, the parameter is simply the wrong type, emit an error about this
  // problem.
  DeclResolver::DiagnosticDeclContextChanger x(&(getDeclScope()));
  MojoInflightDiag &diag = getMojoDiag({});
  if (declIfKnown) // Why only structs? Seems arbitrary, push higher?
    diag << "'" << *declIfKnown->getUserNameIfOperation() << "' ";
  diag << "parameter "
       << ParamDeclRefAttr::get(declaredParamPogs.getName(paramIdx),
                                declaredParamTypes[paramIdx])
       << " has " << expectedType << " type, but value has type "
       << bindingVal.getType() << binding.expr->getRange();

  return failure();
}

/// Infer all of the parameters we can from 'givenBindings'.
///
/// The 'partial' field specifies this is
/// performing a partial binding - e.g. because this is not a full type
/// binding, or because more params can be inferred from arguments to the
/// call.
///
/// On failure, this will emit a diagnostic through the 'getDiag' callback.
LogicalResult ParamInf::inferFromParamList() {
  bool hasEllipsis = paramBindings.bindingKind == ParamBindings::kWithEllipsis;

  // Use the temporary operands list if we had to remove an ellipsis, otherwise
  // use the original operands list.
  const CallOperands &givenBindings = this->getGivenBindings();

  // Do basic validation of the argument list using shared logic.
  // TODO: Integrate this into the logic below.
  // FIXME: why the verification here does not guarantee there is no parameter
  // number mismatch/missing kw error below?
  OperandValueList variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] = givenBindings.diagnoseKeywordOperands(
      declaredParamPogs, variadicKwOperands, /*allowMissingKwOnly=*/true);
  if (kwDiagRes != CallOperands::KwDiagResult::kValid) {
    MojoInflightDiag &diag = getMojoDiag({});
    switch (kwDiagRes) {
    case CallOperands::KwDiagResult::kMissingKwOnly:
      emitMissing(diag, kwDiagNames, "keyword-only parameter");
      break;
    case CallOperands::KwDiagResult::kOutOfOrderInferredKw: {
      size_t numNames = kwDiagNames.size();
      diag << "inferred parameter" << plural(numNames)
           << " passed out of order: ";
      llvm::interleave(
          kwDiagNames, [&](StringAttr str) { diag << str; },
          [&]() { diag << ", "; });
      break;
    }
    case CallOperands::KwDiagResult::kPosOnlyPassedByKw:
      emitPosOnlyPassedByKw(diag, kwDiagNames, "parameter");
      break;
    case CallOperands::KwDiagResult::kUnknownKeywords:
      emitUnknownKeywords(diag, kwDiagNames, "parameter");
      break;
    default:
      llvm_unreachable("unknown KwDiagResult");
    }
    return failure();
  }

  auto [posDiagRes, posDiagNames] = givenBindings.diagnosePosOperands(
      declaredParamPogs, /*allowCountMismatch=*/true);
  if (posDiagRes == CallOperands::PosDiagResult::kByPosAndKw) {
    emitByPosAndKw(getMojoDiag({}), posDiagNames, "parameter");
    return failure();
  }

  // Parameter inference and call emission rely on this function not failing
  // early due to missing or too many positional parameters.
  assert(posDiagRes == CallOperands::PosDiagResult::kValid &&
         "positional parameter operand check failed unexpectedly");

  // We may have pre-checked and out-of-order inferred parameters.  Avoid
  // stomping on them.
  auto applyBinding = [&](size_t idx, TypedAttr paramVal) -> LogicalResult {
    // Ignore this if the parameter value is deferred.
    if (!paramVal)
      return success();

    auto existing = evaluator.getIndexBindings()[idx];
    if (!existing)
      return setInferredValue(idx, paramVal);

    assert(isEqualCanon(existing, paramVal) &&
           "inferred to different values but didn't notice");

    return success();
  };

  size_t posIdx = 0, numParams = givenBindings.size();
  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = evaluator.getReboundType(declaredParamTypes[idx]);

    // Skip over any provided keyword parameters when matching things up, we
    // handle them separately below.
    while (posIdx < numParams && givenBindings[posIdx].keyword)
      ++posIdx;

    // If we have a varargs parameters, then it will eat the rest of the
    // parameters, but we have to check each of them.
    if (declaredParamPogs.isPosVarArg(idx)) {
      // If there are no parameter values, then leave the parameter uninferred
      // for now.  It could be inferred from an call-argument or be left
      // unbound.
      if (posIdx == numParams)
        continue;

      // Unpacked variadics (`Tuple[*elts]` where elts is a variadic list) can
      // be passed directly as a whole variadic parameter.
      auto [varArgsEltType, expectedValueList] =
          ASTType(expectedType).getParameterListInfo();
      if (auto unpacked = dyn_cast_or_null<UnpackedAttr>(
              givenBindings[posIdx].ir.getIfPValue().get())) {
        // FIXME: Make sure to only unpack *x in pos varargs and **x in kw
        // varargs.
        FailureOr<TypedAttr> paramVal = inferAndEmitOneParam(
            {unpacked.getValue(), givenBindings[posIdx].expr}, expectedType,
            idx);
        // Exit if an error was already emitted.
        if (failed(paramVal) || failed(applyBinding(idx, *paramVal)))
          return failure();
        ++posIdx;
        continue;
      }

      // Otherwise, we infer the variadic to be the elements of the variadic
      // list being passed in.
      SmallVector<TypedAttr> elements;
      bool isDeferred = false;
      while (posIdx != numParams) {
        // This pass just skips keyword parameters, they are handled later.
        if (givenBindings[posIdx].keyword) {
          ++posIdx;
          continue;
        }

        // Passing `_` to a variadic is not allowed. Users should pass `*_` to
        // unbind a variadic parameter.
        if (isa_and_nonnull<UnboundAttr>(
                givenBindings[posIdx].ir.getIfPValue().get())) {
          auto &diag = getMojoDiag(givenBindings[posIdx].expr->getLoc());
          diag << "unbound syntax (i.e. `_`) cannot be passed as a variadic "
                  "parameter";
          return failure();
        }

        // FIXME: pack and install variadics parameter correctly.
        FailureOr<TypedAttr> paramVal =
            inferAndEmitOneParam(givenBindings[posIdx], varArgsEltType, idx);
        if (failed(paramVal)) // Exit if an error was already emitted.
          return failure();

        ++posIdx;
        if (!*paramVal) {
          isDeferred = true;
          continue;
        }

        varArgsEltType = evaluator.getReboundType(varArgsEltType);
        // Realign sugar.
        if (paramVal->getType() != varArgsEltType)
          paramVal = ParamOperatorAttr::getRebind(*paramVal, varArgsEltType);
        elements.push_back(*paramVal);
      }

      if (!isDeferred) {
        // Infer the values list to the elements.
        auto vaType = evaluator.getReboundType(expectedValueList.getType());
        auto paramVA =
            ParamListAttr::get(elements, cast<ParamListType>(vaType));
        ParamMatcher matcher(getGivenBindings().callExpr, *this,
                             /*implConversions*/ false);
        if (failed(matcher.matchParams(paramVA, expectedValueList)))
          return failure();
        // The ParameterList now has a concrete type.
        auto listValue =
            UnknownAttr::get(evaluator.getReboundType(expectedType));
        if (failed(applyBinding(idx, listValue)))
          return failure();
      }
      continue;
    }

    // If we have a non-kw param value, it binds to this parameter if it accepts
    // it.
    if (posIdx < numParams && (pog.getPassingKind() == PassingKind::PosOrKw ||
                               pog.getPassingKind() == PassingKind::PosOnly)) {
      FailureOr<TypedAttr> paramVal =
          inferAndEmitOneParam(givenBindings[posIdx], expectedType, idx);
      // Exit if an error was already emitted.
      if (failed(paramVal) || failed(applyBinding(idx, *paramVal)))
        return failure();
      ++posIdx;
      continue;
    }

    // If we're out of positional bindings, or this works with a keyword, try
    // looking for a provided keyword parameter binding.
    if ((pog.getPassingKind() != PassingKind::PosOnly &&
         pog.getPassingKind() != PassingKind::Implicit)) {
      if (const OperandValue *param =
              givenBindings.findKwArg(declaredParamPogs.getName(idx))) {

        FailureOr<TypedAttr> paramVal =
            inferAndEmitOneParam(*param, expectedType, idx);
        // Exit if an error was already emitted.
        if (failed(paramVal) || failed(applyBinding(idx, *paramVal)))
          return failure();
        continue;
      }
    }

    // If this parameter is unspecified but we have a ... in the parameter list,
    // leave it unbound even if it has a default.
    if (hasEllipsis)
      continue;
  }

  // Check and complain if we have bindings that didn't get used.
  // FIXME: why do we still need this? should it has already been verified
  // above?
  if (posIdx != numParams) {
    // Hide the implicit trait parameter from the diagnostic.
    size_t hidden = 0;
    if (declIfKnown)
      if (auto fn = dyn_cast<FnOp>(declIfKnown->getIfOperation()))
        hidden = isa_and_nonnull<TraitDeclOp>(fn->getParentOp());

    size_t numExpected = countNumPositional(declaredParamPogs) - hidden;
    auto &diag = getMojoDiag({});
    if (declIfKnown)
      diag << "'" << *declIfKnown->getUserNameIfOperation() << "'";
    else
      diag << "parametric value";
    emitWrongArgOrParamCount(diag, /*minRequired=*/numExpected,
                             /*maxAllowed=*/numExpected,
                             givenBindings.getNumPositional() - hidden,
                             "positional parameter");
    return failure();
  }

  return success();
}

ParameterExprArrayAttr ParamInf::inferForStruct(bool emitConstraintFailure) {
  CrashReporter handler(paramBindings.getExprLoc(), "ParamInf::inferForStruct",
                        getShared());

  auto attachNoteOnError = llvm::scope_exit([&]() {
    if (diag.hasErrorEmitted() && declIfKnown) {
      if (llvm::isa_and_nonnull<FnOp>(declIfKnown->getIfOperation())) {
        diag.attachNote(declIfKnown->getLoc()) << "function declared here";
      } else {
        diag.attachNote(declIfKnown->getLoc())
            << "'" << *declIfKnown->getUserNameIfOperation()
            << "' declared here";
      }
    }

    if (emitConstraintFailure && !unprovableConstraints.empty()) {
      emitUnprovableConstraintsFromFitness(
          unprovableConstraints, paramBindings.shared,
          paramBindings.getExprLoc(), declIfKnown);
    }
  });

  isInferForStruct = true;

  if (failed(inferFromParamList()))
    return nullptr;

  if (paramBindings.bindingKind != ParamBindings::kWithEllipsis &&
      failed(inferFromDefaults())) {
    return nullptr;
  }

  if (failed(finalizeWithUnbound()))
    return nullptr;

  return getInferredValues();
}

// Infer any missing parameter from defaulted value (this is supposed to be
// invoked after both parameter list and argument list has been scanned).
LogicalResult ParamInf::inferFromDefaults() {

  auto setDefault = [&](TypedAttr value, size_t idx) -> LogicalResult {
    // The default value is explicitly unbound.
    if (explicitlyUnboundParams[idx])
      return success();

    value = evaluator.getReboundAttribute(value);

    // Don't try to infer from default values that have unresolved references
    // to other parameters.
    //
    // TODO: If the references points to a `_` parameter, we might still want to
    // install it without erasing the index reference.
    if (paramFinder.hasReferences(value))
      return success();

    auto argType = evaluator.getReboundType(declaredParamTypes[idx]);
    FailureOr<TypedAttr> paramVal = inferAndEmitOneParam(
        {value, getGivenBindings().getExpr()}, argType, idx);
    if (failed(paramVal))
      return failure();
    if (*paramVal && !evaluator.getIndexBindings()[idx])
      return setInferredValue(idx, *paramVal, /*isDefaulted=*/true);
    return success();
  };

  // Lastly, See if we can fulfill any missing parameters with default values
  // for their type (variadic attr always have a default empty value if not
  // inferable).
  for (size_t idx = 0, e = declaredParamTypes.size(); idx != e; ++idx) {
    if (evaluator.getIndexBindings()[idx])
      continue;

    // If available, we use a default parameter value.
    if (TypedAttr defaultParam = declaredParamPogs.getDefault(idx)) {
      // Default parameter values may reference other parameter values, so we
      // need to evaluate these.
      // If the default value is dependent, and we can not fully resolve all its
      // dependencies, do not try to set the value of it.
      if (failed(setDefault(defaultParam, idx)))
        return failure();
      continue;
    }

    // FIXME: this need a more systematical fix.
    // Determine if we can use a default parameter for CTAD
    if (paramBindings.ctadPogs.size() > idx) {
      if (TypedAttr defaultCTAD =
              paramBindings.ctadPogs[idx].getDefaultValue()) {
        defaultCTAD = evaluator.getReboundAttribute(defaultCTAD);
        if (!paramFinder.hasReferences(defaultCTAD)) {
          if (failed(setDefault(defaultCTAD, idx)))
            return failure();
          continue;
        }
      }
    }

    // TODO: move the special handling of Origin outside the default parameter
    // inference.
    if (isInferForStruct)
      continue;

    // Otherwise, check to see if this is an singleton parameter like Origin. So
    // long as its type is fully resolved, we can go ahead and instantiate it.
    if (auto paramType =
            sugarDynCast<LIT::StructType>(declaredParamTypes[idx])) {
      if (paramType.getSymbol().getLeafReference().strref() == "Origin" &&
          paramType.getParamValues().size() == 2 &&
          isa<OriginType>(paramType.getParamValues()[1].getType())) {
        IREmitter emitter(getDeclScope(), EC_TypeParamValue);

        paramType = cast<LIT::StructType>(evaluator.getReboundType(paramType));
        auto origin = // Get the Origin value.
            evaluator.getReboundAttribute(paramType.getParamValues()[1]);

        // If the !lit.origin is unbound, then we have a partial binding - don't
        // bind a concrete Origin around an unbound Origin, just let other
        // things leave it unbound also.  We don't want things like
        // Span[mut=False] to bind the Origin.
        if (isa<UnboundAttr>(origin))
          continue;

        TypedAttr paramVal =
            emitter.getStdlibOriginOf(origin, getDeclScope().getLoc());
        if (failed(setInferredValue(idx, paramVal)))
          return failure();
        continue;
      }
    }
  }

  // Do another pass to fill in empty variadic, we need to do it after user
  // provided default value is installed, the variadic might be dependent by
  // those value in cases like:
  //
  // struct HasParamList[*values: Int]:
  //     def __init__(out self):
  //         pass
  //
  // struct HasDefaultParam[strides: HasParamList[...] = HasParamList[4]()]:
  //     pass
  for (size_t idx = 0, e = declaredParamTypes.size(); idx != e; ++idx) {
    if (evaluator.getIndexBindings()[idx])
      continue;

    // If not specified/inferrable, variadic always have a default empty value.
    bool isInferableVA = [&]() -> bool {
      auto pog = declaredParamPogs.getPogs()[idx];
      // Since we reached this point, the parameter binding can not have `...`,
      // and according to the rules. It must be producing the most concrete
      // type. So, if this is a positional variadic, we always default it to
      // empty.
      if (pog.isPosVarArg())
        return true;

      // Parameters from an enclosing struct are smashed onto the beginning of
      // method parameter lists, and their types are switched to Inferred. As
      // part of that, we lose track of whether it was pos_var_arg.
      // E.g.,
      //
      // struct S[*values: Int]:
      //     @staticmethod
      //     def foo():
      //         pass
      //
      // # we should be able to infer *value to empty.
      // S.foo()
      //
      // FIXME: maybe we really should preserve the variadic kind when
      // prepending contextual parameters such that we don't need the check
      // here? But on the other hand, what does it mean to have a
      // inferred-pos-var-arg parameter?
      if (pog.getPassingKind() == PassingKind::Inferred &&
          ASTType(declaredParamTypes[idx]).getParameterListInfo().valueList)
        return !isInferForStruct;

      return false;
    }();

    if (isInferableVA) {
      // Infer the param_list to an empty list, and the ParameterList itself to
      // UnknownAttr.
      auto [varArgsEltType, expectedValueList] =
          ASTType(evaluator.getReboundType(declaredParamTypes[idx]))
              .getParameterListInfo();

      // If there are no values, default to an empty list.
      if (isa<ParamIndexRefAttr>(expectedValueList)) {
        auto paramVA = ParamListAttr::get(
            {}, cast<ParamListType>(expectedValueList.getType()));
        ParamMatcher matcher(getGivenBindings().callExpr, *this,
                             /*implConversions*/ false);
        if (failed(matcher.matchParams(paramVA, expectedValueList))) {
          auto &diag = getMojoDiag({});
          diag << "could not infer default variadic parameter "
               << declaredParamPogs.getPogs()[idx].getName();
          return failure();
        }
      }

      // The list itself doesn't have a value, so default it to {} now that it
      // has a concrete type.
      auto listValue =
          UnknownAttr::get(evaluator.getReboundType(declaredParamTypes[idx]));
      if (failed(setInitialInferredValue(idx, listValue))) {
        auto &diag = getMojoDiag({});
        diag << "failed to install default variadic parameter "
             << declaredParamPogs.getPogs()[idx].getName();
        return failure();
      }
    }
  }

  return success();
}

// TODO: We probably don't have to do this? This is just to make sure we reached
// the same end state as the old parameter inference. Understand why.
LogicalResult ParamInf::finalizeWithUnbound() {
  bool defaultToUnbound = paramBindings.bindingKind != ParamBindings::kStandard;

  // All kw-only parameter that is not inferrable.
  SmallVector<StringAttr> kwDiagNames;

  auto emitInferenceFailure = [&](size_t paramIdx) {
    MojoInflightDiag &diag = getMojoDiag(paramBindings.getExprLoc());
    if (declIfKnown && isa<StructDeclOp>(declIfKnown->getIfOperation()))
      diag << "'" << *declIfKnown->getUserNameIfOperation() << "' ";

    {
      // The parameter name is scoped to 'declScope'.
      DeclResolver::DiagnosticDeclContextChanger x(&paramBindings.declScope);
      diag << "failed to infer parameter "
           << ParamDeclRefAttr::get(declaredParamPogs.getName(paramIdx),
                                    declaredParamTypes[paramIdx]);
    }

    // If this is a method on a struct and we couldn't infer something from
    // its self parameters, complain about the struct.
    if (declIfKnown && isa<FnOp>(declIfKnown->getIfOperation())) {
      if (auto structOp = dyn_cast<StructDeclOp>(
              cast<FnOp>(declIfKnown->getIfOperation())->getParentOp())) {
        auto structSig = structOp.getSignature();
        if (paramIdx < structSig.getNumParams()) {
          diag << " of parent struct '" << structOp.getDeclName().getValue()
               << "'";
          diag.attachNote(structOp.getLoc()) << " struct declared here";
          return;
        }
      }
    }

    if (isInferForStruct)
      diag << ", specify the parameter or use '_' or '...' to unbind the "
              "parameter explicitly";
  };

  // This is the end of parameter inference, replace any fail-to-infer parameter
  // to unboundAttr.
  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    TypedAttr inferred = evaluator.getIndexBindings()[idx];
    if (inferred) {
      assert(!sugarIsa<UnboundAttr>(inferred));
      continue;
    }

    bool installUnbound = [&]() -> bool {
      // Call must produce a concrete type
      if (!isInferForStruct)
        return false;

      // There is a explicit unbound value provided and unbound is allowed.
      if (isExplicitlyUnbound(idx))
        return true;

      // `...` provides a default `_` value for any missing parameter. Besides,
      // we always allow inferred-only/implicit auto-parameterized parameter to
      // be defaulted to `_`. This is to allow:
      //
      // struct S[a: Int, //, b: Param[a]]:
      //   pass
      //
      // comptime _ = S[_] # NOTE that we don't require `a = _` here.
      //
      return pog.getPassingKind() == PassingKind::Inferred ||
             pog.getPassingKind() == PassingKind::Implicit || defaultToUnbound;
    }();

    if (installUnbound) {
      Type targetType = evaluator.getReboundType(declaredParamTypes[idx]);
      evaluator.overwriteIndexBinding(idx, UnboundAttr::get(targetType));
      continue;
    }

    if (pog.getPassingKind() != PassingKind::KwOnly) {
      emitInferenceFailure(idx);
      return failure();
    }
    // Collect all missing keyword-only and report.
    kwDiagNames.push_back(pog.getName());
  }

  if (!kwDiagNames.empty()) {
    MojoInflightDiag &diag = getMojoDiag({});
    emitMissing(diag, kwDiagNames, "keyword-only parameter");

    if (isInferForStruct)
      diag << ", specify the parameter or use '_' or '...' to unbind the "
              "parameter explicitly";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallParamInf Implementation
//===----------------------------------------------------------------------===//

/// Calculate the minimum required and maximum allowed number of positional
/// operands for a signature, assuming that the signature has a variadic pack;
static std::optional<std::pair<size_t, size_t>>
calculateRequiredPosOperandsForPacks(FnTypeGeneratorType signature,
                                     ASTType variadicPackType) {
  // This function heavily assumes that a signature has at most
  // one pack variadic argument and that variadics are always the last
  // positional args.
  size_t numPosArgs = countNumPositional(signature.getArgListAttrs());

  // We don't require any positional operands (because this function does not
  // check for passing kinds).
  if (!numPosArgs)
    return std::make_pair(0, numPosArgs);

  // The caller should ensure that the signature takes a variadic pack (and the
  // provided `variadicPackType` is the rebound type during this inference
  // session).
  size_t lastPosIdx = numPosArgs - 1;
  assert(signature.isPack(lastPosIdx));

  // If we have a non-empty variadic pack argument, we do require a certain
  // number of positional operands (since the value of positional packs cannot
  // be provided by keyword operands).
  // NOTE: in this case, it doesn't matter if there are preceding positional
  // arguments with default values: the pack cannot have a default value and
  // _must_ be provided positional operands explicitly, and therefore the
  // preceding defaults won't be used anyway.
  auto packInfo = variadicPackType.getVariadicPackInfo();
  // See if resolved.
  auto packed = sugarDynCast<ParamListAttr>(packInfo.typeList);

  // The caller should know the concrete type list unless we bound the pack
  // directly as a parameter.  This is an unpack like situation.
  // TODO: This happens in error cases and needs to be re-evaluated.
  if (!packed)
    return std::nullopt;

  // NOTE: we adjust the number of user declared pos args since that
  // includes the pack itself (hence the "-1").
  size_t packSize = packed.getValues().size();
  return std::make_pair(numPosArgs - 1 + packSize, numPosArgs - 1 + packSize);
}

CallParamInf::CallParamInf(const ParamBindings &paramBinding,
                           ArrayRef<Type> declaredParamTypes,
                           PogListAttr declaredParamPogs,
                           bool allowImplicitConversions, ASTDecl *declIfDirect,
                           bool discardError,
                           FnTypeGeneratorType calleeSignature,
                           const CallOperands &callOperands,
                           const OperandValueList &variadicKwOperands,
                           OperandsNeedingOriginsList &operandsNeedingOrigins)
    : ParamInf(paramBinding, declaredParamTypes, declaredParamPogs,
               allowImplicitConversions, declIfDirect, discardError),
      calleeSignature(calleeSignature), callOperands(callOperands),
      variadicKwOperands(variadicKwOperands),
      operandsNeedingOrigins(operandsNeedingOrigins) {}

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type, which it handled by emitting a diagnostic
/// and returning failure.
///
/// This can be called on a function signature with incomplete bindings, which
/// means that 'origExpectedType' may have unbound parameters.  As such, this
/// will infer parameters from the operand and return the inferred type.
///
/// operandIdx indicates the index of the operand in the CallOperands list, the
/// argIdx indicates which declared argument this corresponds to.  Note that
/// these may differ when using keyword arguments, and variadics have multiple
/// values that fulfill the same declared argument.
///
/// TODO: This is a more general mirror of 'OverloadFitness::checkOneOperand':
/// unify it into this.
LogicalResult CallParamInf::inferOneOperand(ASTExprAnd<AnyValue> operand,
                                            size_t operandIdx, size_t argIdx,
                                            ASTType expectedType,
                                            ArgConvention expectedConvention) {

  auto argPogs = calleeSignature.getArgListAttrs();

  // Make sure the diagnostic machinery knows about our getDeclScope() so
  // parameter names get emitted correctly.
  DeclResolver::DiagnosticDeclContextChanger x(declIfKnown);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getMojoDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, expectedType, argIdx, argPogs,
                        callOperands.syntax, getShared());
    return diag;
  };

  expectedType = evaluator.getReboundType(expectedType);
  ASTType expectedRVType =
      RefType::stripRefConvention(expectedType, expectedConvention);

  // TODO: Calculate OverloadFitness's fitness (# implicit conversions etc).
  ParamMatcher matcher(operand.expr, *this, allowImplicitConversions);

  // This gets set if we need to spill the argument to memory to get an origin.
  bool needsArgInMemory = false;

  // We'll bind the next provided value.
  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal) {
      auto &diag = getMojoDiag(operand.expr->getLoc());
      if ((callOperands.syntax == CallSyntax::kMethodCall ||
           callOperands.syntax == CallSyntax::kMethodCallSynthetic) &&
          argIdx == 0) {
        diag << "invalid use of mutating method on rvalue of type ";
        if (ASTType type = operand.ir.getRValueTypeIfResolvable())
          diag << type;
        else
          printUValueTypeInfo(operand.ir, diag);
      } else {
        diag << "value passed to mutable argument " << argPogs.getName(argIdx)
             << " must be mutable";
      }
      diag << operand.expr->getRange();
      return failure();
    }

    // If this is a wildcard type, we can match any operand.
    if (sugarIsa<NameLookupArgWildcardType>(argVal.getRValueType()))
      return success();

    // Ok we have an LValue.  The reference element types must match.
    if (failed(matcher.matchTypes(argVal.getRValueType(), expectedRVType))) {
      // ByRef argument types must exactly match, no conversions are allowed.
      auto &diag = getMojoDiag(operand.expr->getLoc());
      diag << "l-value of type " << operand.ir.getIfLValue().getRValueType()
           << " cannot be converted to reference of type " << expectedRVType
           << operand.expr->getRange();
      matcher.failureReason->addExplanation(diag);
      return failure();
    }
    break;
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    auto expectedRef = sugarCast<RefType>(expectedType);

    // If we are binding the reference to a value in memory directly, check for
    // reference compatibility directly.
    if (operand.ir.isMValue()) {
      RefType valueRefType = operand.ir.getMValueType();
      // If the IRValue type is MBValue or MRValue then we need infer an
      // immutable ref, to match behavior where we don't allow passing an
      // MBValue or MRValue as 'mut'.
      if (!operand.ir.getIfMLValue() && !operand.ir.getIfMBPValue() &&
          !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

      // Refine the element type first.
      if (failed(matcher.matchTypes(valueRefType.getElementType(),
                                    expectedRef.getElementType()))) {
        emitWrongTypeDiag(expectedType);
        return failure();
      }
      expectedType = evaluator.getReboundType(expectedType);

      // Now that element type has been matched, see if the origin is already
      // specified, allow implicit conversions, allowing you to pass a concrete
      // origin to something expecting a union or AnyOrigin.  This check happens
      // here (instead of in matchTypes) because function arguments can be
      // rebound when origins disagree, but this isn't correct/possible in
      // arbitrary nested positions.
      if (!paramFinder.hasReferences(expectedType)) {
        if (!IREmitter::canZeroCostConvert(valueRefType, expectedType,
                                           getShared())) {
          emitWrongTypeDiag(expectedType);
          return failure();
        }
      } else {
        // Otherwise, match the references as a whole - this matches the origins
        // up to infer from the value.
        if (failed(matcher.matchTypes(valueRefType, expectedType))) {
          emitWrongTypeDiag(expectedType);
          return failure();
        }
      }
      break;
    }

    // Otherwise, we are binding something like a PValue or SRValue to a
    // reference argument, which doesn't have a origin.  This is a problem
    // because origins can be propagated through the type system of the
    // function call to other arguments and they all need to line up.  We
    // handle this in two phases: during overload resolution we bind this to
    // an immortal origin, and then after the candidate is selected, we
    // re-emit these arguments to memory and re-infer all the parameters.
    //
    // One detail is how we do this: we bind these arguments to immutable
    // temporaries, because we specifically do NOT want 'ref' arguments with
    // parametric mutability to treat these things as mutable.
    if (sugarCast<RefType>(expectedType).isMutableKnown(true)) {
      auto &diag = getMojoDiag(operand.expr->getLoc());
      diag << "mutable reference argument " << argPogs.getName(argIdx)
           << "cannot bind to temporary value";
      return diag;
    }

    // Otherwise, we'll need to drop this value into a temporary. Notice this so
    // we can handle it after we infer the element type.
    needsArgInMemory = true;

    // Until then, infer it as AnyOrigin.  We bind the origin directly and then
    // handle it like any other argument because we can support
    // implicit conversions.
    auto anyOrigin =
        AnyOriginAttr::get(expectedRef.getContext(), /*isMut=*/false);
    ParamMatcher::FailableScope failableScope1(matcher);
    if (failed(
            matcher.matchSingleEltStruct(anyOrigin, expectedRef.getOrigin()))) {
      // Ignore failures because we only want to set a value if none is
      // already known so things aren't ambiguous.
      // TODO: it would be cleaner to check to see if this is already inferred
      // and only default it if not.
      failableScope1.revert();
    }

    // The address space of the temp will be the default.
    auto addrSpace =
        IntegerAttr::get(IndexType::get(expectedRef.getContext()), 0);

    ParamMatcher::FailableScope failableScope2(matcher);
    if (failed(matcher.matchSingleEltStruct(addrSpace,
                                            expectedRef.getAddressSpace()))) {
      failableScope2.revert();
    }

    // Handle the element type compatibility check below to allow implicit
    // conversions etc.
    [[fallthrough]];
  }
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
  case ArgConvention::ReadMem:
  case ArgConvention::ReadReg:
    break;
  }

  // Call the core matching logic after handling the convention.
  if (failed(inferFromRVType(operand, argIdx, expectedRVType, argPogs,
                             callOperands.syntax)))
    return failure();

  // We may have refined expectedRVType.
  expectedRVType = evaluator.getReboundType(expectedRVType);

  // If the argument needed to be spilled to memory to get an origin,
  // record it so call emission can reinfer and reemit this candidate if
  // selected from the overload set, but with the argument in a temporary
  // vardecl.
  if (needsArgInMemory) {
    assert(operandIdx != ~0ULL && "FIXME: KWVarArgs not passing correctly");
    operandsNeedingOrigins.push_back({operandIdx, argIdx, expectedRVType});
  }

  // If a register-passable type is being passed in-memory, remember this.
  if (expectedConvention != ArgConvention::ReadReg &&
      expectedRVType.isRegisterPassable(operand.expr->getLoc(), getShared()))
    ++numMismatchedConventions;

  // Allow overloading on "owned" vs "by-ref" arguments.
  // If the argument convention is owned but the operand is not an RValue then
  // we'll need to copy the value (or this is entirely invalid).  If the
  // argument convention is borrowed/ref but the value is an RValue then we have
  // an RValue decay.  Model these so that APIs can overload on owned vs
  // borrowed effectively.
  if (!operand.ir.getIfCValue() ||
      operand.ir.getIfCValue().getRValueType().isEqualCanon(expectedRVType)) {
    if (operand.ir.getIfBValue() || operand.ir.getIfLValue()) {
      // Heavily penalize implicit copies.
      if (expectedConvention == ArgConvention::OwnedMem ||
          expectedConvention == ArgConvention::DeinitMem)
        numMismatchedConventions += 2;
    } else {
      assert((operand.ir.getIfUValue() || operand.ir.getIfRValue()) &&
             "UValue and RValue expressions are always owned");
      // Slightly penalize RValue->ref conversions.
      if (expectedConvention != ArgConvention::OwnedMem &&
          expectedConvention != ArgConvention::DeinitMem)
        ++numMismatchedConventions;
    }
  }

  return success();
}

/// Try to infer parameters of Self from an initializer if specialized.
///
/// Consider:
///    struct S[a: Int]:
///      def __init__(out self): ...
///      def __init__(out self: S[1], x: Int): ...
///
/// When constructed with no arguments, the first constructor must be used and
/// it is impossible to infer the value of 'a', so you must use `S[1]()`.  This
/// is the usual case.
///
/// However the second initializer is more specialized due to its custom Self -
/// it only applies when 'a' is 1, so we can infer that would be the value to
/// use if it is selected because one arg is passed to the initializer `S(42)`.
///
/// This function helps to infer the 'a' parameter when more specialized.  This
/// custom logic is required because often (eg in this case) the "actual" type
/// will have UnboundAttr parameters, instead of fully bound ones like a normal
/// argument.
LogicalResult CallParamInf::inferSelfFromInitResult() {
  DeclResolver::DiagnosticDeclContextChanger x(declIfKnown);

  ASTType returnedType =
      evaluator.getReboundType(calleeSignature.getUserResultType());

  auto reportConflict = [&](size_t paramIdx, TypedAttr actual,
                            TypedAttr expected) -> LogicalResult {
    getMojoDiag(getGivenBindings().callExpr->getLoc())
        << "return type " << returnedType << " parameter "
        << ParamIndexRefAttr::get(/*depth*/ 0, paramIdx, actual.getType())
        << " value " << actual << " doesn't match expected value " << expected;
    return failure();
  };

  // Match up the parameter bindings if the 'actual' param is an UnboundAttr and
  // the expected has something more specific than a reference to the contextual
  // parameter.
  for (auto [idx, retParam] :
       llvm::enumerate(returnedType.getParamBindings())) {
    // If this is simply a reference to the enclosing parameter (as in a normal
    // Self) init, then we can't infer anything from it.  In the example above,
    // this ignores the "a" parameter in "def __init__() -> S[a]:" which is what
    // "out self" desugars to.
    auto selfParam = evaluator.getIndexBindings()[idx];
    if (retParam == selfParam)
      continue;

    // Otherwise, if the self parameter got inferred, propagate the result
    // from it to the returned parameter.  This handles things like:
    //   struct X[A: AnyType]:
    //     def __init__[T: Movable](arg: Int, out self: X[T]):
    // which gets used as X[String](42) inferring T and A.
    ParamMatcher matcher(getGivenBindings().callExpr, *this,
                         allowImplicitConversions);
    if (selfParam) {
      // TODO: Macro'ize this when error handling logic is fixed.
      if (failed(matcher.matchParams(selfParam, retParam))) {
        return reportConflict(idx, retParam, selfParam);
      }
    } else if (!paramFinder.hasReferences(retParam)) {
      // Otherwise if the the returned parameter has no unbound parameter
      // references then we infer the self parameter from it. This infers X=42:
      //   struct X[A: Int]:
      //     def __init__(out self: X[42]):
      auto selfType =
          evaluator.getReboundType(calleeSignature.getInputParamTypes()[idx]);
      auto selfParam = ParamIndexRefAttr::get(/*depth*/ 0, idx, selfType);
      if (failed(matcher.matchParams(retParam, selfParam))) {
        return reportConflict(idx, selfParam, retParam);
      }
    }
  }

  return success();
}

/// This method is called for ByRefResult arguments of the callee.  It checks to
/// see if the callee has a parametric address space or origin. If so, it looks
/// at the ExprDest the call is being emitted into and infers the desired
/// values, or marks it as needing to be spilled if not.
LogicalResult CallParamInf::inferResultSlot(RefType expectedRef, size_t argIdx,
                                            const ExprDest &dest) {

  // Penalize generic code slightly.
  if (ASTType(expectedRef.getElementType())
          .isRegisterPassable(getGivenBindings().callExpr->getLoc(),
                              getShared()))
    ++numMismatchedConventions;

  bool needsAddrSpace =
      paramFinder.hasReferences(expectedRef.getAddressSpace());
  bool needsOrigin = paramFinder.hasReferences(expectedRef.getOrigin());
  if (!needsAddrSpace && !needsOrigin)
    return success(); // Nothing to do.

  RefType actualRef;
  // If we have a concrete MLValue, we can use it to infer the desired values.
  if (MLValue mlDest = dest.getDirectMLValueIfPresent()) {
    actualRef = mlDest.getRefType();
  } else {
    // If the ExprDest lacks a concrete MLValue, we can't infer anything. We
    // need the caller to spill the result into a buffer and reinfer us. Until
    // then, bind it as AnyOrigin to avoid failing to infer the parameters.
    operandsNeedingOrigins.push_back({OperandNeedingOrigin::kExprDestOperandIdx,
                                      argIdx, expectedRef.getElementType()});

    if (needsOrigin)
      actualRef = expectedRef.getWithOrigin(
          AnyOriginAttr::get(expectedRef.getContext(), /*isMut=*/true));
    if (needsAddrSpace)
      actualRef = actualRef.getWithAddressSpace(
          IntegerAttr::get(IndexType::get(expectedRef.getContext()), 0));
  }

  ParamMatcher matcher(getGivenBindings().callExpr, *this,
                       /*allowImplicitConversions=*/false);

  if (failed(matcher.matchSingleEltStruct(actualRef.getAddressSpace(),
                                          expectedRef.getAddressSpace())) ||
      failed(matcher.matchSingleEltStruct(actualRef.getOrigin(),
                                          expectedRef.getOrigin())))
    return failure();
  return success();
}

/// Given an incomplete parameter binding set, try to infer parameters on Self
/// of a method from the first argument.
LogicalResult CallParamInf::inferCTADParams() {
  // Consider "conditional conformance" cases like:
  //     struct X[A: AnyType]:
  //       def foo[B: Movable](self: X[B]): ...
  //
  // When resolving a function call like `someX.foo()`, we install the
  // bindings for 'A' from the typeof(someX) when resolving the
  // AttributeRefExpr and then infer 'B' from someX again.
  //
  // However, when we have something like `X.foo(someX)` we cannot install the
  // bindings for 'A' at AttributeRef resolution time, and 'someX' is only
  // bound by parameter inference to 'B'.  Notice this and infer the parameter
  // directly from A.  This is also important for operator resolution, which
  // works effectively the same way.
  //
  // TODO: Provide a first class representation for conditional conformance
  // that doesn't have us shadowing parameters like this!

  // We can only do this if we have an argument.
  assert(!callOperands.empty() && !callOperands[0].keyword &&
         "init should have positional self argument");

  auto selfConvention = calleeSignature.getArgConventions()[0];
  ASTType declaredSelfType = RefType::stripRefConvention(
      calleeSignature.getArgument(0), selfConvention);

  // Get the ASTDecl for the declared self type.  This will give us the struct
  // that we are referring to without bound parameters.
  ASTDecl *decl = declaredSelfType.getDecl(getShared());
  if (!decl)
    return success();

  // Get the Self type, with parameters bound to the structs CTAD parameters.
  ASTType selfType = decl->getTypeDeclSelf();
  if (!selfType)
    return success();

  // We need to convert named parameters like "T", which are ParamDeclRefAttr
  // into ParamIndexRefAttr(0) style of representation.
  if (auto structDecl = dyn_cast<StructDeclOp>(decl->getIfOperation())) {
    IndexRefRemapper remapper(structDecl.getParams(), /*resultParams*/ {});
    selfType = remapper.replace(selfType.mlirType);
  }

  // If passing self by reference, wrap the Self type with the RefType
  // paraphernalia like origins.
  if (hasAddress(selfConvention))
    selfType = sugarCast<RefType>(calleeSignature.getArgument(0))
                   .getWithElement(selfType);

  // Infer the first operand against this type - it was presumably already
  // inferred against the methods declared type of 'self' as well.
  return inferOneOperand(callOperands[0], /*operandIdx*/ 0, /*argIdx*/ 0,
                         selfType, selfConvention);
}

LogicalResult CallParamInf::inferForCall() {
  isInferForStruct = false;

  CrashReporter handler(paramBindings.getExprLoc(),
                        "CallParamInf::inferForCall", getShared());

  // First try to infer parameters from the already provided bindings.
  if (failed(inferFromParamList()))
    return failure();

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  size_t numOperands = callOperands.size();
  PogListAttr argPogs = calleeSignature.getArgListAttrs();
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(calleeSignature.getArgConventions())) {

    // Note that 'calleeSignature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType =
        evaluator.getReboundType(calleeSignature.getArgument(expectedArgIdx));

    // There is no provided operand for a by-ref result and error slot.
    if (isResultSlot(expectedConvention)) {
      auto expectedRef = sugarCast<RefType>(expectedType);

      // If this is the result slot with parametric components, attempt to infer
      // result origin/address space from it.
      if (expectedConvention == ArgConvention::ByRefResult)
        if (failed(inferResultSlot(expectedRef, expectedArgIdx,
                                   callOperands.dest)))
          return failure();
      continue;
    }

    // Check for any more positional operands. This ensures the code handling
    // positional arguments below is looking at the next one to process or that
    // we have run out.
    while (posOperandIdx != numOperands && callOperands[posOperandIdx].keyword)
      ++posOperandIdx;

    if (calleeSignature.isKwVarArg(expectedArgIdx)) {
      Type valTy = ASTType(expectedType).getKwargsDictRefValueType();
      auto refValType = RefType::getAnyOrigin(valTy, /*isMut=*/true);
      for (auto operand : variadicKwOperands) {
        // TODO: Passing OwnedMem is a hack that is needed because the value
        // type is not a reference type (and doesn't have a origin), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Pointer` does not
        // (and in fact cannot) conform to `Copyable & Movable`.
        if (failed(inferOneOperand(operand, /*operandIdx unknown*/ ~0ULL,
                                   expectedArgIdx, refValType,
                                   ArgConvention::OwnedMem)))
          return failure();
      }
      // This is always last in the operand list.
      posOperandIdx = numOperands;
      continue;
    }

    // Determine if we can use an value for this argument directly, or
    // if we need an implicit conversion, or memory materialization to get
    // an origin.
    auto canUseMValue = [&](AnyValue value, ASTType expectedType,
                            ArgConvention convention) -> bool {
      // The operand must an MValue and must have the same element type as
      // the variadic list element type (otherwise a conversion is needed).
      if (!value.isMValue())
        return false; // Can't use it if not an MValue obviously.

      // The origin has to be in the default address space.
      if (!value.getMValueType().isDefaultAddrSpace())
        return false;

      // The argument must have a compatible element type (and we might
      // infer the type of the variadic from it.  If not, there must be an
      // implicit conversion going on.  We can test for type equality here
      // because inferOneOperand will have inferred the type from the arg.
      // TODO: Move this logic into inferOneOperand.
      expectedType = evaluator.getReboundType(expectedType);
      if (!expectedType.isEqualCanon(value.getMValueType().getElementType()))
        return false; // Implicit conversion will generate a new temp.

      // If this is a owned operand, we can use it if we have an RValue.
      if (convention == ArgConvention::OwnedMem)
        return !!value.getIfRValue();

      // TODO: What about "mut" arguments getting passed MBValues?
      return true;
    };

    // Given a call argument that will be bound to the specified operand of a
    // callee, get the memory origin of the value (if it can be used) or mark it
    // as needing to be spilled if not.
    auto getArgOrigin = [&](AnyValue value, ASTType expectedType, size_t argIdx,
                            size_t operandIdx, ArgConvention convention,
                            OriginType expectedOriginType) -> TypedAttr {
      if (canUseMValue(value, expectedType, convention)) {
        // The argument could be mutable, but the arg convention may expect
        // immutable.
        auto opOrigin = value.getMValueType().getOrigin();
        return OriginMutCastAttr::get(opOrigin, expectedOriginType);
      }
      // The value isn't in memory (or isn't usable in memory) yet.  We will
      // tell call emission that it needs to dump it in memory and try again
      // to use this callee.  Until then, we use AnyOrigin as a placeholder.
      operandsNeedingOrigins.push_back(
          {operandIdx, argIdx, evaluator.getReboundType(expectedType)});
      return AnyOriginAttr::get(expectedOriginType);
    };

    // Handle ranking for variadics.  Packs and positional rank the same way.
    if (calleeSignature.isPosVarArg(expectedArgIdx) ||
        calleeSignature.isPack(expectedArgIdx)) {
      if (posOperandIdx != numOperands) {
        // Remember that there is a variadic argument for overload ranking.
        passesVarArgArgument = true;
      } else {
        // We consider an empty varargs list to be an implicit conversion,
        // so an exact signature match takes precedence.
        ++numImplicitConversions;
      }
    }

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (calleeSignature.isPosVarArg(expectedArgIdx)) {
      ASTType expectedRVType =
          RefType::stripRefConvention(expectedType, expectedConvention);
      // The expected origin type will always have its mutability known because
      // the arg convention of the VariadicList is always constant.
      auto variadicListInfo = expectedRVType.getVariadicListInfo();
      auto expectedOriginType =
          cast<OriginType>(variadicListInfo.origin.getType());
      auto argConvention =
          calleeSignature.getVariadicConvention(expectedArgIdx);

      // Support forwarding an entire list with "*list".
      if (posOperandIdx != numOperands &&
          callOperands[posOperandIdx].isUnpackedPositional()) {

        if (failed(inferOneOperand(callOperands[posOperandIdx], posOperandIdx,
                                   expectedArgIdx, expectedType,
                                   expectedConvention)))
          return failure();
        ++posOperandIdx;
        continue;
      }

      // Otherwise, we're binding a sequence of values into the list.

      // TODO: This is subtly wrong in a way that doesn't matter. We're passing
      // the ultimate origin in as the origin for the RefType, but we need to
      // infer the union all of the arg origins: not just the first arg's
      // origin.  inferOneOperand currently doesn't do anything with this except
      // for 'ref' convention, that we don't support in variadics.  When we do
      // or when we get rid of implicit origins, this will need to be adjusted
      // to pass in something that matches anything so the code below can
      // infer the correct origin union.
      auto varArgsEltType =
          RefType::get(variadicListInfo.elementType, variadicListInfo.origin);

      SmallVector<TypedAttr> argOrigins;
      while (posOperandIdx != numOperands) {
        auto &operand = callOperands[posOperandIdx++];

        // Passed arguments with keywords specified don't bind to varargs.
        if (operand.keyword)
          continue;

        if (operand.isUnpackedPositional()) {
          getMojoDiag(operand.expr->getLoc())
              << "cannot unpack a value into a variadic argument";
          return failure();
        }

        if (failed(inferOneOperand(operand, posOperandIdx - 1, expectedArgIdx,
                                   varArgsEltType, argConvention)))
          return failure();

        // Keep track of all the arg origins so we can infer from them later.
        argOrigins.push_back(getArgOrigin(
            operand.ir, variadicListInfo.elementType, expectedArgIdx,
            posOperandIdx - 1, argConvention, expectedOriginType));
      }

      // Infer the origin of the variadic list from the unified origins of the
      // arguments.
      auto commonOrigin = OriginUnionAttr::get(argOrigins, expectedOriginType);
      ParamMatcher matcher(getGivenBindings().callExpr, *this,
                           /*noImplicitConversions=*/false);
      if (failed(matcher.matchParams(commonOrigin, variadicListInfo.origin)))
        return failure();

      continue;
    }

    // If we have a pack argument, then we're binding a variadic parameter with
    // multiple type values.  We need to consume all remaining arguments and use
    // their RValue types as bindings.
    if (calleeSignature.isPack(expectedArgIdx)) {
      ASTType variadicPackType =
          RefType::stripRefConvention(expectedType, expectedConvention);
      variadicPackType = evaluator.getReboundType(variadicPackType);
      ASTType::VariadicPackInfo expectedInfo =
          variadicPackType.getVariadicPackInfo();

      // Support forwarding an entire pack with "*pack".
      if (posOperandIdx != numOperands &&
          callOperands[posOperandIdx].isUnpackedPositional()) {

        ASTType actualPackType = // FIXME: This is wrong for UValues.
            callOperands[posOperandIdx].ir.getRValueTypeIfResolvable();
        assert(actualPackType &&
               "unpacked positional operand must have a resolvable type");

        // Check that the actual type is the same struct type as the expected
        // VariadicPack. If not, the user tried to unpack a non-pack type
        // (e.g., a List) which is not allowed.
        ASTDecl *actualDecl = actualPackType.getDecl(getShared());
        ASTDecl *expectedDecl = variadicPackType.getDecl(getShared());
        if (!actualDecl || actualDecl != expectedDecl) {
          auto &diag = getMojoDiag(callOperands[posOperandIdx].expr->getLoc());
          diag << "cannot unpack value of type " << actualPackType
               << " into a variadic pack argument; expected a VariadicPack";
          return failure();
        }

        ASTType::VariadicPackInfo actualInfo =
            actualPackType.getVariadicPackInfo();
        if (actualInfo.isOwned != expectedInfo.isOwned) {
          auto &diag = getMojoDiag(callOperands[posOperandIdx].expr->getLoc());
          diag << "cannot unpack a variadic pack into a call that requires a "
                  "different ownership. Expected "
               << expectedInfo.isOwned << ", got " << actualInfo.isOwned;
          return failure();
        }

        // Skip matching the origin, since the expected origin is an implicit
        // origin that will be filled in during call emission. Just make sure
        // that the element types match.
        RefPackType actualRefPackType =
            actualPackType.getVariadicPackInfo(getShared());
        RefPackType expectedRefPackType =
            variadicPackType.getVariadicPackInfo(getShared());

        auto actualMutable = actualRefPackType.getOriginType().getIsMutable();
        auto expectedMutable =
            expectedRefPackType.getOriginType().getIsMutable();
        auto bothMutable =
            ParamOperatorAttr::get(POC::And, actualMutable, expectedMutable);
        if (bothMutable != expectedMutable) {
          auto &diag = getMojoDiag(callOperands[posOperandIdx].expr->getLoc());
          diag << "cannot unpack a variadic pack into a call that requires a "
                  "stricter mutability. Expected "
               << expectedMutable << ", got " << actualMutable;
          return failure();
        }

        ParamMatcher matcher(callOperands[posOperandIdx].expr, *this,
                             allowImplicitConversions);
        if (failed(matcher.matchParams(actualRefPackType.getVariadic(),
                                       expectedRefPackType.getVariadic())) ||
            failed(matcher.matchParams(actualRefPackType.getOrigin(),
                                       expectedRefPackType.getOrigin()))) {
          auto &diag = getMojoDiag(callOperands[posOperandIdx].expr->getLoc());
          diag << "cannot unpack a pack of type "
               << actualRefPackType.getParamListElementType()
               << " into a call that expects a pack of type "
               << expectedRefPackType.getParamListElementType();
          matcher.failureReason->addExplanation(diag);
          return failure();
        }

        // Now that we bound the elements of the TypeList, we can infer the
        // value of the TypeList struct.
        auto typeListType =
            evaluator.getReboundType(expectedInfo.typeListStruct.getType());
        auto typeListValue = UnknownAttr::get(typeListType);
        (void)matcher.matchParams(typeListValue, expectedInfo.typeListStruct);

        ++posOperandIdx;
        continue;
      }

      // Otherwise, we're binding a sequence of values into the pack.
      RefPackType packType = variadicPackType.getVariadicPackInfo(getShared());

      // Figure out that the element type of the list is, e.g. AnyType or
      // Stringable.
      Type elementType = packType.getParamListElementType();
      auto expectedOriginType = packType.getOriginType();

      // It is possible the pack element types are not being inferred - for
      // example, they could have been explicitly specified.  If this is the
      // case, then we need to perform an implicit conversion to the element
      // type that was explicitly specified.  Be careful though, it is possible
      // the specified type list is completely wrong in length or content.
      ParamListAttr eltsTypesIfResolved =
          dyn_cast<ParamListAttr>(packType.getVariadic());

      SmallVector<TypedAttr> types;
      SmallVector<TypedAttr> argOrigins;
      IREmitter emitter(getDeclScope(), EC_TypeParamValue);
      const ExprNode *packArgExpr = nullptr;
      while (posOperandIdx != numOperands) {
        const auto &operand = callOperands[posOperandIdx++];
        if (operand.keyword) // Ignore keyword operands.
          continue;

        // Remember the first argument expression for the pack.
        if (packArgExpr == nullptr)
          packArgExpr = operand.expr;

        // If the element types for the pack were specified, convert the value
        // to that type.
        TypedAttr eltTypeValue;
        if (eltsTypesIfResolved &&
            types.size() < eltsTypesIfResolved.getValues().size()) {
          eltTypeValue = eltsTypesIfResolved.getValues()[types.size()];
        } else {
          // Otherwise, infer the variadic element type from the value's type.
          ASTType toPush = operand.ir.getRValueTypeIfResolvable();

          // Initializer UValues (list/dict/set/slice literals) don't have a
          // resolvable RValue type until they're bound to a target type.
          // Apply the same fallback `inferCValue` uses so a literal passed to
          // a trait-bound pack binds to its default type instead of bailing
          // out with a bogus "unresolved type" diagnostic.
          if (!toPush) {
            if (auto initValue = operand.ir.getIfInitializer())
              toPush = tryInferInitializerType(getDeclScope(), *initValue,
                                               operand, ASTType(elementType));
          }

          if (!toPush) {
            getMojoDiag(operand.expr->getLoc())
                << "could not infer type of parameter pack "
                << argPogs.getName(expectedArgIdx)
                << " given value with unresolved type";
            return failure();
          }

          // Infer nonmaterializable types as their materialization target.
          if (ASTType nmTarget = toPush.getNonmaterializableTarget(getShared()))
            toPush = nmTarget;

          Type metatype = toPush.extractMetaType();
          eltTypeValue = TypeParamAttr::get(toPush, metatype);
          // Make sure the value is compatible with the expected trait, this
          // produces better error messages.  It would be great to sink this
          // into matchType at some point!
          if (!IREmitter::canImplicitlyConvertToType(
                  {eltTypeValue, operand.expr}, elementType,
                  emitter.getDeclScope())) {
            getMojoDiag(operand.expr->getLoc())
                << "could not convert element of "
                << argPogs.getName(expectedArgIdx) << " with type " << toPush
                << " to expected type " << elementType;
            return failure();
          }

          // Perform a conversion (e.g. from a concrete to trait type) as
          // needed.
          // FIXME(MOCO-3601): We have been very unprincipled about converting
          // using TypeParamAttr/UpcastAttr: They both are used as a way to
          // `rebind` type values. We have to use upcast here because we
          // have a upcast inserted for variadic element type for Tuple.
          if (!ASTType(eltTypeValue.getType()).isEqualCanon(elementType)) {
            if (isa<NonStructTypeType>(eltTypeValue.getType())) {
              eltTypeValue = emitter.emitPValue({eltTypeValue, operand.expr},
                                                EC_TypeParamValue, elementType);
            } else {
              eltTypeValue = UpcastAttr::get(elementType, eltTypeValue);
            }
          }
        }

        RefType refType =
            packType.getElementRefTypeFor(ASTType(eltTypeValue).mlirType);
        ArgConvention packEltConvention =
            calleeSignature.getVariadicConvention(expectedArgIdx);
        if (failed(inferOneOperand(operand, posOperandIdx - 1, expectedArgIdx,
                                   refType, packEltConvention))) {
          return failure();
        }

        // Keep track of all the arg origins so we can infer from them later.
        argOrigins.push_back(getArgOrigin(
            operand.ir, refType.getElementType(), expectedArgIdx,
            posOperandIdx - 1, packEltConvention, expectedOriginType));
        types.push_back(eltTypeValue);
      }

      ParamMatcher matcher(packArgExpr, *this, allowImplicitConversions);

      // Infer the origin of the pack from the unified origins of the
      // arguments.
      auto commonOrigin = OriginUnionAttr::get(argOrigins, expectedOriginType);
      if (failed(matcher.matchParams(commonOrigin, packType.getOrigin())))
        return failure();

      // Infer the value of type list from the types we have.
      auto variadicType =
          sugarCast<ParamListType>(packType.getVariadic().getType());

      // If there are no arguments for the pack, use the location of the call.
      if (!packArgExpr)
        packArgExpr = getGivenBindings().getExpr();
      auto actualVA = ParamListAttr::get(types, variadicType);
      if (succeeded(matcher.matchParams(actualVA, packType.getVariadic()))) {
        // Now that we bound the elements of the TypeList, we can infer the
        // value of the TypeList struct.
        auto typeListType =
            evaluator.getReboundType(expectedInfo.typeListStruct.getType());
        auto typeListValue = UnknownAttr::get(typeListType);
        (void)matcher.matchParams(typeListValue, expectedInfo.typeListStruct);
        continue;
      }

      // Match failed, diagnose why:
      std::optional<std::pair<size_t, size_t>> posNumBoundOr =
          calculateRequiredPosOperandsForPacks(calleeSignature,
                                               variadicPackType);
      // This means that we can not determine a concrete number of packed
      // arguments, this is always an error.
      MojoInflightDiag &diag = getMojoDiag({packArgExpr->getLoc()});
      if (!posNumBoundOr) {
        diag << "assigning " << numOperands << " operand" << plural(numOperands)
             << " to an unresolvable variadic pack argument";
        return failure();
      }

      auto [minPosOperands, maxPosOperands] = *posNumBoundOr;
      size_t numPosOperands = callOperands.getNumPositional();
      if (numPosOperands < minPosOperands || maxPosOperands < numPosOperands) {
        diag << "callee with non-empty variadic pack argument";
        emitWrongArgOrParamCount(diag, minPosOperands, maxPosOperands,
                                 numOperands, "positional operand");
        return failure();
      }
      llvm_unreachable("unhandled variadic pack failure?");
    }

    // Handle positional arguments.
    if (posOperandIdx < numOperands) {
      const OperandValue &operand = callOperands[posOperandIdx];
      if (operand.isUnpackedPositional()) {
        getMojoDiag(operand.expr->getLoc())
            << "unpacked positional arguments are only supported for callees "
               "that expect a variadic pack argument at this position";
        return failure();
      }
      if (failed(inferOneOperand(operand, posOperandIdx, expectedArgIdx,
                                 expectedType, expectedConvention)))
        return failure();
      ++posOperandIdx;
      continue;
    }

    // Handle case when there are no more provided positional operands.
    // Check if a keyword operand was provided for this argument
    if (const OperandValue *kwOperandOr = callOperands.findKwArg(
            calleeSignature.getArgName(expectedArgIdx))) {
      size_t operandIdx = kwOperandOr - callOperands.values.begin();
      if (failed(inferOneOperand(*kwOperandOr, operandIdx, expectedArgIdx,
                                 expectedType, expectedConvention)))
        return failure();
      continue;
    }

    // If not, and this argument has a default value, then infer from default
    // values - it might not match the argument type in a parametric situation.
    if (auto defaultVal = argPogs.getDefault(expectedArgIdx)) {
      defaultVal = evaluator.getReboundAttribute(defaultVal);
      if (failed(inferOneOperand({defaultVal, getGivenBindings().getExpr()},
                                 /*FIXME*/ ~0ULL, expectedArgIdx, expectedType,
                                 expectedConvention)))
        return failure();
      continue;
    }

    // Otherwise we have an argument count mismatch, just fail.
    return failure();
  }

  // If we have left over operands, then this signature cannot match.
  if (posOperandIdx != numOperands &&
      !calleeSignature.getMetadata().hasAnyVarArg())
    return failure();

  // If this is a result in a returnsSelf function like an __init__, infer
  // self parameters (which could be specialized and shadowed).
  //   struct Example[T: AnyType]:
  //      def __init__[U: Movable](owned value: U) -> Example[U]:
  //         pass
  // All of the arguments have been resolved here so all parameters must be
  // inferred (or not able to).
  if (declIfKnown && cast<FnOp>(declIfKnown->getIfOperation())
                         .getSpecialFunctionInfo()
                         .hasSelfResult()) {
    if (failed(inferSelfFromInitResult()))
      return failure();
  }

  // Check to see if this is a CTAD parameter - a parameter on the struct
  // that encloses the method.  Consider "conditional conformance" cases like:
  //     struct X[A: AnyType]:
  //       def foo[B: Movable](self: X[B]): ...
  // When resolving a function call like `someX.foo()`, we install the
  // bindings for 'A' from the typeof(someX) when resolving the
  // AttributeRefExpr and then infer 'B' from someX again.
  //
  // However, when we have something like `X.foo(someX)` we cannot install the
  // bindings for 'A' at AttributeRef resolution time, and 'someX' is only
  // bound by parameter inference to 'B'.  Notice this and infer the parameter
  // directly from A.  This is also important for operator resolution, which
  // works effectively the same way.
  //
  // TODO: Provide a first class representation for conditional conformance
  // that doesn't have us shadowing parameters like this!
  if (declIfKnown) {
    auto fnOp = cast<FnOp>(declIfKnown->getIfOperation());
    if (!fnOp.getIsStatic() && isa<StructDeclOp>(fnOp->getParentOp())) {
      if (failed(inferCTADParams()))
        return failure();
    }
  }

  // Lastly, See if we can fulfill any missing parameters with default values
  // for their type (variadic attr always have a default empty value if not
  // inferable).
  if (failed(inferFromDefaults()))
    return failure();

  if (hasDeferredGivenParam) {
    // Simply try it again now that more parameter has been inferred.
    if (failed(inferFromParamList()))
      return failure();
  }

  // See if we still have any unbound attr, if so, report error. (This must be a
  // full binding context).
  if (failed(finalizeWithUnbound()))
    return failure();

  // We succeed iff we inferred a value for this parameter.
  return success();
}
