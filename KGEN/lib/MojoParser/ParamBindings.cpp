//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ParamBindings.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/ParameterInference.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"

#include "OperandDiagnostics.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ParamBindings
//===----------------------------------------------------------------------===//

/// Replace our bindings with another set.  This can't be done with operator=
/// because we have
void ParamBindings::operator=(ParamBindings &&other) {
  posBindings = std::move(other.posBindings);
  kwBindings = std::move(other.kwBindings);
  defaultTypeParams = other.defaultTypeParams;
  numCtadParams = other.numCtadParams;
  numPreTypeChecked = other.numPreTypeChecked;
}

/// Create a (possibly partially unbound) set of bindings for the given type.
/// This can be used to initialize the binding set for methods. If the given
/// type is not a parametric user defined type, this returns empty bindings.
ParamBindings
ParamBindings::getForDeclaredType(const TypeCheckScopeInfo &scopeInfo,
                                  ASTType type, const ExprNode *expr) {
  ParamBindings paramBindings(scopeInfo);
  paramBindings.numCtadParams = type.getParamBindings().size();
  paramBindings.defaultTypeParams = type.getDefaultPosParams();

  // When binding a trait function, add the self type bindings.
  if (auto trait = dyn_cast_or_null<TraitType>(type.getMetaType()))
    paramBindings.addPrechecked(expr, PValue(type).get());

  ArrayRef<TypedAttr> paramValues = type.getParamBindings();
  for (TypedAttr value : paramValues)
    paramBindings.addPrechecked(expr, value);
  return paramBindings;
}

void ParamBindings::addPrechecked(const ExprNode *expr,
                                  TypedAttr precheckedBinding) {
  assert(numPreTypeChecked == posBindings.size() &&
         "Cannot add type prechecked after other bindings!");
  posBindings.push_back({precheckedBinding, expr});
  ++numPreTypeChecked;
}

void ParamBindings::add(const ExprNode *expr, TypedAttr value) {
  posBindings.push_back({value, expr});
}

void ParamBindings::add(const ExprNode *expr, PValue value, StringAttr name) {
  auto [_, addedNew] =
      kwBindings.try_emplace(name, ASTExprAnd<PValue>{value, expr});
  assert(addedNew && "duplicate keyword parameter");
}

//===----------------------------------------------------------------------===//
// verifyBindings
//===----------------------------------------------------------------------===//

/// Check a single binding and emit a parameter value if possible. If an
/// implicit conversion is required, the provided counter is incremented.
static PValue emitSingleParameterValue(ASTExprAnd<AnyValue> binding,
                                       ASTType expectedType,
                                       size_t &numImplicitConversions,
                                       ExprEmitter &emitter,
                                       ParserParamEvaluator &evaluator) {
  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

  PValue bindingVal = binding.ir.getIfPValue();
  assert(bindingVal && "Parameters are always PValue's");

  // We don't typecheck the '_' magic parameter, we propagate it.
  if (isa<UnboundAttr>(bindingVal.get()))
    return PValue(UnboundAttr::get(expectedType));

  // If the parameter already has the right type, then we're good.
  if (expectedType.isEqualCanon(bindingVal.getType()))
    return bindingVal;

  // If the parameter can be implicitly converted, do so.
  if (OverloadSet::canImplicitlyConvertToType(
          {bindingVal, binding.expr}, expectedType, emitter.getScopeInfo())) {
    numImplicitConversions += 2;
    return emitter.emitPValue(binding, EC_CallParamValue, expectedType);
  }
  return {};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindingsImpl(
    ArrayRef<Type> expectedParamTypes, PogListAttr paramListAttr,
    ParameterInferenceHookTy parameterInferenceHook,
    const DiagEmitter *diagEmitter, bool partial) const {
  size_t numParams = expectedParamTypes.size();

  // Handle *_ if it is present expanding posBindings into unpackedPosBindings.
  Fitness fitness{0, false};
  SmallVector<ASTExprAnd<AnyValue>> unpackedPosBindings;
  unpackedPosBindings.reserve(numParams);

  for (auto [idx, binding] : llvm::enumerate(posBindings)) {
    if (!isa<UnpackedAttr>(binding.ir.getIfPValue().get())) {
      unpackedPosBindings.push_back(binding);
      continue;
    }

    // UnpackedAttr is aka "*_": it fills up all remaining positional slots and
    // must be at the end of the parameter list.
    if (idx != posBindings.size() - 1) {
      if (diagEmitter)
        diagEmitter->emitUnboundPackNotEnd(binding);
      return {{}, fitness};
    }

    // *_ doesn't work with variadic parameter lists.
    // TODO: Why not? It should expand to a variadic on the enclosing context.
    if (paramListAttr.hasVariadic()) {
      if (diagEmitter)
        diagEmitter->emitUnboundPackInVariadic(binding);
      return {{}, fitness};
    }

    // Check if we have too many parameters after unpacking.
    size_t numPosPassable = countNumPositional(paramListAttr);
    size_t numUnpackedPositionals = unpackedPosBindings.size();
    if (unpackedPosBindings.size() > numPosPassable) {
      if (diagEmitter) {
        diagEmitter->emitTooManyPositional(numPosPassable,
                                           numUnpackedPositionals);
      }
      return {{}, fitness};
    }

    // If missing at least one positional parameter, we inject some number of
    // UnboundAttr's, just like the user wrote the right number of _'s.
    auto unboundAttr =
        UnboundAttr::get(UnresolvedType::get(shared.getContext()));
    ASTExprAnd<PValue> unboundBinding{PValue(unboundAttr), binding.expr};

    // Calculate how many UnboundAttr's (_'s) we need to inject, and put them
    // where the *_ was found.
    ssize_t numUnbounds = numPosPassable - numUnpackedPositionals;
    unpackedPosBindings.append(numUnbounds, unboundBinding);
    assert(unpackedPosBindings.size() == numPosPassable);
  }

  // Create a view of the operands for ease of access.
  OperandContainer operands(unpackedPosBindings, &kwBindings);

  KeywordOperandContainer variadicKwOperands;
  bool allowMissingKwOnly = partial || parameterInferenceHook;
  auto [kwDiagRes, kwDiagNames] = diagnoseKeywordOperands(
      paramListAttr, variadicKwOperands, operands, allowMissingKwOnly);
  if (kwDiagRes != KwDiagResult::kValid) {
    switch (kwDiagRes) {
    case KwDiagResult::kMissingKwOnly:
      if (diagEmitter)
        diagEmitter->emitMissing(kwDiagNames, "keyword-only");
      break;
    case KwDiagResult::kPosOnlyPassedByKw:
      if (diagEmitter)
        diagEmitter->emitPosOnlyPassedByKw(kwDiagNames);
      break;
    case KwDiagResult::kUnknownKeywords:
      if (diagEmitter)
        diagEmitter->emitUnknownKeywords(kwDiagNames);
      break;
    default:
      llvm_unreachable("unknown KwDiagResult");
    }
    return {{}, fitness};
  }

  auto [posDiagRes, posDiagNames] =
      diagnosePosOperands(paramListAttr, operands, /*allowCountMismatch=*/true);
  if (posDiagRes == PosDiagResult::kByPosAndKw) {
    if (diagEmitter)
      diagEmitter->emitRedundantKeywords(posDiagNames);
    return {{}, fitness};
  }

  // Parameter inference and call emission rely on this function not failing
  // early due to missing or too many positional parameters.
  assert(posDiagRes == PosDiagResult::kValid &&
         "positional parameter operand check failed unexpectedly");

  /// We will attempt to find a binding for every expected parameter.
  SmallVector<TypedAttr> newBindings;
  newBindings.reserve(numParams);

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParserParamEvaluator to keep track of the
  // mapping so far and remap types on demand.
  ParserParamEvaluator evaluator(*shared.declResolver);

  // This lambda installs the decl's value in the parameter evaluator and new
  // binding array.
  auto setParamValue = [&](TypedAttr value) {
    evaluator.addInputValue(value);
    newBindings.push_back(value);
  };

  // Use an expr emitter to perform implicit conversions within a parameter
  // context.
  ExprEmitter emitter(shared, declScope, EC_ParameterList);

  size_t posBindingIdx = 0;
  size_t numPosBindings = operands.posOperands.size();

  auto inferParameter = [&](Type requestedType) {
    if (parameterInferenceHook) {
      if (PValue value = parameterInferenceHook(newBindings, evaluator)) {
        assert(value.getType().mlirType == requestedType &&
               "inferred a parameter value of wrong type");
        return value;
      }
    }
    return PValue();
  };

  DefaultValueHandler defaultHandler(paramListAttr);
  auto fulfillValue = [&](Type requestedType) -> PValue {
    // If we have a method to infer parameter values, invoke it to see if we
    // can get an inferred value for the parameter.
    if (PValue value = inferParameter(requestedType))
      return value;

    // If the parameter decl is a variadic parameter list, and do not have
    // pack operands that could be used to infer those parameters, then we can
    // fulfill it with an empty list.  We know it must be the last parameter
    // decl. If this isn't actually a variadic type, then we simply reached
    // the end of the parameter list.
    size_t idx = newBindings.size();
    if (paramListAttr.isVariadic(idx))
      if (auto varType = dyn_cast<VariadicType>(requestedType))
        return VariadicAttr::get({}, varType);

    // If available, we use a default parameter value.
    if (TypedAttr defaultOr = defaultHandler.getDefault(idx)) {
      // Default parameter values may reference other parameter values, so we
      // need to evaluate these.
      return evaluator.getReboundAttribute(defaultOr);
    }

    // Determine if we can use a default parameter for CTAD.
    size_t defaultStartIdx = numCtadParams - defaultTypeParams.size();
    if (idx < numCtadParams && idx >= defaultStartIdx) {
      return cast<TypedAttr>(evaluator.getReboundAttribute(
          defaultTypeParams[idx - defaultStartIdx]));
    }

    return {};
  };

  for (auto [idx, sigType, pogAttr] :
       llvm::enumerate(expectedParamTypes, paramListAttr.getPogs())) {
    // This is the refined type expected by the signature.
    Type requestedType = evaluator.getReboundType(sigType);
    // This is the expected type of a value satisfying this parameter.
    ASTType expectedType = requestedType;
    // If this is a vararg parameter, infer using the element type.
    if (paramListAttr.isVariadic(idx))
      if (auto varType = dyn_cast<VariadicType>(expectedType))
        expectedType = ASTType(varType.getElementType());

    // Check to see if we ran out of bindings to provide to this param decl.
    // Implicit parameters are infer-only. They cannot be explicitly passed.
    PassingKind passingKind = pogAttr.getPassingKind();
    StringAttr paramName = pogAttr.getName();
    if (posBindingIdx == numPosBindings) {
      // We first check if we have a keyword parameter.
      if (std::optional<ASTExprAnd<AnyValue>> binding =
              operands.findKwArg(paramName)) {
        assert(passingKind != PassingKind::PosOnly);

        PValue pValue = emitSingleParameterValue(*binding, expectedType,
                                                 fitness.numImplicitConversions,
                                                 emitter, evaluator);
        if (!pValue) {
          if (diagEmitter)
            diagEmitter->emitKwType(paramName, *binding, expectedType);
          return {{}, fitness};
        }
        setParamValue(pValue);
        continue;
      }

      // If we couldn't find a keyword binding for this parameter, then we must
      // be able to infer it or otherwise provide a default value.
      if (PValue value = fulfillValue(requestedType)) {
        setParamValue(value);
        continue;
      }

      // If this is a partial binding context, then we don't have a full binding
      // list. Allow parameters to be missing.
      if (partial) {
        setParamValue(UnboundAttr::get(requestedType));
        continue;
      }

      // Otherwise, if this is a parameter that we expected to be inferred, emit
      // an inference failure.
      if (passingKind == PassingKind::Implicit ||
          passingKind == PassingKind::Inferred) {
        if (diagEmitter)
          diagEmitter->emitInferOnlyFailure(idx);
        return {{}, fitness};
      }

      if (passingKind == PassingKind::KwOnly) {
        // If this is a missing keyword-only, we collect them. We put pretend
        // this is implicitly unbound, so we can error out in the end.
        setParamValue(UnboundAttr::get(requestedType));
        kwDiagNames.push_back(paramName);
        continue;
      }

      if (!fitness.lastExpectedType)
        fitness.lastExpectedType = expectedType;

      if (diagEmitter)
        diagEmitter->emitDeductionFailure(idx);
      return {{}, fitness};
    }

    // If we still have positional bindings left, first check if we are dealing
    // with an UnboundAttr we might have to deduce.
    ASTExprAnd<AnyValue> binding = operands.posOperands[posBindingIdx];
    PValue bindingVal = binding.ir.getIfPValue();
    assert(bindingVal && "Parameters are always PValues");
    if (!partial && isa<UnboundAttr>(bindingVal.get())) {
      if (PValue value = fulfillValue(requestedType)) {
        setParamValue(value);
        ++posBindingIdx;
        continue;
      }
      // We tried but couldn't infer an unbound parameter, we must error.
      if (diagEmitter)
        diagEmitter->emitDeductionFailure(idx);
      return {{}, fitness};
    }

    // If this value was already bound and checked, use it.
    /// FIXME: Remove this, why is this needed?
    if (posBindingIdx < numPreTypeChecked) {
      setParamValue(bindingVal);
      ++posBindingIdx;
      continue;
    }

    // Disallow implicit parameters to be explicit specified. If we see one,
    // complain about too many parameters.
    if (passingKind == PassingKind::Implicit) {
      if (diagEmitter) {
        diagEmitter->emitParamCount(numPosBindings,
                                    passingKind == PassingKind::PosOnly);
      }
      return {{}, fitness};
    }

    // Otherwise, if this is an inferred parameter, a value could not have been
    // explicitly provided and we must have an inference hook.
    if (passingKind == PassingKind::Inferred) {
      // TODO: Enable this assert. We always need to be able to infer these.
      // assert(parameterInferenceHook &&
      //        "require parmeter inference in this context");
      if (PValue value = inferParameter(requestedType)) {
        setParamValue(value);
        continue;
      }
      // If this context allows partial binding, leave the value as unbound.
      if (partial) {
        setParamValue(UnboundAttr::get(requestedType));
        continue;
      }
      // Otherwise, emit an inference failure.
      if (diagEmitter)
        diagEmitter->emitInferOnlyFailure(idx);
      return {{}, fitness};
    }

    // This lambda hides the diagnostic and error handling logic for checking a
    // single positional parameter binding.
    auto handlePosBinding = [&, &kwDiagNames = kwDiagNames](
                                size_t index, ASTExprAnd<AnyValue> binding,
                                ASTType expectedType) -> PValue {
      if (passingKind == PassingKind::KwOnly) {
        // If this is a keyword-only passed positionally, we remember it.
        kwDiagNames.push_back(paramName);
        return UnboundAttr::get(expectedType);
      }

      PValue pValue = emitSingleParameterValue(binding, expectedType,
                                               fitness.numImplicitConversions,
                                               emitter, evaluator);
      if (!pValue)
        if (diagEmitter)
          diagEmitter->emitPosType(index, binding, expectedType);
      return pValue;
    };

    // Scalar parameter values are installed directly. Or, if we have a variadic
    // of the same type, we can use it as the value of the parameter directly.
    // FIXME: This allows passing a variadic `Ts` directly. Do we want a new
    // PValue classification for `*Ts`, which is required to pass this legally?
    if (!paramListAttr.isVariadic(idx) ||
        bindingVal.getType().isEqualCanon(requestedType)) {
      PValue paramValue = handlePosBinding(idx, binding, requestedType);
      if (!paramValue)
        return {{}, fitness};
      setParamValue(paramValue);
      ++posBindingIdx;
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    fitness.hasVariadicParams = true;
    SmallVector<TypedAttr> elements;
    auto variadicType = cast<VariadicType>(requestedType);
    do {
      ASTExprAnd<AnyValue> binding = unpackedPosBindings[posBindingIdx++];
      PValue pValue = handlePosBinding(idx, binding, expectedType);
      if (!pValue)
        return {{}, fitness};
      elements.emplace_back(pValue);
    } while (posBindingIdx != numPosBindings);

    auto varType = VariadicType::get(evaluator.getReboundType(expectedType),
                                     variadicType.getConvention());
    setParamValue(VariadicAttr::get(elements, varType));
  }

  // Complain if we collected any missing keyword-only parameters.
  if (!kwDiagNames.empty()) {
    if (diagEmitter)
      diagEmitter->emitMissing(kwDiagNames, "keyword-only");
    return {{}, fitness};
  }

  // Check and complain if we have bindings that didn't get used.
  if (posBindingIdx != numPosBindings) {
    if (diagEmitter)
      diagEmitter->emitParamCount(numPosBindings, /*posOnly=*/false);
    return {{}, fitness};
  }

  return {ParameterExprArrayAttr::get(emitter.getContext(), newBindings),
          fitness};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(LITSignatureType sig,
                              const DiagEmitter *diagEmitter,
                              ParameterInferenceHookTy parameterInferenceHook,
                              bool partial) const {
  return verifyBindingsImpl(sig.getParamTypes(), sig.getParamListAttrs(),
                            parameterInferenceHook, diagEmitter, partial);
}

ParameterExprArrayAttr ParamBindings::verifyBindings(StructDeclOp structOp,
                                                     TypeSignatureType sig,
                                                     SMLoc exprLoc,
                                                     bool partial) const {
  auto [bindingValuesAttr, _, diag] =
      verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                     Twine("'") + structOp.getName() + "'", exprLoc,
                     structOp.getLoc(), partial);
  return bindingValuesAttr;
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(LITSignatureType sig, StringRef baseName,
                              SMLoc exprLoc,
                              std::optional<Location> opLoc) const {
  auto [newBindings, _, diag] =
      verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                     opLoc ? Twine("'") + baseName + "'" : Twine(baseName),
                     exprLoc, opLoc, /*partial=*/true);
  return newBindings;
}

std::tuple<ParameterExprArrayAttr, ParamBindings::Fitness,
           std::optional<InflightDiag>>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              PogListAttr paramListAttr, const Twine &baseName,
                              SMLoc exprLoc, std::optional<Location> opLoc,
                              bool partial) const {
  size_t maxAllowed = expectedParamTypes.size() -
                      countNumImplicitKinds(paramListAttr) -
                      countNumInferredKinds(paramListAttr);
  ParameterInferenceDiagnostics inferenceDiags;
  std::optional<InflightDiag> diag;
  DiagEmitter diagEmitter{
      /*emitParamCount=*/[&](size_t numActual, bool posOnly) {
        diag = shared.emitError(exprLoc, baseName);
        if (posOnly) {
          emitWrongArgOrParamCount(
              *diag, /*minRequired=*/countNumPosOnly(paramListAttr), maxAllowed,
              numActual, "positional parameter");
        } else {
          size_t minRequired = expectedParamTypes.size() -
                               paramListAttr.getDefaultPos().size() -
                               paramListAttr.getDefaultKwOnly().size();
          emitWrongArgOrParamCount(*diag, minRequired, maxAllowed, numActual,
                                   "parameter");
        }
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
        inferenceDiags.attach(paramListAttr, *diag, numActual);
      },
      /*emitPosType=*/
      [&](size_t index, ASTExprAnd<AnyValue> binding, ASTType expectedType) {
        PValue paramVal = binding.ir.getIfPValue();
        diag = shared.emitError(binding.expr->getLoc(), baseName)
               << " parameter #" << index << " has " << expectedType
               << " type, but value has type " << paramVal.getType()
               << binding.expr->getRange();
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitKwType=*/
      [&](StringAttr paramName, ASTExprAnd<AnyValue> binding,
          ASTType expectedType) {
        PValue paramVal = binding.ir.getIfPValue();
        diag = shared.emitError(binding.expr->getLoc(), baseName)
               << " parameter " << paramName << " has " << expectedType
               << " type, but value has type " << paramVal.getType()
               << binding.expr->getRange();
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnknownKeywords=*/
      [&](ArrayRef<StringAttr> unknownKeywords) {
        diag = shared.emitError(exprLoc);
        emitUnknownKeywords(*diag, unknownKeywords, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitRedundantKeywords=*/
      [&](ArrayRef<StringAttr> names) {
        diag = shared.emitError(exprLoc);
        emitByPosAndKw(*diag, names, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosOnlyPassedByKw=*/
      [&](ArrayRef<StringAttr> names) {
        diag = shared.emitError(exprLoc);
        emitPosOnlyPassedByKw(*diag, names, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        assert(!partial && "parameter deduction failure in a context that "
                           "doesn't allow deduction");
        if (paramListAttr.getPassingKind(paramIdx) != PassingKind::Inferred) {
          diag = shared.emitError(exprLoc, baseName) << " missing required ";
          if (StringAttr name = paramListAttr.getName(paramIdx); !name.empty())
            *diag << "parameter " << name;
          else
            *diag << nameForPosOnly(paramIdx, "parameter");
        } else {
          diag = shared.emitError(exprLoc) << "failed to infer parameter ";
          printNameOrIdx(paramListAttr.getName(paramIdx), paramIdx, *diag);
          inferenceDiags.attach(paramListAttr, *diag);
        }
      },
      /*emitUnboundPackInVariadic=*/
      [&](ASTExprAnd<AnyValue> binding) {
        diag = shared.emitError(binding.expr->getLoc());
        *diag << "unbound pack syntax cannot be used where variadic parameters "
                 "are expected";
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnboundPackNotEnd=*/
      [&](ASTExprAnd<AnyValue> binding) {
        diag = shared.emitError(binding.expr->getLoc());
        *diag << "unbound pack must be at the end of the parameter list";
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        diag = shared.emitError(exprLoc) << "failed to infer parameter ";
        printNameOrIdx(paramListAttr.getName(paramIdx), paramIdx, *diag);
        inferenceDiags.attach(paramListAttr, *diag);
      },
      /*emitMissing=*/
      [&](ArrayRef<StringAttr> names, const Twine &kindStr) {
        diag = shared.emitError(exprLoc);
        emitMissing(*diag, names, kindStr + " parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitTooManyPositional=*/
      [&](size_t numMaxAllowed, size_t numActual) {
        diag = shared.emitError(exprLoc);
        emitTooManyPositional(*diag, numMaxAllowed, numActual, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      }};

  SyntheticNode errorLoc(exprLoc);
  auto parameterInferenceHook = [&](ArrayRef<TypedAttr> bindingsSoFar,
                                    const ParserParamEvaluator &evaluator) {
    ParameterInferenceState inferrence(*this, getPosBindings(),
                                       &getKWBindings(), bindingsSoFar,
                                       evaluator, inferenceDiags,
                                       /*allowImplicitConversions=*/true);

    // Infer information from the current parameter list.
    inferrence.infer(expectedParamTypes, paramListAttr);

    // See if we inferred information about the next value.
    if (auto result = inferrence.getInferredValue(bindingsSoFar.size()))
      return PValue(result);

    // If we succeeded inference but didn't get a value for this parameter, then
    // the parameter must not be present: complain.
    inferenceDiags.addFailure(bindingsSoFar.size(), errorLoc,
                              InferenceFailure::NotFoundFailure());
    return PValue();
  };
  auto [bindings, fitness] =
      verifyBindingsImpl(expectedParamTypes, paramListAttr,
                         parameterInferenceHook, &diagEmitter, partial);
  return {bindings, fitness, std::move(diag)};
}

TypedAttr ParamBindings::getBoundConstAttrFor(LIT::FuncOp funcOp,
                                              StringRef baseName,
                                              const ExprNode *expr) const {
  LITSignatureType signature = funcOp.getFullSignature();

  // If this is a global function or struct reference, bind it directly.
  auto parentTrait = dyn_cast<TraitDeclOp>(funcOp->getParentOp());
  if (!parentTrait) {
    // If there are no parameters specified and if we allow unbound symbols,
    // just return the unbound symbol.
    if (empty())
      return funcOp.getBoundReference();

    // Check that the signature can be rebound with our set of bindings.
    ParameterExprArrayAttr newBindings =
        verifyBindings(signature, baseName, expr->getLoc(), funcOp.getLoc());
    if (!newBindings)
      return {};

    // Now that we checked the types match, form the binding.
    return funcOp.getBoundReference(newBindings);
  }

  // The first parameter to the fully bound signature will be the type (confined
  // to the current trait type) that ultimately expands to the concrete type
  // that conforms to the trait.
  assert(!posBindings.empty());
  PValue selfExpr = posBindings.front().ir.getIfPValue();
  assert(selfExpr && "parameters are always PValues");

  // When referencing a trait function, bind the reference using a parameter
  // expression instead of the direct reference. Also, drop the implicit trait
  // parameter.
  ParamBindings bindings = *this;
  SmallVector<TypedAttr> paramValues;
  paramValues.push_back(selfExpr);

  auto it = bindings.posBindings.begin();
  bindings.posBindings.erase(it, it + 1);
  for (Type type : signature.getParamTypes().drop_front())
    paramValues.push_back(UnboundAttr::get(type));
  signature = signature.getSpecializedSignature(paramValues, [&]() {
    return mlir::emitError(shared.translateLocation(expr->getLoc()))
           << "internal error: ";
  });
  assert(signature && "Error binding trait Self type");

  TypedAttr fnRef = ParamOperatorAttr::get(
      POC::GetTypeMethod,
      {PValue(selfExpr),
       StringAttr::get(baseName, StringType::get(funcOp.getContext()))},
      signature);
  if (bindings.empty())
    return fnRef;

  // Attempt to partially bind the parameters to the signature of the function.
  ParameterExprArrayAttr newBindings = bindings.verifyBindings(
      signature, baseName, expr->getLoc(), funcOp.getLoc());
  if (!newBindings)
    return {};

  SmallVector<TypedAttr> bindSigOperands;
  bindSigOperands.push_back(fnRef);
  llvm::append_range(bindSigOperands, newBindings);
  return ParamOperatorAttr::get(POC::BindSignature, bindSigOperands);
}

void ParamBindings::dump() const {
  auto &os = llvm::errs();
  os << "Positional bindings:\n";
  for (auto [i, binding] : llvm::enumerate(posBindings)) {
    os << "  " << i << "[" << (i < numPreTypeChecked)
       << "]: " << binding.ir.getIfPValue().get() << "\n";
  }
  os << "Keyword bindings:\n";
  for (auto [name, binding] : kwBindings)
    os << "  " << name.getValue() << ": " << binding.ir.getIfPValue() << "\n";
}
