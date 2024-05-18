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
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/IRValues.h"
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
}

/// Create a (possibly partially unbound) set of bindings for the given type.
/// This can be used to initialize the binding set for methods. If the given
/// type is not a parametric user defined type, this returns empty bindings.
ParamBindings
ParamBindings::getForDeclaredType(const TypeCheckScopeInfo &scopeInfo,
                                  ASTType type) {
  ParamBindings paramBindings(scopeInfo);
  ArrayRef<Type> params = type.getParameters();
  paramBindings.numCtadParams = params.size();
  paramBindings.defaultTypeParams = type.getDefaultPosParams();

  // When binding a trait function, add the self type bindings.
  if (auto trait = dyn_cast_or_null<TraitType>(type.getMetaType()))
    paramBindings.addPrechecked(PValue(type).get());

  ArrayRef<TypedAttr> paramValues = type.getParamBindings();
  for (TypedAttr value : paramValues)
    paramBindings.addPrechecked(value);
  return paramBindings;
}

void ParamBindings::addPrechecked(TypedAttr precheckedBinding) {
  posBindings.push_back({nullptr, precheckedBinding, /*typeChecked=*/true});
}

void ParamBindings::addPrechecked(TypedAttr precheckedBinding,
                                  StringAttr name) {
  auto [_, addedNew] = kwBindings.try_emplace(
      name, Binding{nullptr, precheckedBinding, /*typeChecked=*/true});
  assert(addedNew && "duplicate keyword parameter");
}

void ParamBindings::add(const ExprNode *expr, TypedAttr value) {
  posBindings.push_back({expr, value, /*typeChecked=*/false});
}

void ParamBindings::add(const ExprNode *expr, TypedAttr value,
                        StringAttr name) {
  auto [_, addedNew] =
      kwBindings.try_emplace(name, Binding{expr, value, /*typeChecked=*/false});
  assert(addedNew && "duplicate keyword parameter");
}

//===----------------------------------------------------------------------===//
// verifyBindings
//===----------------------------------------------------------------------===//

/// Check a single binding and emit a parameter value if possible. If an
/// implicit conversion is required, the provided counter is incremented.
static PValue emitSingleParameterValue(ParamBindings::Binding binding,
                                       ASTType expectedType,
                                       size_t &numImplicitConversions,
                                       ExprEmitter &emitter,
                                       ParserParamEvaluator &evaluator) {
  assert(binding.expr &&
         "should always have an expr tree for unchecked bindings");

  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

  // We don't typecheck the '_' magic parameter, we propagate it.
  if (isa<UnboundAttr>(binding.value))
    return PValue(UnboundAttr::get(expectedType));

  // If the parameter already has the right type, then we're good.
  PValue bindingPVal(binding.getValue());
  if (expectedType.isEqualCanon(bindingPVal.getType()))
    return bindingPVal;

  // If the parameter can be implicitly converted, do so.
  if (emitter.canImplicitlyConvertToType({bindingPVal, binding.expr},
                                         expectedType)) {
    numImplicitConversions += 2;
    return emitter.emitPValue({bindingPVal, binding.expr}, EC_CallParamValue,
                              expectedType);
  }
  return {};
};

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              PogListAttr paramListAttr,
                              ParameterInferenceHookTy parameterInferenceHook,
                              const DiagEmitter &diagEmitter,
                              Boundness boundness) const {
  size_t numParams = expectedParamTypes.size();

  // Handle *_ if it is present expanding posBindings into unpackedPosBindings.
  Fitness fitness{0, false};
  SmallVector<Binding> unpackedPosBindings;
  unpackedPosBindings.reserve(numParams);

  for (auto [idx, binding] : llvm::enumerate(posBindings)) {
    if (!isa<UnpackedAttr>(binding.value)) {
      unpackedPosBindings.push_back(binding);
      continue;
    }

    // UnpackedAttr is aka "*_": it fills up all remaining positional slots and
    // must be at the end of the parameter list.
    if (idx != posBindings.size() - 1) {
      if (diagEmitter.emitUnboundPackNotEnd)
        diagEmitter.emitUnboundPackNotEnd(binding);
      return {{}, fitness};
    }

    // *_ doesn't work with variadic parameter lists.
    // TODO: Why not? It should expand to a variadic on the enclosing context.
    if (paramListAttr.hasVariadic()) {
      if (diagEmitter.emitUnboundPackInVariadic)
        diagEmitter.emitUnboundPackInVariadic(binding);
      return {{}, fitness};
    }

    // Check if we have too many parameters after unpacking.
    size_t numPosPassable = countNumPositional(paramListAttr);
    size_t numUnpackedPositionals = unpackedPosBindings.size();
    if (unpackedPosBindings.size() > numPosPassable) {
      if (diagEmitter.emitTooManyPositional) {
        diagEmitter.emitTooManyPositional(numPosPassable,
                                          numUnpackedPositionals);
      }
      return {{}, fitness};
    }

    // If missing at least one positional parameter, we inject some number of
    // UnboundAttr's, just like the user wrote the right number of _'s.
    auto unboundAttr =
        UnboundAttr::get(UnresolvedType::get(shared.getContext()));
    Binding unboundBinding{binding.expr, unboundAttr, binding.typeChecked};

    // Calculate how many UnboundAttr's (_'s) we need to inject, and put them
    // where the *_ was found.
    ssize_t numUnbounds = numPosPassable - numUnpackedPositionals;
    unpackedPosBindings.append(numUnbounds, unboundBinding);
    assert(unpackedPosBindings.size() == numPosPassable);
  }

  // Create a view of the operands for ease of access.
  OperandContainer<Binding> operands(unpackedPosBindings, &kwBindings);

  KeywordOperandContainer<Binding> variadicKwOperands;
  bool allowMissingKwOnly =
      boundness == Boundness::Partial || parameterInferenceHook;
  auto [kwDiagRes, kwDiagNames] = diagnoseKeywordOperands(
      paramListAttr, variadicKwOperands, operands, allowMissingKwOnly);
  if (kwDiagRes != KwDiagResult::kValid) {
    switch (kwDiagRes) {
    case KwDiagResult::kMissingKwOnly:
      if (diagEmitter.emitMissing)
        diagEmitter.emitMissing(kwDiagNames, "keyword-only");
      break;
    case KwDiagResult::kPosOnlyPassedByKw:
      if (diagEmitter.emitPosOnlyPassedByKw)
        diagEmitter.emitPosOnlyPassedByKw(kwDiagNames);
      break;
    case KwDiagResult::kUnknownKeywords:
      if (diagEmitter.emitUnknownKeywords)
        diagEmitter.emitUnknownKeywords(kwDiagNames);
      break;
    default:
      llvm_unreachable("unknown KwDiagResult");
    }
    return {{}, fitness};
  }

  auto [posDiagRes, posDiagNames] =
      diagnosePosOperands(paramListAttr, operands, /*allowCountMismatch=*/true);
  if (posDiagRes == PosDiagResult::kByPosAndKw) {
    if (diagEmitter.emitRedundantKeywords)
      diagEmitter.emitRedundantKeywords(posDiagNames);
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

  DefaultValueHandler defaultHandler(paramListAttr);
  auto fulfillValue = [&](Type requestedType) -> PValue {
    // If we have a method to infer parameter values, invoke it to see if we
    // can get an inferred value for the parameter.
    if (parameterInferenceHook) {
      if (PValue value = parameterInferenceHook(newBindings, evaluator)) {
        assert(value.getType().mlirType == requestedType &&
               "inferred a parameter value of wrong type");
        return value;
      }
    }

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
    if (posBindingIdx == numPosBindings ||
        (parameterInferenceHook && (passingKind == PassingKind::Implicit ||
                                    passingKind == PassingKind::Inferred))) {
      // We first check if we have a keyword parameter.
      if (std::optional<Binding> binding = operands.findKwArg(paramName)) {
        assert(passingKind != PassingKind::PosOnly);

        // If this value was already bound and checked, use it.
        if (binding->typeChecked) {
          setParamValue(binding->value);
          continue;
        }

        PValue pValue = emitSingleParameterValue(*binding, expectedType,
                                                 fitness.numImplicitConversions,
                                                 emitter, evaluator);
        if (!pValue) {
          if (diagEmitter.emitKwType)
            diagEmitter.emitKwType(paramName, *binding, expectedType);
          return {{}, fitness};
        }
        setParamValue(pValue);
        continue;
      }

      if (PValue value = fulfillValue(requestedType)) {
        setParamValue(value);
        continue;
      }

      if (boundness == Boundness::Partial) {
        setParamValue(UnboundAttr::get(requestedType));
        continue;
      }

      if (passingKind == PassingKind::Implicit ||
          passingKind == PassingKind::Inferred) {
        if (diagEmitter.emitInferOnlyFailure)
          diagEmitter.emitInferOnlyFailure(idx);
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

      if (diagEmitter.emitParamCount) {
        diagEmitter.emitParamCount(numPosBindings,
                                   passingKind == PassingKind::PosOnly);
      }
      return {{}, fitness};
    }

    // If we still have positional bindings left, first check if we are dealing
    // with an UnboundAttr we might have to deduce.
    const Binding &binding = operands.posOperands[posBindingIdx];
    if (isa<UnboundAttr>(binding.value)) {
      if (boundness == Boundness::Full) {
        if (PValue value = fulfillValue(requestedType)) {
          setParamValue(value);
          ++posBindingIdx;
          continue;
        }

        // We tried but couldn't infer an unbound parameter, we must error.
        if (diagEmitter.emitDeductionFailure)
          diagEmitter.emitDeductionFailure(idx);
        return {{}, fitness};
      }
    }

    // If this value was already bound and checked, use it.
    if (binding.typeChecked) {
      setParamValue(binding.value);
      ++posBindingIdx;
      continue;
    }

    // This lambda hides the diagnostic and error handling logic for checking a
    // single positional parameter binding.
    auto handlePosBinding =
        [&, &kwDiagNames = kwDiagNames](size_t index, const Binding &binding,
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
        if (diagEmitter.emitPosType)
          diagEmitter.emitPosType(index, binding, expectedType);
      return pValue;
    };

    // Scalar parameter values are installed directly. Or, if we have a variadic
    // of the same type, we can use it as the value of the parameter directly.
    // FIXME: This allows passing a variadic `Ts` directly. Do we want a new
    // PValue classification for `*Ts`, which is required to pass this legally?
    if (!paramListAttr.isVariadic(idx) ||
        binding.getValue().getType() == requestedType) {
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
      const Binding &binding = unpackedPosBindings[posBindingIdx++];
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
    if (diagEmitter.emitMissing)
      diagEmitter.emitMissing(kwDiagNames, "keyword-only");
    return {{}, fitness};
  }

  // Check and complain if we have bindings that didn't get used.
  if (posBindingIdx != numPosBindings) {
    if (diagEmitter.emitParamCount)
      diagEmitter.emitParamCount(numPosBindings, /*posOnly=*/false);
    return {{}, fitness};
  }

  return {ParameterExprArrayAttr::get(emitter.getContext(), newBindings),
          fitness};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              PogListAttr paramListAttr, const Twine &baseName,
                              llvm::SMLoc exprLoc,
                              std::optional<Location> opLoc,
                              Boundness boundness) const {
  size_t maxAllowed = expectedParamTypes.size() -
                      countNumImplicitKinds(paramListAttr) -
                      countNumInferredKinds(paramListAttr);
  DiagEmitter diagEmitter{
      /*emitParamCount=*/[&](size_t numActual, bool posOnly) {
        InflightDiag diag = shared.emitError(exprLoc, baseName);
        if (posOnly) {
          emitWrongArgOrParamCount(
              diag, /*minRequired=*/countNumPosOnly(paramListAttr), maxAllowed,
              numActual, "positional parameter");
        } else {
          size_t minRequired = expectedParamTypes.size() -
                               paramListAttr.getDefaultPos().size() -
                               paramListAttr.getDefaultKwOnly().size();
          emitWrongArgOrParamCount(diag, minRequired, maxAllowed, numActual,
                                   "parameter");
        }
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosType=*/
      [&](size_t index, const Binding &binding, ASTType expectedType) {
        auto diag = shared.emitError(binding.expr->getLoc(), baseName)
                    << " parameter #" << index << " has " << expectedType
                    << " type, but value has type "
                    << ASTType(binding.getValue().getType())
                    << binding.expr->getRange();
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitKwType=*/
      [&](StringAttr paramName, const Binding &binding, ASTType expectedType) {
        auto diag = shared.emitError(binding.expr->getLoc(), baseName)
                    << " parameter " << paramName << " has " << expectedType
                    << " type, but value has type "
                    << ASTType(binding.getValue().getType())
                    << binding.expr->getRange();
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnknownKeywords=*/
      [&](ArrayRef<StringAttr> unknownKeywords) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitUnknownKeywords(diag, unknownKeywords, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitRedundantKeywords=*/
      [&](ArrayRef<StringAttr> names) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitByPosAndKw(diag, names, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosOnlyPassedByKw=*/
      [&](ArrayRef<StringAttr> names) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitPosOnlyPassedByKw(diag, names, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        assert(boundness == Boundness::Full &&
               "parameter deduction failure in a context that doesn't allow "
               "deduction");
        InflightDiag diag = shared.emitError(exprLoc, baseName)
                            << " missing required ";
        if (StringAttr name = paramListAttr.getName(paramIdx); !name.empty())
          diag << "parameter " << name;
        else
          diag << nameForPosOnly(paramIdx, "parameter");
      },
      /*emitUnboundPackInVariadic=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "unbound pack syntax cannot be used where variadic parameters "
                "are expected";
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnboundPackNotEnd=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "unbound pack must be at the end of the parameter list";
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        llvm_unreachable("parameter deduction failure in a context that "
                         "doesn't allow deduction");
      },
      /*emitMissing=*/
      [&](ArrayRef<StringAttr> names, const Twine &kindStr) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitMissing(diag, names, kindStr + " parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitTooManyPositional=*/
      [&](size_t numMaxAllowed, size_t numActual) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitTooManyPositional(diag, numMaxAllowed, numActual, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      }};

  return verifyBindings(expectedParamTypes, paramListAttr,
                        /*parameterInferenceHook=*/{}, diagEmitter, boundness);
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(
    LITSignatureType sig, const DiagEmitter &diagEmitter,
    ParameterInferenceHookTy parameterInferenceHook) const {
  return verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                        parameterInferenceHook, diagEmitter, Boundness::Full);
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(LITSignatureType sig) const {
  DiagEmitter diagEmitter{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  return verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                        /*parameterInferenceHook=*/{}, diagEmitter);
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(StructDeclOp structOp, TypeSignatureType sig,
                              llvm::SMLoc exprLoc, Boundness boundness) const {
  auto [bindingValuesAttr, _] =
      verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                     Twine("'") + structOp.getName() + "'", exprLoc,
                     structOp.getLoc(), boundness);
  return bindingValuesAttr;
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(LITSignatureType sig, StringRef baseName,
                              llvm::SMLoc exprLoc,
                              std::optional<Location> opLoc) const {
  auto [newBindings, _] = verifyBindings(
      sig.getParamTypes(), sig.getParamListAttrs(),
      opLoc ? Twine("'") + baseName + "'" : Twine(baseName), exprLoc, opLoc,
      opLoc ? Boundness::Partial : Boundness::Explicit);
  return newBindings;
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
  PValue selfExpr = posBindings.front().value;

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
  llvm::dbgs() << "Positional bindings:\n";
  for (auto [i, binding] : llvm::enumerate(posBindings)) {
    llvm::dbgs() << "  " << i << "[" << binding.typeChecked
                 << "]: " << binding.value << "\n";
  }
  llvm::dbgs() << "Kewword bindings:\n";
  for (auto [name, binding] : kwBindings) {
    llvm::dbgs() << "  " << name.getValue() << "[" << binding.typeChecked
                 << "]: " << binding.value << "\n";
  }
}
